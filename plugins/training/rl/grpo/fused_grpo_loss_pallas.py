from __future__ import annotations

import os
from typing import Any


def _compute_fused_available() -> bool:
    try:
        from jax.experimental import pallas as _pl  # noqa: F401
        from jax.experimental.pallas import tpu as _pltpu  # noqa: F401
    except ImportError:
        return False
    return True


# True when JAX + Pallas are importable. On CPU/GPU runs we can still use the
# kernel in interpret-mode for debugging.
FUSED_AVAILABLE = _compute_fused_available()


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _select_block_t() -> int:
    if not FUSED_AVAILABLE:
        return 128
    try:
        import jax
    except Exception:
        return 128
    try:
        device_kind = (jax.devices()[0].device_kind or "").lower()
    except Exception:
        return 128
    return 1024 if "v6" in device_kind else 128


def _select_block_v() -> int:
    """Select vocab tile size (multiple of 128)."""

    if not FUSED_AVAILABLE:
        return 128
    try:
        import jax
    except Exception:
        return 128
    try:
        device_kind = (jax.devices()[0].device_kind or "").lower()
    except Exception:
        return 128
    if "v6" in device_kind:
        # On v6/v6e, use a larger vocab tile to reduce the number of per-token
        # reduction steps (and vocab blocks) while staying aligned to 128.
        return 1024
    if "v4" in device_kind:
        return 4096
    return 128


# Kernel tile sizes.
BLOCK_T = _select_block_t()
BLOCK_V = _select_block_v()


def _is_tpu_runtime() -> bool:
    if not FUSED_AVAILABLE:
        return False
    import jax

    return any(d.platform == "tpu" for d in jax.devices())


def _parse_env_flag(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in ("1", "true", "t", "yes", "y", "on"):
        return True
    if lowered in ("0", "false", "f", "no", "n", "off"):
        return False
    raise ValueError(f"Invalid boolean env value: {value!r}")


def grpo_fused_enabled(*, default: bool | None = None) -> bool:
    """Return whether the fused GRPO kernel should be used.

    Env override:
      - MLLM_JAX_GRPO_FUSED=1 enables fused
      - MLLM_JAX_GRPO_FUSED=0 disables fused

    When unset, defaults to TPU-only (safe) if Pallas is importable.
    """

    if not FUSED_AVAILABLE:
        return False

    env = os.environ.get("MLLM_JAX_GRPO_FUSED")
    if env is not None and env.strip() != "":
        return _parse_env_flag(env)

    if default is None:
        default = _is_tpu_runtime()
    return bool(default)


def _grpo_logp_entropy_jax(
    *,
    logits: Any,  # [B, L+1, V]
    completion_ids: Any,  # [B, L]
    temperature: float,
) -> tuple[Any, Any]:
    """Pure-JAX logp + entropy without materializing full-vocab log_softmax/softmax."""

    import jax
    import jax.numpy as jnp

    logits = jnp.asarray(logits)
    completion_ids = jnp.asarray(completion_ids, dtype=jnp.int32)

    seq_len = int(logits.shape[1]) - 1
    scaled = logits[:, :seq_len, :].astype(jnp.float32) / float(temperature)
    token_logits = jnp.take_along_axis(scaled, completion_ids[:, :, None], axis=-1)[:, :, 0]
    lse = jax.nn.logsumexp(scaled, axis=-1)
    logp = token_logits - lse

    # Entropy = logsumexp(z) - E_p[z], where p = softmax(z), z = logits/temp.
    expected_z = jnp.sum(jnp.exp(scaled - lse[:, :, None]) * scaled, axis=-1)
    entropy = lse - expected_z
    return logp, entropy


def _ppo_loss_from_logp(
    *,
    logp: Any,  # [B, L]
    old_logp: Any,  # [B, L]
    ref_logp: Any,  # [B, L]
    advantages: Any,  # [B]
    completion_mask: Any,  # [B, L] i32
    beta: float,
    eps_low: float,
    eps_high: float,
) -> Any:
    import jax.numpy as jnp

    logp = jnp.asarray(logp, dtype=jnp.float32)
    old = jnp.asarray(old_logp, dtype=jnp.float32)
    ref = jnp.asarray(ref_logp, dtype=jnp.float32)
    adv = jnp.asarray(advantages, dtype=jnp.float32)[:, None]

    ratio = jnp.exp(logp - old)
    clipped_ratio = jnp.clip(ratio, 1.0 - float(eps_low), 1.0 + float(eps_high))
    per_token_loss = -jnp.minimum(ratio * adv, clipped_ratio * adv)

    if beta != 0.0:
        delta = ref - logp
        kl = jnp.exp(delta) - delta - 1.0
        per_token_loss = per_token_loss + float(beta) * kl

    keep = jnp.asarray(completion_mask, dtype=jnp.int32) != 0
    return jnp.where(keep, per_token_loss, 0.0)


def _grpo_fused_logsumexp_stats_pallas(
    *,
    logits: Any,  # [B, L+1, V]
    temperature: float,
    interpret: bool,
) -> tuple[Any, Any, Any]:
    """Return per-token logsumexp state and weighted sum for entropy.

    Returns (m_flat, l_flat, s_flat) flattened token-major as [B*L] each, where:
      - m is max(z) for z = logits/temp
      - l is sum(exp(z - m))
      - s is sum(exp(z - m) * z)
    """

    if not FUSED_AVAILABLE:
        raise RuntimeError("Pallas fused path requires JAX + Pallas")

    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    logits = jnp.asarray(logits)

    bsz, l_add_1, vocab_size = logits.shape
    seq_len = l_add_1 - 1

    logits_tokens = logits[:, :seq_len, :]
    num_tokens = bsz * seq_len
    logits_flat = logits_tokens.reshape((num_tokens, vocab_size))

    token_blocks = _ceil_div(num_tokens, BLOCK_T)
    vocab_blocks = _ceil_div(vocab_size, BLOCK_V)

    device_kind = ""
    try:
        import jax as _jax  # Avoid capturing jax in the kernel closure.

        device_kind = (_jax.devices()[0].device_kind or "").lower()
    except Exception:
        device_kind = ""

    # On TPU v6/v6e, prefer parallel vocab blocks to avoid the sequential
    # ("arbitrary") streaming reduction over the vocab axis. We compute per-vocab
    # block stats (m, l, s) and then reduce across vocab blocks in JAX.
    use_parallel_vocab = _is_tpu_runtime() and ("v6" in device_kind) and ("v4" not in device_kind)

    if use_parallel_vocab:
        # TPU v6/v6e BlockSpec constraints require the *last two* dimensions of
        # output block shapes to be divisible by 8 and 128 respectively. A
        # natural `(num_tokens, vocab_blocks)` output with block shape
        # `(BLOCK_T, 1)` violates this constraint on v6/v6e. We also want to
        # avoid emitting huge `(vocab_blocks_inner, num_tokens)` intermediates.
        #
        # Instead, we emit *group-level* stats where each group covers
        # `GROUP_V=1024` vocab columns (i.e. 8 inner 128-wide blocks). To satisfy
        # the BlockSpec constraint, each program writes an `(8, BLOCK_T_STATS)`
        # output tile (8 groups x 128 tokens) into arrays shaped
        # `(groups_padded, tokens_padded)`.
        BLOCK_V_INNER = 128
        GROUP_V = BLOCK_V  # 1024 on v6/v6e
        BLOCKS_PER_GROUP = GROUP_V // BLOCK_V_INNER
        GROUPS_PER_PROGRAM = 8
        PROGRAM_V = GROUPS_PER_PROGRAM * GROUP_V
        BLOCK_T_STATS = 128

        vocab_groups = _ceil_div(vocab_size, GROUP_V)
        groups_padded = _ceil_div(vocab_groups, GROUPS_PER_PROGRAM) * GROUPS_PER_PROGRAM
        group_chunks = groups_padded // GROUPS_PER_PROGRAM

        vocab_blocks_inner = _ceil_div(vocab_size, BLOCK_V_INNER)
        token_blocks_stats = _ceil_div(num_tokens, BLOCK_T_STATS)
        tokens_padded = token_blocks_stats * BLOCK_T_STATS

        def kernel(logits_ref, m_group_ref, l_group_ref, s_group_ref):
            pid_t = pl.program_id(0)
            pid_g = pl.program_id(1)

            temp = jnp.asarray(temperature, dtype=jnp.float32)

            t_start = pid_t * BLOCK_T_STATS
            rows = t_start + jnp.arange(BLOCK_T_STATS, dtype=jnp.int32)[:, None]
            t_in_bounds = rows < num_tokens

            v_start = pid_g * PROGRAM_V
            cols = v_start + jnp.arange(PROGRAM_V, dtype=jnp.int32)[None, :]
            v_in_bounds = cols < vocab_size
            active_mask = t_in_bounds & v_in_bounds

            logits_block = jnp.asarray(logits_ref[...])
            logits_f32 = logits_block.astype(jnp.float32) / temp

            logits_for_reduce = jax.lax.select(
                active_mask,
                logits_f32,
                jnp.full_like(logits_f32, -jnp.inf),
            )
            logits_for_mul = jax.lax.select(
                active_mask,
                logits_f32,
                jnp.zeros_like(logits_f32),
            )

            logits_reduce_blocks = logits_for_reduce.reshape(
                (BLOCK_T_STATS, GROUPS_PER_PROGRAM, BLOCKS_PER_GROUP, BLOCK_V_INNER)
            )
            logits_mul_blocks = logits_for_mul.reshape(
                (BLOCK_T_STATS, GROUPS_PER_PROGRAM, BLOCKS_PER_GROUP, BLOCK_V_INNER)
            )

            # Identify which of the inner 128-wide blocks are real (vs padding in
            # the last group chunk). For invalid blocks we must avoid
            # (-inf) - (-inf) -> NaN.
            blocks_per_program = GROUPS_PER_PROGRAM * BLOCKS_PER_GROUP
            block_ids = pid_g * blocks_per_program + jnp.arange(blocks_per_program, dtype=jnp.int32)
            block_valid = (block_ids < vocab_blocks_inner).reshape((GROUPS_PER_PROGRAM, BLOCKS_PER_GROUP))
            block_valid_bt = jnp.broadcast_to(
                block_valid[None, :, :], (BLOCK_T_STATS, GROUPS_PER_PROGRAM, BLOCKS_PER_GROUP)
            )
            t_in_bounds_bt = jnp.broadcast_to(
                t_in_bounds.reshape((BLOCK_T_STATS, 1, 1)),
                (BLOCK_T_STATS, GROUPS_PER_PROGRAM, BLOCKS_PER_GROUP),
            )

            block_m = jnp.max(logits_reduce_blocks, axis=3, keepdims=True)  # (T, 8, 8, 1)
            block_m_safe = jax.lax.select(block_valid_bt[:, :, :, None], block_m, jnp.zeros_like(block_m))
            block_m_safe = jax.lax.select(t_in_bounds_bt[:, :, :, None], block_m_safe, jnp.zeros_like(block_m_safe))

            logits_minus = logits_reduce_blocks - block_m_safe
            exp_minus = jnp.exp(logits_minus)

            block_l = jnp.sum(exp_minus, axis=3, keepdims=True)  # (T, 8, 8, 1)
            block_s = jnp.sum(exp_minus * logits_mul_blocks, axis=3, keepdims=True)  # (T, 8, 8, 1)

            block_l = jax.lax.select(block_valid_bt[:, :, :, None], block_l, jnp.zeros_like(block_l))
            block_s = jax.lax.select(block_valid_bt[:, :, :, None], block_s, jnp.zeros_like(block_s))
            block_l = jax.lax.select(t_in_bounds_bt[:, :, :, None], block_l, jnp.zeros_like(block_l))
            block_s = jax.lax.select(t_in_bounds_bt[:, :, :, None], block_s, jnp.zeros_like(block_s))

            block_m_out = jax.lax.select(
                block_valid_bt[:, :, :, None],
                block_m,
                jnp.full_like(block_m, -jnp.inf),
            )
            block_m_out = jax.lax.select(t_in_bounds_bt[:, :, :, None], block_m_out, jnp.zeros_like(block_m_out))

            # Combine the 8 inner blocks -> group-level (m, l, s) for each of the
            # 8 groups in this program.
            group_m = jnp.max(block_m_out, axis=2)  # (T, 8, 1)
            group_m = jnp.squeeze(group_m, axis=2)  # (T, 8)

            group_ids = pid_g * GROUPS_PER_PROGRAM + jnp.arange(GROUPS_PER_PROGRAM, dtype=jnp.int32)
            group_valid = group_ids < vocab_groups  # (8,)
            group_valid_bt = jnp.broadcast_to(group_valid[None, :], (BLOCK_T_STATS, GROUPS_PER_PROGRAM))
            t_in_bounds_groups = jnp.broadcast_to(t_in_bounds, (BLOCK_T_STATS, GROUPS_PER_PROGRAM))

            group_m_safe = jax.lax.select(
                group_valid_bt & t_in_bounds_groups,
                group_m,
                jnp.zeros_like(group_m),
            )

            block_m_squeezed = jnp.squeeze(block_m_out, axis=3)  # (T, 8, 8)
            weights = jnp.exp(block_m_squeezed - group_m_safe[:, :, None])[:, :, :, None]  # (T, 8, 8, 1)
            group_l = jnp.sum(block_l * weights, axis=2)  # (T, 8, 1)
            group_s = jnp.sum(block_s * weights, axis=2)  # (T, 8, 1)

            group_l = jnp.squeeze(group_l, axis=2)
            group_s = jnp.squeeze(group_s, axis=2)

            group_l = jax.lax.select(group_valid_bt & t_in_bounds_groups, group_l, jnp.zeros_like(group_l))
            group_s = jax.lax.select(group_valid_bt & t_in_bounds_groups, group_s, jnp.zeros_like(group_s))

            group_m_out = jax.lax.select(
                group_valid_bt,
                group_m,
                jnp.full_like(group_m, -jnp.inf),
            )
            group_m_out = jax.lax.select(t_in_bounds_groups, group_m_out, jnp.zeros_like(group_m_out))

            m_group_ref[...] = jnp.transpose(group_m_out, (1, 0))
            l_group_ref[...] = jnp.transpose(group_l, (1, 0))
            s_group_ref[...] = jnp.transpose(group_s, (1, 0))

        out_shape = (
            jax.ShapeDtypeStruct((groups_padded, tokens_padded), dtype=jnp.float32),
            jax.ShapeDtypeStruct((groups_padded, tokens_padded), dtype=jnp.float32),
            jax.ShapeDtypeStruct((groups_padded, tokens_padded), dtype=jnp.float32),
        )

        logits_spec = pl.BlockSpec((BLOCK_T_STATS, PROGRAM_V), lambda pid_t, pid_g: (pid_t, pid_g))
        stats_spec = pl.BlockSpec(
            (GROUPS_PER_PROGRAM, BLOCK_T_STATS), lambda pid_t, pid_g: (pid_g, pid_t)
        )

        call = pl.pallas_call(
            kernel,
            grid=(token_blocks_stats, group_chunks),
            out_shape=out_shape,
            in_specs=[logits_spec],
            out_specs=[stats_spec, stats_spec, stats_spec],
            interpret=interpret,
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
        )

        group_m_2d, group_l_2d, group_s_2d = call(logits_flat)
        group_m_2d = group_m_2d[:, :num_tokens]
        group_l_2d = group_l_2d[:, :num_tokens]
        group_s_2d = group_s_2d[:, :num_tokens]

        # Reduce across vocab groups on the JAX side:
        #   m = max(group_m)
        #   l = sum(group_l * exp(group_m - m))
        #   s = sum(group_s * exp(group_m - m))
        m_flat = jnp.max(group_m_2d, axis=0)
        l_flat = jnp.sum(group_l_2d * jnp.exp(group_m_2d - m_flat[None, :]), axis=0)
        s_flat = jnp.sum(group_s_2d * jnp.exp(group_m_2d - m_flat[None, :]), axis=0)
        return m_flat, l_flat, s_flat

    def kernel(logits_ref, m_ref, l_ref, s_ref, m_scratch_ref, l_scratch_ref, s_scratch_ref):
        pid_t = pl.program_id(0)
        pid_v = pl.program_id(1)

        temp = jnp.asarray(temperature, dtype=jnp.float32)

        t_start = pid_t * BLOCK_T
        rows = t_start + jnp.arange(BLOCK_T, dtype=jnp.int32)[:, None]
        t_in_bounds = rows < num_tokens

        v_start = pid_v * BLOCK_V
        cols = v_start + jnp.arange(BLOCK_V, dtype=jnp.int32)[None, :]
        v_in_bounds = cols < vocab_size
        active_mask = t_in_bounds & v_in_bounds

        logits_block = jnp.asarray(logits_ref[...])
        logits_f32 = logits_block.astype(jnp.float32) / temp

        logits_for_reduce = jax.lax.select(
            active_mask,
            logits_f32,
            jnp.full_like(logits_f32, -jnp.inf),
        )
        logits_for_mul = jax.lax.select(
            active_mask,
            logits_f32,
            jnp.zeros_like(logits_f32),
        )

        @pl.when(pid_v == 0)
        def _init_state():
            m_scratch_ref[...] = jnp.full((BLOCK_T, 1), -jnp.inf, dtype=jnp.float32)
            l_scratch_ref[...] = jnp.zeros((BLOCK_T, 1), dtype=jnp.float32)
            s_scratch_ref[...] = jnp.zeros((BLOCK_T, 1), dtype=jnp.float32)

        m_prev = jnp.asarray(m_scratch_ref[...]).astype(jnp.float32)
        l_prev = jnp.asarray(l_scratch_ref[...]).astype(jnp.float32)
        s_prev = jnp.asarray(s_scratch_ref[...]).astype(jnp.float32)

        # Avoid NaNs on out-of-bounds token padding.
        m_prev = jax.lax.select(t_in_bounds, m_prev, jnp.zeros_like(m_prev))
        l_prev = jax.lax.select(t_in_bounds, l_prev, jnp.zeros_like(l_prev))
        s_prev = jax.lax.select(t_in_bounds, s_prev, jnp.zeros_like(s_prev))

        seg = BLOCK_V // 128
        logits_seg = logits_for_reduce.reshape((BLOCK_T, seg, 128))
        seg_max = jnp.max(logits_seg, axis=2, keepdims=True)
        block_m = jnp.max(seg_max, axis=1, keepdims=True)
        block_m = block_m.reshape((BLOCK_T, 1))
        block_m = jax.lax.select(t_in_bounds, block_m, jnp.zeros_like(block_m))

        m_next = jnp.maximum(m_prev, block_m)
        alpha = jnp.exp(m_prev - m_next)

        logits_minus = (logits_for_reduce - m_next).reshape((BLOCK_T, seg, 128))
        exp_minus = jnp.exp(logits_minus)
        seg_sum = jnp.sum(exp_minus, axis=2, keepdims=True)
        block_l = jnp.sum(seg_sum, axis=1, keepdims=True)
        block_l = block_l.reshape((BLOCK_T, 1))
        block_l = jax.lax.select(t_in_bounds, block_l, jnp.zeros_like(block_l))

        logits_mul_seg = logits_for_mul.reshape((BLOCK_T, seg, 128))
        seg_sumz = jnp.sum(exp_minus * logits_mul_seg, axis=2, keepdims=True)
        block_s = jnp.sum(seg_sumz, axis=1, keepdims=True)
        block_s = block_s.reshape((BLOCK_T, 1))
        block_s = jax.lax.select(t_in_bounds, block_s, jnp.zeros_like(block_s))

        l_next = l_prev * alpha + block_l
        s_next = s_prev * alpha + block_s

        m_scratch_ref[...] = m_next
        l_scratch_ref[...] = l_next
        s_scratch_ref[...] = s_next

        @pl.when(pid_v == vocab_blocks - 1)
        def _write_final():
            m_safe = jax.lax.select(t_in_bounds, m_next, jnp.zeros_like(m_next))
            l_safe = jax.lax.select(t_in_bounds, l_next, jnp.ones_like(l_next))
            s_safe = jax.lax.select(t_in_bounds, s_next, jnp.zeros_like(s_next))
            m_ref[...] = m_safe
            l_ref[...] = l_safe
            s_ref[...] = s_safe

    out_shape = (
        jax.ShapeDtypeStruct((num_tokens, 1), dtype=jnp.float32),
        jax.ShapeDtypeStruct((num_tokens, 1), dtype=jnp.float32),
        jax.ShapeDtypeStruct((num_tokens, 1), dtype=jnp.float32),
    )

    logits_spec = pl.BlockSpec((BLOCK_T, BLOCK_V), lambda pid_t, pid_v: (pid_t, pid_v))
    token_spec = pl.BlockSpec((BLOCK_T, 1), lambda pid_t, pid_v: (pid_t, 0))

    call = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(token_blocks, vocab_blocks),
            in_specs=[logits_spec],
            out_specs=[token_spec, token_spec, token_spec],
            scratch_shapes=[
                pltpu.VMEM((BLOCK_T, 1), jnp.float32),
                pltpu.VMEM((BLOCK_T, 1), jnp.float32),
                pltpu.VMEM((BLOCK_T, 1), jnp.float32),
            ],
        ),
        out_shape=out_shape,
        interpret=interpret,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
    )

    m_2d, l_2d, s_2d = call(logits_flat)
    m_flat = jnp.squeeze(m_2d, axis=1)
    l_flat = jnp.squeeze(l_2d, axis=1)
    s_flat = jnp.squeeze(s_2d, axis=1)
    return m_flat, l_flat, s_flat


def _grpo_fused_forward_pallas_with_intermediates(
    *,
    logits: Any,  # [B, L+1, V]
    old_logp: Any,  # [B, L] f32 (ignored when use_old_logp=False)
    ref_logp: Any,  # [B, L] f32
    completion_ids: Any,  # [B, L] i32
    advantages: Any,  # [B] f32
    completion_mask: Any,  # [B, L] i32 (0/1)
    use_old_logp: bool,
    temperature: float,
    beta: float,
    eps_low: float,
    eps_high: float,
    interpret: bool,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Forward pass returning outputs + saved intermediates for backward."""

    if not FUSED_AVAILABLE:
        raise RuntimeError("Pallas fused path requires JAX + Pallas")

    import jax
    import jax.numpy as jnp

    logits = jnp.asarray(logits)
    completion_ids = jnp.asarray(completion_ids, dtype=jnp.int32)
    advantages = jnp.asarray(advantages, dtype=jnp.float32)
    completion_mask = jnp.asarray(completion_mask, dtype=jnp.int32)
    old_logp = jnp.asarray(old_logp, dtype=jnp.float32)
    ref_logp = jnp.asarray(ref_logp, dtype=jnp.float32)

    bsz, l_add_1, vocab_size = logits.shape
    seq_len = l_add_1 - 1

    logits_tokens = logits[:, :seq_len, :]
    num_tokens = bsz * seq_len
    logits_flat = logits_tokens.reshape((num_tokens, vocab_size))
    ids_flat = completion_ids.reshape((num_tokens, 1))

    m_flat, l_flat, s_flat = _grpo_fused_logsumexp_stats_pallas(
        logits=logits,
        temperature=temperature,
        interpret=interpret,
    )

    temp = jnp.asarray(temperature, dtype=jnp.float32)
    token_logit_flat = jnp.take_along_axis(logits_flat, ids_flat, axis=1)[:, 0].astype(jnp.float32) / temp

    lse = m_flat + jnp.log(l_flat)
    logp_flat = token_logit_flat - lse

    expected_z = s_flat / l_flat
    entropy_flat = lse - expected_z

    # Mask + scalar inputs.
    keep = completion_mask.reshape((num_tokens,)) != 0
    adv_flat = jnp.broadcast_to(advantages[:, None], (bsz, seq_len)).reshape((num_tokens,))

    if use_old_logp:
        old_flat = old_logp.reshape((num_tokens,))
    else:
        # old=stop_gradient(logp) => numerically equal to logp, but treated as constant.
        old_flat = logp_flat

    eps_low_f = jnp.asarray(eps_low, dtype=jnp.float32)
    eps_high_f = jnp.asarray(eps_high, dtype=jnp.float32)
    one_f = jnp.asarray(1.0, dtype=jnp.float32)

    ratio = jnp.exp(logp_flat - old_flat)
    clipped_ratio = jnp.minimum(jnp.maximum(ratio, one_f - eps_low_f), one_f + eps_high_f)

    per_token_loss1 = ratio * adv_flat
    per_token_loss2 = clipped_ratio * adv_flat
    loss_flat = -jnp.minimum(per_token_loss1, per_token_loss2)

    if beta != 0.0:
        beta_f = jnp.asarray(beta, dtype=jnp.float32)
        ref_flat = ref_logp.reshape((num_tokens,))
        delta = ref_flat - logp_flat
        kl_flat = jnp.exp(delta) - delta - one_f
        loss_flat = loss_flat + beta_f * kl_flat

    loss_flat = jax.lax.select(keep, loss_flat, jnp.zeros_like(loss_flat))

    loss = loss_flat.reshape((bsz, seq_len))
    logp = logp_flat.reshape((bsz, seq_len))
    entropy = entropy_flat.reshape((bsz, seq_len))

    return loss, logp, entropy, m_flat, l_flat, token_logit_flat


def _grpo_fused_backward_pallas(
    *,
    logits: Any,  # [B, L+1, V]
    old_logp: Any,  # [B, L]
    ref_logp: Any,  # [B, L]
    completion_ids: Any,  # [B, L]
    advantages: Any,  # [B]
    completion_mask: Any,  # [B, L] i32 (0/1)
    dloss: Any,  # [B, L]
    m_flat: Any,  # [B*L]
    l_flat: Any,  # [B*L]
    token_logit_flat: Any,  # [B*L] (scaled by temperature)
    temperature: float,
    beta: float,
    eps_low: float,
    eps_high: float,
    interpret: bool,
) -> Any:
    """Backward for dlogits (adapted from tests/grpo_fused_kernel)."""

    if not FUSED_AVAILABLE:
        raise RuntimeError("Pallas fused path requires JAX + Pallas")

    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    logits = jnp.asarray(logits)
    completion_ids = jnp.asarray(completion_ids, dtype=jnp.int32)
    old_logp = jnp.asarray(old_logp, dtype=jnp.float32)
    ref_logp = jnp.asarray(ref_logp, dtype=jnp.float32)
    advantages = jnp.asarray(advantages, dtype=jnp.float32)
    completion_mask = jnp.asarray(completion_mask, dtype=jnp.int32)
    dloss = jnp.asarray(dloss, dtype=jnp.float32)
    m_flat = jnp.asarray(m_flat, dtype=jnp.float32)
    l_flat = jnp.asarray(l_flat, dtype=jnp.float32)
    token_logit_flat = jnp.asarray(token_logit_flat, dtype=jnp.float32)

    bsz, l_add_1, vocab_size = logits.shape
    seq_len = l_add_1 - 1

    logits_tokens = logits[:, :seq_len, :]
    num_tokens = bsz * seq_len

    logits_flat = logits_tokens.reshape((num_tokens, vocab_size))
    ids_flat = completion_ids.reshape((num_tokens, 1))
    old_logp_flat = old_logp.reshape((num_tokens, 1))
    ref_logp_flat = ref_logp.reshape((num_tokens, 1))
    mask_flat = completion_mask.reshape((num_tokens, 1))
    dloss_flat = dloss.reshape((num_tokens, 1))
    adv_flat = jnp.broadcast_to(advantages[:, None], (bsz, seq_len)).reshape((num_tokens, 1))

    # TPU Mosaic: keep per-token intermediates as (num_tokens, 1).
    m_flat = m_flat.reshape((num_tokens, 1))
    l_flat = l_flat.reshape((num_tokens, 1))
    token_logit_flat = token_logit_flat.reshape((num_tokens, 1))

    token_blocks = _ceil_div(num_tokens, BLOCK_T)
    vocab_blocks = _ceil_div(vocab_size, BLOCK_V)

    def kernel(
        logits_ref,
        ids_ref,
        old_logp_ref,
        ref_logp_ref,
        adv_ref,
        mask_ref,
        dloss_ref,
        m_ref,
        l_ref,
        token_logit_ref,
        dlogits_ref,
    ):
        pid_t = pl.program_id(0)
        pid_v = pl.program_id(1)

        temp = jnp.asarray(temperature, dtype=jnp.float32)
        eps_low_f = jnp.asarray(eps_low, dtype=jnp.float32)
        eps_high_f = jnp.asarray(eps_high, dtype=jnp.float32)
        beta_f = jnp.asarray(beta, dtype=jnp.float32)
        one_f = jnp.asarray(1.0, dtype=jnp.float32)

        t_start = pid_t * BLOCK_T
        rows = t_start + jnp.arange(BLOCK_T, dtype=jnp.int32)[:, None]
        t_in_bounds = rows < num_tokens
        t_in_bounds_i32 = jax.lax.select(
            t_in_bounds,
            jnp.ones_like(rows, dtype=jnp.int32),
            jnp.zeros_like(rows, dtype=jnp.int32),
        )

        v_start = pid_v * BLOCK_V
        cols = v_start + jnp.arange(BLOCK_V, dtype=jnp.int32)[None, :]
        v_in_bounds = cols < vocab_size
        v_in_bounds_i32 = jax.lax.select(
            v_in_bounds,
            jnp.ones_like(cols, dtype=jnp.int32),
            jnp.zeros_like(cols, dtype=jnp.int32),
        )
        v_mask_i32 = jnp.broadcast_to(v_in_bounds_i32, (BLOCK_T, BLOCK_V))

        token_ids = jnp.asarray(ids_ref[...]).astype(jnp.int32)
        old = jnp.asarray(old_logp_ref[...]).astype(jnp.float32)
        ref = jnp.asarray(ref_logp_ref[...]).astype(jnp.float32)
        adv = jnp.asarray(adv_ref[...]).astype(jnp.float32)
        keep_i32 = jnp.asarray(mask_ref[...]).astype(jnp.int32)
        dloss_local = jnp.asarray(dloss_ref[...]).astype(jnp.float32)

        m = jnp.asarray(m_ref[...]).astype(jnp.float32)
        l = jnp.asarray(l_ref[...]).astype(jnp.float32)
        token_logit = jnp.asarray(token_logit_ref[...]).astype(jnp.float32)

        m = jax.lax.select(t_in_bounds, m, jnp.zeros_like(m))
        l = jax.lax.select(t_in_bounds, l, jnp.ones_like(l))
        token_logit = jax.lax.select(t_in_bounds, token_logit, jnp.zeros_like(token_logit))

        row_active_i32 = keep_i32 * t_in_bounds_i32
        row_active = row_active_i32 != 0
        row_mask_i32 = jnp.broadcast_to(row_active_i32, (BLOCK_T, BLOCK_V))
        active_mask = (row_mask_i32 != 0) & (v_mask_i32 != 0)

        old = jax.lax.select(row_active, old, jnp.zeros_like(old))
        ref = jax.lax.select(row_active, ref, jnp.zeros_like(ref))
        adv = jax.lax.select(row_active, adv, jnp.zeros_like(adv))
        dloss_local = jax.lax.select(row_active, dloss_local, jnp.zeros_like(dloss_local))
        token_logit = jax.lax.select(row_active, token_logit, jnp.zeros_like(token_logit))

        lse = m + jnp.log(l)
        logp = jax.lax.select(row_active, token_logit - lse, jnp.zeros_like(lse))

        ratio = jnp.exp(logp - old)
        is_low_clipped = (ratio < one_f - eps_low_f) & (adv < 0.0)
        is_high_clipped = (ratio > one_f + eps_high_f) & (adv > 0.0)
        not_clipped = ~(is_low_clipped | is_high_clipped)

        not_clipped_f32 = jax.lax.select(
            not_clipped,
            jnp.ones_like(old, dtype=jnp.float32),
            jnp.zeros_like(old, dtype=jnp.float32),
        )
        dlogp = (-adv * ratio) * not_clipped_f32
        dlogp = dlogp + beta_f * (one_f - jnp.exp(ref - logp))

        scale = dloss_local * dlogp / temp

        logits_block = jnp.asarray(logits_ref[...])
        logits_tile = logits_block.astype(jnp.float32) / temp
        logits_tile = jax.lax.select(
            active_mask,
            logits_tile,
            jnp.full_like(logits_tile, -jnp.inf),
        )

        lse_safe = jax.lax.select(row_active, lse, jnp.zeros_like(lse))
        probs = jnp.exp(logits_tile - lse_safe)
        one_hot = (token_ids == cols) & active_mask
        one_hot_f32 = jax.lax.select(
            one_hot,
            jnp.ones_like(probs, dtype=jnp.float32),
            jnp.zeros_like(probs, dtype=jnp.float32),
        )
        grad_f32 = (one_hot_f32 - probs) * scale
        dlogits_ref[...] = grad_f32.astype(logits_block.dtype)

    logits_spec = pl.BlockSpec((BLOCK_T, BLOCK_V), lambda pid_t, pid_v: (pid_t, pid_v))
    token_spec = pl.BlockSpec((BLOCK_T, 1), lambda pid_t, pid_v: (pid_t, 0))

    out_shape = jax.ShapeDtypeStruct((num_tokens, vocab_size), dtype=logits.dtype)
    call = pl.pallas_call(
        kernel,
        grid=(token_blocks, vocab_blocks),
        out_shape=out_shape,
        in_specs=[
            logits_spec,
            token_spec,
            token_spec,
            token_spec,
            token_spec,
            token_spec,
            token_spec,
            token_spec,
            token_spec,
            token_spec,
        ],
        out_specs=logits_spec,
        interpret=interpret,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )

    dlogits_flat = call(
        logits_flat,
        ids_flat,
        old_logp_flat,
        ref_logp_flat,
        adv_flat,
        mask_flat,
        dloss_flat,
        m_flat,
        l_flat,
        token_logit_flat,
    )

    dlogits_tokens = dlogits_flat.reshape((bsz, seq_len, vocab_size))
    zeros_last = jnp.zeros((bsz, 1, vocab_size), dtype=logits.dtype)
    return jnp.concatenate([dlogits_tokens, zeros_last], axis=1)


if FUSED_AVAILABLE:
    import jax

    def _grpo_loss_logp_entropy_fused_pallas_jax_impl(
        logits: Any,
        old_logp: Any,
        ref_logp: Any,
        completion_ids: Any,
        advantages: Any,
        completion_mask: Any,
        use_old_logp: bool,
        temperature: float,
        beta: float,
        eps_low: float,
        eps_high: float,
    ) -> tuple[Any, Any, Any]:
        loss, logp, entropy, _m, _l, _token_logit = _grpo_fused_forward_pallas_with_intermediates(
            logits=logits,
            old_logp=old_logp,
            ref_logp=ref_logp,
            completion_ids=completion_ids,
            advantages=advantages,
            completion_mask=completion_mask,
            use_old_logp=use_old_logp,
            temperature=temperature,
            beta=beta,
            eps_low=eps_low,
            eps_high=eps_high,
            interpret=(not _is_tpu_runtime()),
        )
        return loss, logp, entropy

    _grpo_loss_logp_entropy_fused_pallas_jax = jax.custom_vjp(
        _grpo_loss_logp_entropy_fused_pallas_jax_impl,
        nondiff_argnums=(6, 7, 8, 9, 10),
    )

    def _grpo_loss_logp_entropy_fused_pallas_jax_fwd(
        logits: Any,
        old_logp: Any,
        ref_logp: Any,
        completion_ids: Any,
        advantages: Any,
        completion_mask: Any,
        use_old_logp: bool,
        temperature: float,
        beta: float,
        eps_low: float,
        eps_high: float,
    ):
        loss, logp, entropy, m_flat, l_flat, token_logit_flat = _grpo_fused_forward_pallas_with_intermediates(
            logits=logits,
            old_logp=old_logp,
            ref_logp=ref_logp,
            completion_ids=completion_ids,
            advantages=advantages,
            completion_mask=completion_mask,
            use_old_logp=use_old_logp,
            temperature=temperature,
            beta=beta,
            eps_low=eps_low,
            eps_high=eps_high,
            interpret=(not _is_tpu_runtime()),
        )
        residuals = (
            logits,
            old_logp,
            ref_logp,
            completion_ids,
            advantages,
            completion_mask,
            m_flat,
            l_flat,
            token_logit_flat,
        )
        return (loss, logp, entropy), residuals

    def _grpo_loss_logp_entropy_fused_pallas_jax_bwd(
        use_old_logp: bool,
        temperature: float,
        beta: float,
        eps_low: float,
        eps_high: float,
        residuals,
        g,
    ):
        import importlib

        jnp = importlib.import_module("jax.numpy")

        (
            logits,
            old_logp,
            ref_logp,
            completion_ids,
            advantages,
            completion_mask,
            m_flat,
            l_flat,
            token_logit_flat,
        ) = residuals

        dloss, _dlogp, _dentropy = g
        if dloss is None or type(dloss).__name__ == "Zero":
            dlogits = jnp.zeros_like(jnp.asarray(logits))
        else:
            if use_old_logp:
                old_for_kernel = old_logp
            else:
                # old=stop_gradient(logp); numeric == logp.
                logits_shape = jnp.asarray(old_logp).shape
                lse = jnp.asarray(m_flat) + jnp.log(jnp.asarray(l_flat))
                logp_flat = jnp.asarray(token_logit_flat) - lse
                old_for_kernel = logp_flat.reshape(logits_shape)

            dlogits = _grpo_fused_backward_pallas(
                logits=logits,
                old_logp=old_for_kernel,
                ref_logp=ref_logp,
                completion_ids=completion_ids,
                advantages=advantages,
                completion_mask=completion_mask,
                dloss=dloss,
                m_flat=m_flat,
                l_flat=l_flat,
                token_logit_flat=token_logit_flat,
                temperature=temperature,
                beta=beta,
                eps_low=eps_low,
                eps_high=eps_high,
                interpret=(not _is_tpu_runtime()),
            )

        return (dlogits, None, None, None, None, None)

    _grpo_loss_logp_entropy_fused_pallas_jax.defvjp(  # pyright: ignore[reportFunctionMemberAccess]
        _grpo_loss_logp_entropy_fused_pallas_jax_fwd,
        _grpo_loss_logp_entropy_fused_pallas_jax_bwd,
    )


def grpo_loss_logp_entropy(
    logits: Any,
    *,
    old_logp: Any | None,
    ref_logp: Any | None,
    completion_ids: Any,
    advantages: Any,
    completion_mask: Any | None,
    temperature: float = 1.0,
    beta: float = 0.0,
    eps_low: float = 0.2,
    eps_high: float = 0.2,
    use_fused: bool | None = None,
) -> tuple[Any, Any, Any]:
    """Compute per-token PPO/GRPO loss plus token logp and token entropy.

    This is intended for `TrainGRPOModule`:
      - avoids materializing full-vocab softmax/log_softmax on the fused path
      - returns per-token logp/entropy for monitoring (recommended to stop_grad)

    Shapes:
      - logits: [B, L+1, V]
      - completion_ids: [B, L]
      - advantages: [B]
      - old_logp/ref_logp: [B, L] (optional)
      - completion_mask: [B, L] (0/1, optional)
    """

    import jax
    import jax.numpy as jnp

    logits_jax = jnp.asarray(logits)
    completion_ids_jax = jnp.asarray(completion_ids, dtype=jnp.int32)
    advantages_jax = jnp.asarray(advantages, dtype=jnp.float32)

    completion_mask_jax = (
        jnp.ones(completion_ids_jax.shape, dtype=jnp.int32)
        if completion_mask is None
        else jnp.asarray(completion_mask, dtype=jnp.int32)
    )

    use_old_logp = old_logp is not None
    if old_logp is None:
        # Placeholder (ignored by the fused path when use_old_logp=False).
        old_logp_jax = jnp.zeros(completion_ids_jax.shape, dtype=jnp.float32)
    else:
        old_logp_jax = jnp.asarray(old_logp, dtype=jnp.float32)

    if beta != 0.0 and ref_logp is None:
        raise ValueError("ref_logp must be provided when beta != 0")
    ref_logp_jax = (
        jnp.zeros_like(old_logp_jax)
        if (beta == 0.0 and ref_logp is None)
        else jnp.asarray(ref_logp, dtype=jnp.float32)
    )

    if use_fused is None:
        use_fused = grpo_fused_enabled()

    if not use_fused or not FUSED_AVAILABLE:
        logp, entropy = _grpo_logp_entropy_jax(
            logits=logits_jax,
            completion_ids=completion_ids_jax,
            temperature=temperature,
        )
        if old_logp is None:
            old_logp_jax = jax.lax.stop_gradient(logp.astype(jnp.float32))
        loss = _ppo_loss_from_logp(
            logp=logp,
            old_logp=old_logp_jax,
            ref_logp=ref_logp_jax,
            advantages=advantages_jax,
            completion_mask=completion_mask_jax,
            beta=beta,
            eps_low=eps_low,
            eps_high=eps_high,
        )
        return loss, logp, entropy

    loss, logp, entropy = _grpo_loss_logp_entropy_fused_pallas_jax(
        logits_jax,
        old_logp_jax,
        ref_logp_jax,
        completion_ids_jax,
        advantages_jax,
        completion_mask_jax,
        use_old_logp,
        temperature,
        beta,
        eps_low,
        eps_high,
    )
    return loss, logp, entropy


__all__ = [
    "BLOCK_T",
    "BLOCK_V",
    "FUSED_AVAILABLE",
    "grpo_fused_enabled",
    "grpo_loss_logp_entropy",
]
