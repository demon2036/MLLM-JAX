# pyright: reportAttributeAccessIssue=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportMissingParameterType=false, reportCallIssue=false, reportFunctionMemberAccess=false

from __future__ import annotations


from tests.grpo_fused_kernel import grpo_reference


def _compute_fused_available() -> bool:
    try:
        from jax.experimental import pallas as _pl
        from jax.experimental.pallas import tpu as _pltpu
    except ImportError:
        return False
    _ = (_pl, _pltpu)
    return True


# True when JAX + Pallas are importable. Non-TPU runs use interpret=True.
FUSED_AVAILABLE = _compute_fused_available()


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# Kernel tile sizes.
#
# Default to a small token tile to keep interpret-mode CPU tests fast, but on
# TPU v6/v6e match XLA's 1D layout tiling (T(1024)) for token-major operands
# (e.g. ids[num_tokens]) to avoid Mosaic layout verification errors.
#
# NOTE: On TPU v3, large BLOCK_V values can trigger Mosaic lowering patterns
# involving sublane gathers (e.g. during row-wise reductions like max/sum) that
# are not supported. Keep BLOCK_V small and a multiple of 128.


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


BLOCK_T = _select_block_t()


def _select_block_v() -> int:
    """Select vocab tile size (multiple of 128).

    Keep interpret-mode CPU tests fast with a small tile, but use larger tiles
    on newer TPUs to reduce the number of vocab reduction steps.
    """

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
        return 512
    if "v4" in device_kind:
        # TPU v4 Mosaic: wider reduce (e.g. 256) can lower to unsupported
        # sublane-gather patterns for row-wise reductions (max/sum). Keep the
        # reduction width at 128 for maximum compatibility.
        return 128
    return 128


BLOCK_V = _select_block_v()


def _is_tpu_runtime() -> bool:
    if not FUSED_AVAILABLE:
        return False
    import jax

    return any(d.platform == "tpu" for d in jax.devices())


def _grpo_fused_forward_pallas(
    *,
    logits: object,  # [B, L+1, V] bf16
    old_logp: object,  # [B, L] f32
    ref_logp: object,  # [B, L] f32 (ignored when beta==0)
    completion_ids: object,  # [B, L] i32
    advantages: object,  # [B] f32
    completion_mask: object,  # [B, L] i32 (0/1)
    temperature: float,
    beta: float,
    eps_low: float,
    eps_high: float,
    interpret: bool,
) -> tuple[object, object, object]:
    """Pallas fused forward.

    Returns (loss, kl, is_clipped_i32) with shapes [B, L].
    """

    loss, kl, is_clipped, _m_flat, _l_flat, _token_logit_flat = (
        _grpo_fused_forward_pallas_with_intermediates(
            logits=logits,
            old_logp=old_logp,
            ref_logp=ref_logp,
            completion_ids=completion_ids,
            advantages=advantages,
            completion_mask=completion_mask,
            temperature=temperature,
            beta=beta,
            eps_low=eps_low,
            eps_high=eps_high,
            interpret=interpret,
        )
    )
    return loss, kl, is_clipped


def _grpo_fused_forward_pallas_with_intermediates(
    *,
    logits: object,  # [B, L+1, V] bf16
    old_logp: object,  # [B, L] f32
    ref_logp: object,  # [B, L] f32 (ignored when beta==0)
    completion_ids: object,  # [B, L] i32
    advantages: object,  # [B] f32
    completion_mask: object,  # [B, L] i32 (0/1)
    temperature: float,
    beta: float,
    eps_low: float,
    eps_high: float,
    interpret: bool,
) -> tuple[object, object, object, object, object, object]:
    """Pallas fused forward with saved logsumexp intermediates.

    Returns:
      (loss, kl, is_clipped_i32, m_flat, l_flat, token_logit_flat)
    where m/l/token_logit are the reduction accumulators from the forward pass,
    flattened token-major as [B*L].
    """

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

    bsz, l_add_1, vocab_size = logits.shape
    seq_len = l_add_1 - 1

    logits = logits[:, :seq_len, :]
    num_tokens = bsz * seq_len

    # Flatten token-major to make the token block axis contiguous.
    logits_flat = logits.reshape((num_tokens, vocab_size))
    ids_flat = completion_ids.reshape((num_tokens, 1))
    old_logp_flat = old_logp.reshape((num_tokens, 1))
    ref_logp_flat = ref_logp.reshape((num_tokens, 1))
    mask_flat = completion_mask.reshape((num_tokens, 1))
    adv_flat = jnp.broadcast_to(advantages[:, None], (bsz, seq_len)).reshape((num_tokens, 1))

    token_blocks = _ceil_div(num_tokens, BLOCK_T)
    vocab_blocks = _ceil_div(vocab_size, BLOCK_V)

    # Kernel captures shapes and hyperparameters as static values.
    def kernel(
        logits_ref,
        m_block_ref,
        l_block_ref,
    ):
        pid_t = pl.program_id(0)
        pid_v = pl.program_id(1)

        # Hyperparams may be traced scalars under jax.jit; keep them as JAX
        # scalars inside the kernel (no Python float() / Python branching).
        temp = jnp.asarray(temperature, dtype=jnp.float32)

        t_start = pid_t * BLOCK_T
        rows = t_start + jnp.arange(BLOCK_T, dtype=jnp.int32)[:, None]
        t_in_bounds = rows < num_tokens

        v_start = pid_v * BLOCK_V

        # Load current vocab tile and mask out padded tail.
        cols = v_start + jnp.arange(BLOCK_V, dtype=jnp.int32)[None, :]
        v_in_bounds = cols < vocab_size
        active_mask = t_in_bounds & v_in_bounds

        logits_block = jnp.asarray(logits_ref[...])
        logits_tile = logits_block.astype(jnp.float32) / temp
        logits_tile = jax.lax.select(
            active_mask,
            logits_tile,
            jnp.full_like(logits_tile, -jnp.inf),
        )

        # NOTE: TPU v4 Mosaic: avoid rank-1 vectors. keepdims keeps the
        # reduction output at shape (BLOCK_T, 1) and compiles reliably.
        block_m = jnp.max(logits_tile, axis=1, keepdims=True)
        # Avoid (-inf) - (-inf) -> NaN on padded token rows.
        block_m = jax.lax.select(t_in_bounds, block_m, jnp.zeros_like(block_m))
        l_block = jnp.sum(
            jnp.exp(logits_tile - block_m),
            axis=1,
            keepdims=True,
        )
        l_block = jax.lax.select(t_in_bounds, l_block, jnp.zeros_like(l_block))

        m_block_ref[...] = block_m[None, :, :]
        l_block_ref[...] = l_block[None, :, :]

    out_shape = (
        jax.ShapeDtypeStruct((vocab_blocks, num_tokens, 1), dtype=jnp.float32),  # m_blocks
        jax.ShapeDtypeStruct((vocab_blocks, num_tokens, 1), dtype=jnp.float32),  # l_blocks
    )

    logits_spec = pl.BlockSpec((BLOCK_T, BLOCK_V), lambda pid_t, pid_v: (pid_t, pid_v))
    ml_spec = pl.BlockSpec((1, BLOCK_T, 1), lambda pid_t, pid_v: (pid_v, pid_t, 0))

    call = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(token_blocks, vocab_blocks),
            in_specs=[logits_spec],
            out_specs=[ml_spec, ml_spec],
        ),
        out_shape=out_shape,
        interpret=interpret,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
        ),
    )

    m_blocks, l_blocks = call(logits_flat)
    m_blocks_sq = jnp.squeeze(m_blocks, axis=2)
    l_blocks_sq = jnp.squeeze(l_blocks, axis=2)

    # Reduce block-wise logsumexp state on the JAX side:
    #   m = max_i m_i
    #   l = sum_i l_i * exp(m_i - m)
    m_flat = jnp.max(m_blocks_sq, axis=0)
    l_flat = jnp.sum(l_blocks_sq * jnp.exp(m_blocks_sq - m_flat[None, :]), axis=0)

    # Gather the selected token logit outside the kernel (avoid per-block
    # one_hot / max selection logic inside Pallas).
    temp = jnp.asarray(temperature, dtype=jnp.float32)
    token_logit_flat = (
        jnp.take_along_axis(logits_flat, ids_flat, axis=1)[:, 0].astype(jnp.float32) / temp
    )

    lse = m_flat + jnp.log(l_flat)
    logp = token_logit_flat - lse

    old = jnp.squeeze(old_logp_flat, axis=1)
    ref = jnp.squeeze(ref_logp_flat, axis=1)
    adv = jnp.squeeze(adv_flat, axis=1)
    keep_i32 = jnp.squeeze(mask_flat, axis=1)

    eps_low_f = jnp.asarray(eps_low, dtype=jnp.float32)
    eps_high_f = jnp.asarray(eps_high, dtype=jnp.float32)
    beta_f = jnp.asarray(beta, dtype=jnp.float32)
    one_f = jnp.asarray(1.0, dtype=jnp.float32)

    ratio = jnp.exp(logp - old)
    clipped_ratio = jnp.minimum(
        jnp.maximum(ratio, one_f - eps_low_f),
        one_f + eps_high_f,
    )

    per_token_loss1 = ratio * adv
    per_token_loss2 = clipped_ratio * adv
    loss_flat = -jnp.minimum(per_token_loss1, per_token_loss2)

    is_low_clipped = (ratio < one_f - eps_low_f) & (adv < 0.0)
    is_high_clipped = (ratio > one_f + eps_high_f) & (adv > 0.0)
    is_clipped_bool = is_low_clipped | is_high_clipped

    delta = ref - logp
    kl_flat = jnp.exp(delta) - delta - one_f
    loss_flat = loss_flat + beta_f * kl_flat

    keep = keep_i32 != 0
    loss_flat = jax.lax.select(keep, loss_flat, jnp.zeros_like(loss_flat))
    kl_flat = jax.lax.select(keep, kl_flat, jnp.zeros_like(kl_flat))

    # Avoid i32->i1 truncation (Mosaic) by never doing astype(bool_).
    is_clipped_i32_flat = jax.lax.select(
        is_clipped_bool,
        jnp.ones_like(keep_i32, dtype=jnp.int32),
        jnp.zeros_like(keep_i32, dtype=jnp.int32),
    )
    is_clipped_i32_flat = jax.lax.select(
        keep,
        is_clipped_i32_flat,
        jnp.zeros_like(is_clipped_i32_flat),
    )

    loss = loss_flat.reshape((bsz, seq_len))
    kl = kl_flat.reshape((bsz, seq_len))
    is_clipped = is_clipped_i32_flat.reshape((bsz, seq_len))
    return loss, kl, is_clipped, m_flat, l_flat, token_logit_flat


def _grpo_fused_backward_pallas(
    *,
    logits: object,  # [B, L+1, V] bf16
    old_logp: object,  # [B, L] f32
    ref_logp: object,  # [B, L] f32
    completion_ids: object,  # [B, L] i32
    advantages: object,  # [B] f32
    completion_mask: object,  # [B, L] i32 (0/1)
    dloss: object,  # [B, L] f32
    m_flat: object,  # [B*L] f32
    l_flat: object,  # [B*L] f32
    token_logit_flat: object,  # [B*L] f32 (scaled by temperature)
    temperature: float,
    beta: float,
    eps_low: float,
    eps_high: float,
    interpret: bool,
) -> object:
    """Pallas fused backward for dlogits.

    Only returns gradient w.r.t logits. Masked tokens and logits[:, -1, :] are
    zero.
    """

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

    # TPU v4 Mosaic: keep per-token intermediates as (num_tokens, 1) so the
    # kernel avoids rank-1 vectors (which can trigger sublane-gather lowering).
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

        # Hyperparams may be traced scalars under jax.jit; keep them as JAX
        # scalars inside the kernel (no Python float() / Python branching).
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

        # Per-token inputs.
        token_ids = jnp.asarray(ids_ref[...]).astype(jnp.int32)
        old = jnp.asarray(old_logp_ref[...]).astype(jnp.float32)
        ref = jnp.asarray(ref_logp_ref[...]).astype(jnp.float32)
        adv = jnp.asarray(adv_ref[...]).astype(jnp.float32)
        keep_i32 = jnp.asarray(mask_ref[...]).astype(jnp.int32)
        dloss_local = jnp.asarray(dloss_ref[...]).astype(jnp.float32)

        # Saved forward intermediates for stable logsumexp.
        m = jnp.asarray(m_ref[...]).astype(jnp.float32)
        l = jnp.asarray(l_ref[...]).astype(jnp.float32)
        token_logit = jnp.asarray(token_logit_ref[...]).astype(jnp.float32)

        # Avoid NaNs on token padding by forcing a valid logsumexp state.
        m = jax.lax.select(t_in_bounds, m, jnp.zeros_like(m))
        l = jax.lax.select(t_in_bounds, l, jnp.ones_like(l))
        token_logit = jax.lax.select(t_in_bounds, token_logit, jnp.zeros_like(token_logit))

        # Row-level active mask: completion mask + token in-bounds.
        # NOTE: Avoid bool vector reshapes/broadcasts (e.g. row_active[:, None])
        # which can trigger unsupported Mosaic shape-casts on TPU v6e.
        row_active_i32 = keep_i32 * t_in_bounds_i32
        row_active = row_active_i32 != 0
        row_mask_i32 = jnp.broadcast_to(row_active_i32, (BLOCK_T, BLOCK_V))
        active_mask = (row_mask_i32 != 0) & (v_mask_i32 != 0)

        # Zero out non-active scalars to keep exp() finite.
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

        # Load logits tile and mask padded vocab tail and inactive rows.
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
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
        ),
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
    # The loss only depends on the first L positions; logits[:, -1, :] has zero grad.
    zeros_last = jnp.zeros((bsz, 1, vocab_size), dtype=logits.dtype)
    return jnp.concatenate([dlogits_tokens, zeros_last], axis=1)


if FUSED_AVAILABLE:
    import jax

    # Treat scalar hyperparameters as static/non-differentiable so Pallas kernels
    # don't close over traced scalar constants under jax.jit.
    def _grpo_loss_fused_pallas_jax_impl(
        logits: object,
        old_logp: object,
        ref_logp: object,
        completion_ids: object,
        advantages: object,
        completion_mask: object,
        temperature: float,
        beta: float,
        eps_low: float,
        eps_high: float,
    ) -> tuple[object, object, object]:
        loss, kl, is_clipped_i32, _m, _l, _token_logit = _grpo_fused_forward_pallas_with_intermediates(
            logits=logits,
            old_logp=old_logp,
            ref_logp=ref_logp,
            completion_ids=completion_ids,
            advantages=advantages,
            completion_mask=completion_mask,
            temperature=temperature,
            beta=beta,
            eps_low=eps_low,
            eps_high=eps_high,
            interpret=(not _is_tpu_runtime()),
        )
        return loss, kl, is_clipped_i32

    _grpo_loss_fused_pallas_jax = jax.custom_vjp(
        _grpo_loss_fused_pallas_jax_impl,
        nondiff_argnums=(6, 7, 8, 9),
    )

    def _grpo_loss_fused_pallas_jax_fwd(
        logits: object,
        old_logp: object,
        ref_logp: object,
        completion_ids: object,
        advantages: object,
        completion_mask: object,
        temperature: float,
        beta: float,
        eps_low: float,
        eps_high: float,
    ):
        loss, kl, is_clipped_i32, m_flat, l_flat, token_logit_flat = (
            _grpo_fused_forward_pallas_with_intermediates(
                logits=logits,
                old_logp=old_logp,
                ref_logp=ref_logp,
                completion_ids=completion_ids,
                advantages=advantages,
                completion_mask=completion_mask,
                temperature=temperature,
                beta=beta,
                eps_low=eps_low,
                eps_high=eps_high,
                interpret=(not _is_tpu_runtime()),
            )
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
        return (loss, kl, is_clipped_i32), residuals

    def _grpo_loss_fused_pallas_jax_bwd(temperature, beta, eps_low, eps_high, residuals, g):
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

        dloss, _dkl, _dis_clipped = g
        # When the loss output is unused, JAX may pass a special "Zero" object
        # as the cotangent. Treat it as an all-zero dloss.
        if dloss is None or type(dloss).__name__ == "Zero":
            dlogits = jnp.zeros_like(jnp.asarray(logits))
        else:
            dlogits = _grpo_fused_backward_pallas(
                logits=logits,
                old_logp=old_logp,
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

        # Gradients correspond to the 6 differentiable args only.
        return (dlogits, None, None, None, None, None)

    _grpo_loss_fused_pallas_jax.defvjp(  # pyright: ignore[reportFunctionMemberAccess]
        _grpo_loss_fused_pallas_jax_fwd,
        _grpo_loss_fused_pallas_jax_bwd,
    )


def grpo_loss_fused_pallas(
    logits: object,
    old_logp: object | None,
    ref_logp: object | None,
    completion_ids: object,
    advantages: object,
    completion_mask: object | None = None,
    temperature: float = 1.0,
    beta: float = 0.0,
    eps_low: float = 0.2,
    eps_high: float = 0.2,
    *,
    backend: str | None = None,
) -> tuple[object, object | None, object]:
    """Fused GRPO loss forward.

    - backend="python": always uses the pure-python reference.
    - backend="jax"/None: uses the Pallas kernel when available; otherwise falls
      back to the JAX reference.
    """

    if backend == "python":
        return grpo_reference.grpo_loss_reference(
            logits,
            old_logp,
            ref_logp,
            completion_ids,
            advantages,
            completion_mask=completion_mask,
            temperature=temperature,
            beta=beta,
            eps_low=eps_low,
            eps_high=eps_high,
            backend="python",
        )

    if not FUSED_AVAILABLE:
        # Best-effort fallback that keeps CPU-only environments functional.
        return grpo_reference.grpo_loss_reference(
            logits,
            old_logp,
            ref_logp,
            completion_ids,
            advantages,
            completion_mask=completion_mask,
            temperature=temperature,
            beta=beta,
            eps_low=eps_low,
            eps_high=eps_high,
            backend=backend,
        )

    import jax.numpy as jnp

    # JAX path: prefer fused Pallas when possible.
    logits_jax = jnp.asarray(logits)
    completion_ids_jax = jnp.asarray(completion_ids, dtype=jnp.int32)
    advantages_jax = jnp.asarray(advantages, dtype=jnp.float32)
    completion_mask_jax = (
        jnp.ones(completion_ids_jax.shape, dtype=jnp.int32)
        if completion_mask is None
        else jnp.asarray(completion_mask, dtype=jnp.int32)
    )

    if beta != 0.0 and ref_logp is None:
        raise ValueError("ref_logp must be provided when beta != 0")
    if old_logp is None:
        # The fused kernel is designed for the common PPO-style path where
        # old_logp is precomputed. Fall back to the JAX reference otherwise.
        return grpo_reference.grpo_loss_reference_jax(
            logits_jax,
            old_logp=None,
            ref_logp=ref_logp,
            completion_ids=completion_ids_jax,
            advantages=advantages_jax,
            completion_mask=completion_mask_jax,
            temperature=temperature,
            beta=beta,
            eps_low=eps_low,
            eps_high=eps_high,
        )

    old_logp_jax = jnp.asarray(old_logp, dtype=jnp.float32)
    ref_logp_jax = (
        jnp.zeros_like(old_logp_jax)
        if (beta == 0.0 and ref_logp is None)
        else jnp.asarray(ref_logp, dtype=jnp.float32)
    )

    loss, kl, is_clipped_i32 = _grpo_loss_fused_pallas_jax(
        logits_jax,
        old_logp_jax,
        ref_logp_jax,
        completion_ids_jax,
        advantages_jax,
        completion_mask_jax,
        temperature,
        beta,
        eps_low,
        eps_high,
    )

    is_clipped = jnp.asarray(is_clipped_i32) != 0
    return loss, (None if beta == 0.0 else kl), is_clipped


__all__ = ["FUSED_AVAILABLE", "grpo_loss_fused_pallas"]
