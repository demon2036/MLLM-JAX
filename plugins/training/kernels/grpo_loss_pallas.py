from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GRPOKernelConfig:
    """Configuration for the logits-level GRPO Pallas kernel.

    This kernel computes (per token):
      - selective log-prob for the chosen token id
      - GRPO clipped objective term (no KL term in this repo by default)

    Notes:
    - This is a *logits-level* kernel: it does not fuse the LM head matmul.
    - The kernel streams over vocab tiles (`block_size`) and avoids materializing
      `log_softmax(logits)` (shape `[B,T,V]`), which reduces peak memory.
    - `time_block` controls how many time steps are processed per program.
    - Mosaic Pallas kernels are not SPMD auto-partitionable; multi-device usage
      must wrap the call via `jax.experimental.shard_map` (implemented in this
      repo as a separate wrapper).
    """

    block_size: int = 2048
    time_block: int = 8
    epsilon_low: float = 0.2
    epsilon_high: float = 0.2
    temperature: float = 1.0
    # Backward implementation selector.
    #
    # Why this exists:
    # - A Mosaic `pallas_call` backward produces a custom-call `dlogits` which
    #   can force materialization and block XLA fusion on TPU.
    # - A pure-JAX backward expresses `dlogits` in standard HLO ops which can
    #   sometimes fuse into downstream matmuls, improving peak memory.
    #
    # Allowed values: "pallas", "jax".
    bwd_impl: str = "pallas"


@dataclass(frozen=True)
class GRPOKernelShardingSpec:
    """Logical axis names for shard_map wrapping.

    Contract:
    - Batch is typically sharded across ("dp","fsdp") and replicated over "tp".
    - If the vocab axis is sharded (e.g. tensor-parallel LM head), set
      `vocab_axis="tp"` and use the sharded wrapper (see this module's
      `grpo_per_token_loss_*` APIs).
    """

    batch_axes: tuple[str, ...] = ("dp", "fsdp")
    vocab_axis: str | None = None


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def _pad_vocab(logits: Any, *, block_size: int):
    import jax.numpy as jnp

    vocab = int(logits.shape[-1])
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    pad = (-vocab) % block_size
    if pad == 0:
        return logits, vocab
    logits = jnp.pad(
        logits,
        ((0, 0), (0, 0), (0, pad)),
        constant_values=jnp.finfo(logits.dtype).min,
    )
    return logits, vocab


def _pad_time(x: Any, *, time_block: int, pad_value: Any):
    import jax.numpy as jnp

    if time_block <= 0:
        raise ValueError("time_block must be > 0")
    time = int(x.shape[1])
    pad = (-time) % int(time_block)
    if pad == 0:
        return x, time
    pad_cfg = [(0, 0)] * int(x.ndim)
    pad_cfg[1] = (0, pad)
    return jnp.pad(x, pad_cfg, constant_values=pad_value), time


def _selective_log_softmax_reference(logits, chosen_ids, *, temperature: float) -> Any:
    import jax
    import jax.numpy as jnp

    per_token_logps = jnp.take_along_axis(
        jax.nn.log_softmax(logits, axis=-1),
        chosen_ids[..., None],
        axis=-1,
    )[..., 0]
    return per_token_logps / float(temperature)


def grpo_per_token_loss_reference(
    *,
    logits: Any,
    chosen_ids: Any,
    old_per_token_logps: Any,
    advantages: Any,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    temperature: float = 1.0,
) -> tuple[Any, Any]:
    """Reference GRPO per-token loss (JAX ops).

    Contract:
      - logits: [B, T, V] (aligned with chosen_ids / old_per_token_logps)
      - chosen_ids: [B, T] token ids for each position
      - old_per_token_logps: [B, T]
      - advantages: [B]
      - temperature semantics in this repo: `log_softmax(logits) / temperature`
        (note: this is NOT the same as `log_softmax(logits / temperature)`).

    Returns:
      - per_token_loss: [B, T] float32
      - per_token_logps: [B, T] float32
    """
    import jax.numpy as jnp

    per_token_logps = _selective_log_softmax_reference(logits, chosen_ids, temperature=temperature)

    ratio = jnp.exp(per_token_logps - old_per_token_logps)
    clipped_ratio = jnp.clip(ratio, 1.0 - float(epsilon_low), 1.0 + float(epsilon_high))

    advantages = advantages.astype(jnp.float32)
    per_token_loss1 = ratio * advantages[..., None]
    per_token_loss2 = clipped_ratio * advantages[..., None]
    per_token_loss = -jnp.minimum(per_token_loss1, per_token_loss2)

    return per_token_loss.astype(jnp.float32), per_token_logps.astype(jnp.float32)


def _grpo_pallas_fwd(
    *,
    logits: Any,
    chosen_ids: Any,
    old_per_token_logps: Any,
    advantages: Any,
    cfg: GRPOKernelConfig,
    interpret: bool,
    debug: bool,
    use_self_old: bool,
) -> tuple[Any, Any, Any, int]:
    import jax
    import jax.numpy as jnp

    # Two-pass design:
    # 1) Pallas computes per-vocab-tile softmax stats in parallel:
    #      - tile_max, tile_sumexp, tile_chosen_logit (0 when chosen not in tile)
    # 2) JAX reduces across tiles to get global (max,sumexp,chosen_logit) and then
    #    computes logp + GRPO loss in regular HLO ops.
    #
    # Why: the previous streaming implementation used a sequential "arbitrary"
    # vocab axis, which is correct but can be much slower than XLA's parallel
    # softmax lowering on TPU.

    time_block = int(cfg.time_block)

    logits, original_vocab = _pad_vocab(logits, block_size=cfg.block_size)
    logits, original_time = _pad_time(logits, time_block=time_block, pad_value=0.0)
    chosen_ids, _ = _pad_time(chosen_ids, time_block=time_block, pad_value=-1)
    old_per_token_logps, _ = _pad_time(old_per_token_logps, time_block=time_block, pad_value=0.0)

    max_blocks, sum_blocks, chosen_blocks = _grpo_pallas_fwd_block_stats(
        logits=logits,
        chosen_ids=chosen_ids,
        cfg=cfg,
        interpret=interpret,
        debug=debug,
    )

    # Reduce across vocab tiles.
    max_global = jnp.max(max_blocks, axis=-1)
    sum_global = jnp.sum(sum_blocks * jnp.exp(max_blocks - max_global[..., None]), axis=-1)
    lse = max_global + jnp.log(sum_global)
    chosen_logit = jnp.sum(chosen_blocks, axis=-1)

    temperature = float(cfg.temperature)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    per_token_logps = (chosen_logit - lse) / temperature

    if use_self_old:
        old_per_token_logps = jax.lax.stop_gradient(per_token_logps)

    eps_low = float(cfg.epsilon_low)
    eps_high = float(cfg.epsilon_high)
    ratio = jnp.exp(per_token_logps - old_per_token_logps.astype(jnp.float32))
    clipped_ratio = jnp.clip(ratio, 1.0 - eps_low, 1.0 + eps_high)

    advantages = advantages.astype(jnp.float32)
    per_token_loss1 = ratio * advantages[..., None]
    per_token_loss2 = clipped_ratio * advantages[..., None]
    per_token_loss = -jnp.minimum(per_token_loss1, per_token_loss2)

    # Slice off any time padding (vocab padding stays folded into the tile reduce).
    return (
        per_token_loss[:, :original_time].astype(jnp.float32),
        per_token_logps[:, :original_time].astype(jnp.float32),
        lse[:, :original_time].astype(jnp.float32),
        original_vocab,
    )


def _grpo_pallas_fwd_block_stats(
    *,
    logits: Any,
    chosen_ids: Any,
    cfg: GRPOKernelConfig,
    interpret: bool,
    debug: bool,
) -> tuple[Any, Any, Any]:
    """Return per-vocab-tile softmax stats for logits-level GRPO.

    Outputs (all float32):
      - max_blocks: [B, T, K] where K=ceil(V/block_size)
      - sum_blocks: [B, T, K] sum(exp(logits - max_blocks)) within each tile
      - chosen_blocks: [B, T, K] chosen logit within tile (0.0 if chosen_id not in tile)
    """
    import functools

    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    time_block = int(cfg.time_block)

    logits, _ = _pad_time(logits, time_block=time_block, pad_value=0.0)
    chosen_ids, _ = _pad_time(chosen_ids, time_block=time_block, pad_value=-1)

    batch, time, vocab = (int(logits.shape[0]), int(logits.shape[1]), int(logits.shape[2]))
    time_blocks = int(time // time_block)
    block_size = int(cfg.block_size)
    blocks = _ceil_div(vocab, block_size)

    chosen_ids3 = chosen_ids[..., None]

    out_max = jax.ShapeDtypeStruct((batch, time, blocks), jnp.float32)
    out_sum = jax.ShapeDtypeStruct((batch, time, blocks), jnp.float32)
    out_chosen = jax.ShapeDtypeStruct((batch, time, blocks), jnp.float32)

    def kernel(logits_ref, chosen_ids_ref, out_max_ref, out_sum_ref, out_chosen_ref):
        pid_k = pl.program_id(2)

        logits_tile = logits_ref[0, :, :].astype(jnp.float32)
        tile_max = jnp.max(logits_tile, axis=-1)
        exp_tile = jnp.exp(logits_tile - tile_max[:, None])
        tile_sum = jnp.sum(exp_tile, axis=-1)

        idx = chosen_ids_ref[0, :, 0].astype(jnp.int32)
        block_start = pid_k * block_size
        offset = idx - block_start
        in_range = (offset >= 0) & (offset < block_size)
        safe_offset = jnp.where(in_range, offset, 0).astype(jnp.int32)
        chosen_val = jnp.take_along_axis(logits_tile, safe_offset[:, None], axis=-1)[:, 0]
        chosen_val = chosen_val * in_range.astype(jnp.float32)

        out_max_ref[0, :, 0] = tile_max.astype(out_max_ref.dtype)
        out_sum_ref[0, :, 0] = tile_sum.astype(out_sum_ref.dtype)
        out_chosen_ref[0, :, 0] = chosen_val.astype(out_chosen_ref.dtype)

    call = pl.pallas_call(
        functools.partial(kernel),
        out_shape=(out_max, out_sum, out_chosen),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((1, time_block, block_size), lambda b, t, k: (b, t, k)),
                pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b, t, 0)),
            ],
            out_specs=[
                pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b, t, k)),
                pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b, t, k)),
                pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b, t, k)),
            ],
            grid=(batch, time_blocks, blocks),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel")),
        interpret=interpret,
        debug=bool(debug),
    )

    max3, sum3, chosen3 = call(logits, chosen_ids3)
    return max3, sum3, chosen3


def _grpo_pallas_fwd_stats(
    *,
    logits: Any,
    chosen_ids: Any,
    cfg: GRPOKernelConfig,
    interpret: bool,
    debug: bool,
) -> tuple[Any, Any, Any]:
    """Forward pass that returns local softmax stats for vocab-sharded reduce.

    Returns (all float32):
      - per_token_max: [B, T]
      - per_token_sumexp: [B, T] where sumexp is computed in the stabilized
        space (i.e. Σ exp(logits - max)).
      - per_token_chosen_logit: [B, T] (0.0 when chosen_id is not in-shard)
    """
    import jax.numpy as jnp

    logits, _original_vocab = _pad_vocab(logits, block_size=cfg.block_size)
    logits, original_time = _pad_time(logits, time_block=int(cfg.time_block), pad_value=0.0)
    chosen_ids, _ = _pad_time(chosen_ids, time_block=int(cfg.time_block), pad_value=-1)

    max_blocks, sum_blocks, chosen_blocks = _grpo_pallas_fwd_block_stats(
        logits=logits,
        chosen_ids=chosen_ids,
        cfg=cfg,
        interpret=interpret,
        debug=debug,
    )

    max_blocks = max_blocks[:, :original_time, :]
    sum_blocks = sum_blocks[:, :original_time, :]
    chosen_blocks = chosen_blocks[:, :original_time, :]

    max_local = jnp.max(max_blocks, axis=-1)
    sum_local = jnp.sum(sum_blocks * jnp.exp(max_blocks - max_local[..., None]), axis=-1)
    chosen_local = jnp.sum(chosen_blocks, axis=-1)
    return max_local.astype(jnp.float32), sum_local.astype(jnp.float32), chosen_local.astype(jnp.float32)


def _grpo_pallas_bwd(
    *,
    dloss: Any,
    logits: Any,
    chosen_ids: Any,
    old_per_token_logps: Any,
    advantages: Any,
    per_token_logps: Any,
    lse: Any,
    cfg: GRPOKernelConfig,
    interpret: bool,
    debug: bool,
    original_vocab: int,
    use_self_old: bool,
) -> Any:
    import functools

    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    time_block = int(cfg.time_block)
    original_time = int(dloss.shape[1])

    logits, _ = _pad_vocab(logits, block_size=cfg.block_size)
    logits, _ = _pad_time(logits, time_block=time_block, pad_value=0.0)
    chosen_ids, _ = _pad_time(chosen_ids, time_block=time_block, pad_value=0)
    old_per_token_logps, _ = _pad_time(old_per_token_logps, time_block=time_block, pad_value=0.0)
    per_token_logps, _ = _pad_time(per_token_logps, time_block=time_block, pad_value=0.0)
    lse, _ = _pad_time(lse, time_block=time_block, pad_value=0.0)
    dloss, _ = _pad_time(dloss, time_block=time_block, pad_value=0.0)

    batch, time, vocab = (int(logits.shape[0]), int(logits.shape[1]), int(logits.shape[2]))
    time_blocks = int(time // time_block)
    block_size = int(cfg.block_size)
    blocks = _ceil_div(vocab, block_size)

    eps_low = float(cfg.epsilon_low)
    eps_high = float(cfg.epsilon_high)
    temperature = float(cfg.temperature)
    use_self_old = bool(use_self_old)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    out_dlogits = jax.ShapeDtypeStruct((batch, time, vocab), logits.dtype)

    chosen_ids3 = chosen_ids[..., None]
    old_logps3 = old_per_token_logps[..., None]
    logps3 = per_token_logps[..., None]
    lse3 = lse[..., None]
    dloss3 = dloss[..., None]

    # TPU Pallas lowering has constraints on small block shapes; see the
    # forward kernel for details.
    advantages2 = advantages[:, None]

    def kernel(
        logits_ref,
        chosen_ids_ref,
        old_logps_ref,
        advantages_ref,
        logps_ref,
        lse_ref,
        dloss_ref,
        dlogits_ref,
    ):
        pid_b = pl.program_id(0)
        pid_k = pl.program_id(2)

        logits_tile = logits_ref[0, :, :].astype(jnp.float32)

        idx = chosen_ids_ref[0, :, 0].astype(jnp.int32)
        block_start = pid_k * block_size
        offset = idx - block_start
        lane_ids = jnp.arange(block_size, dtype=jnp.int32)[None, :]
        onehot = (lane_ids == offset[:, None]).astype(jnp.float32)

        lse_val = lse_ref[0, :, 0].astype(jnp.float32)
        probs = jnp.exp(logits_tile - lse_val[:, None])

        logp = logps_ref[0, :, 0].astype(jnp.float32)
        if use_self_old:
            ratio = jnp.ones_like(logp)
            clipped_ratio = ratio
        else:
            old_logp = old_logps_ref[0, :, 0].astype(jnp.float32)
            ratio = jnp.exp(logp - old_logp)
            clipped_ratio = jnp.clip(ratio, 1.0 - eps_low, 1.0 + eps_high)

        advantage = advantages_ref[pid_b, 0].astype(jnp.float32)
        loss1 = ratio * advantage
        loss2 = clipped_ratio * advantage
        unclipped = loss2 >= loss1

        dlogp = -loss1 * unclipped.astype(jnp.float32)
        dlogp = dlogp * dloss_ref[0, :, 0].astype(jnp.float32)

        scale = dlogp / temperature
        dlogits = (onehot - probs) * scale[:, None]
        dlogits_ref[0, :, :] = dlogits.astype(dlogits_ref.dtype)

    call = pl.pallas_call(
        functools.partial(kernel),
        out_shape=out_dlogits,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((1, time_block, block_size), lambda b, t, k: (b, t, k)),
                pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b, t, 0)),
                pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b, t, 0)),
                pl.BlockSpec((batch, 1), lambda b, t, k: (0, 0)),
                pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b, t, 0)),
                pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b, t, 0)),
                pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b, t, 0)),
            ],
            out_specs=pl.BlockSpec((1, time_block, block_size), lambda b, t, k: (b, t, k)),
            grid=(batch, time_blocks, blocks),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel")),
        interpret=interpret,
        debug=bool(debug),
    )

    dlogits = call(
        logits,
        chosen_ids3,
        old_logps3,
        advantages2,
        logps3,
        lse3,
        dloss3,
    )
    return dlogits[:, :original_time, : int(original_vocab)]


def _grpo_jax_bwd(
    *,
    dloss: Any,
    logits: Any,
    chosen_ids: Any,
    old_per_token_logps: Any,
    advantages: Any,
    per_token_logps: Any,
    lse: Any,
    cfg: GRPOKernelConfig,
    use_self_old: bool,
) -> Any:
    import jax
    import jax.numpy as jnp

    eps_low = float(cfg.epsilon_low)
    eps_high = float(cfg.epsilon_high)
    temperature = float(cfg.temperature)
    use_self_old = bool(use_self_old)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    logits_f32 = logits.astype(jnp.float32)
    lse_f32 = lse.astype(jnp.float32)
    logp_f32 = per_token_logps.astype(jnp.float32)
    dloss_f32 = dloss.astype(jnp.float32)

    if use_self_old:
        old_logp_f32 = jax.lax.stop_gradient(logp_f32)
    else:
        old_logp_f32 = old_per_token_logps.astype(jnp.float32)

    ratio = jnp.exp(logp_f32 - old_logp_f32)
    clipped_ratio = jnp.clip(ratio, 1.0 - eps_low, 1.0 + eps_high)

    advantages_f32 = advantages.astype(jnp.float32)
    per_token_loss1 = ratio * advantages_f32[..., None]
    per_token_loss2 = clipped_ratio * advantages_f32[..., None]
    unclipped = per_token_loss2 >= per_token_loss1

    dlogp = (-per_token_loss1) * unclipped.astype(jnp.float32)
    dlogp = dlogp * dloss_f32
    scale = dlogp / float(temperature)

    probs = jnp.exp(logits_f32 - lse_f32[..., None])

    dlogits = (-probs) * scale[..., None]

    vocab = int(logits.shape[-1])
    chosen_ids_i32 = chosen_ids.astype(jnp.int32)
    valid = (chosen_ids_i32 >= 0) & (chosen_ids_i32 < vocab)
    safe_ids = jnp.where(valid, chosen_ids_i32, 0)
    # Avoid materializing a dense one-hot [B,T,V] tensor: scatter the per-token
    # `scale` into the chosen index.
    updates = (scale * valid.astype(jnp.float32)).astype(jnp.float32)
    b_idx = jnp.arange(dlogits.shape[0], dtype=jnp.int32)[:, None]
    t_idx = jnp.arange(dlogits.shape[1], dtype=jnp.int32)[None, :]
    dlogits = dlogits.at[b_idx, t_idx, safe_ids].add(updates)

    return dlogits.astype(logits.dtype)


def build_grpo_per_token_loss_pallas(
    cfg: GRPOKernelConfig,
    *,
    interpret: bool = False,
    debug: bool = False,
    use_self_old: bool = False,
):
    import jax
    import jax.numpy as jnp

    bwd_impl = str(getattr(cfg, "bwd_impl", "pallas") or "pallas").strip().lower()
    if bwd_impl not in {"pallas", "jax"}:
        raise ValueError("cfg.bwd_impl must be one of: pallas, jax")

    cfg = GRPOKernelConfig(
        block_size=int(cfg.block_size),
        time_block=int(cfg.time_block),
        epsilon_low=float(cfg.epsilon_low),
        epsilon_high=float(cfg.epsilon_high),
        temperature=float(cfg.temperature),
        bwd_impl=bwd_impl,
    )

    @jax.custom_vjp
    def _kernel(logits, chosen_ids, old_per_token_logps, advantages):
        per_token_loss, per_token_logps, _lse, _original_vocab = _grpo_pallas_fwd(
            logits=logits,
            chosen_ids=chosen_ids,
            old_per_token_logps=old_per_token_logps,
            advantages=advantages,
            cfg=cfg,
            interpret=interpret,
            debug=debug,
            use_self_old=bool(use_self_old),
        )
        return per_token_loss, per_token_logps

    def fwd(logits, chosen_ids, old_per_token_logps, advantages):
        per_token_loss, per_token_logps, lse, original_vocab = _grpo_pallas_fwd(
            logits=logits,
            chosen_ids=chosen_ids,
            old_per_token_logps=old_per_token_logps,
            advantages=advantages,
            cfg=cfg,
            interpret=interpret,
            debug=debug,
            use_self_old=bool(use_self_old),
        )
        outs = (per_token_loss, per_token_logps)
        res = (logits, chosen_ids, old_per_token_logps, advantages, per_token_logps, lse, original_vocab)
        return outs, res

    def bwd(res, g):
        logits, chosen_ids, old_per_token_logps, advantages, per_token_logps, lse, original_vocab = res
        dloss, _dlogp = g
        if cfg.bwd_impl == "jax":
            dlogits = _grpo_jax_bwd(
                dloss=dloss,
                logits=logits,
                chosen_ids=chosen_ids,
                old_per_token_logps=old_per_token_logps,
                advantages=advantages,
                per_token_logps=per_token_logps,
                lse=lse,
                cfg=cfg,
                use_self_old=bool(use_self_old),
            )
        else:
            dlogits = _grpo_pallas_bwd(
                dloss=dloss.astype(jnp.float32),
                logits=logits,
                chosen_ids=chosen_ids,
                old_per_token_logps=old_per_token_logps,
                advantages=advantages,
                per_token_logps=per_token_logps,
                lse=lse,
                cfg=cfg,
                interpret=interpret,
                debug=debug,
                original_vocab=int(original_vocab),
                use_self_old=bool(use_self_old),
            )
        return (dlogits, None, None, None)

    _kernel.defvjp(fwd, bwd)
    return _kernel


def build_grpo_per_token_loss_pallas_sharded(
    cfg: GRPOKernelConfig,
    *,
    mesh: Any,
    sharding: GRPOKernelShardingSpec | None = None,
    interpret: bool = False,
    debug: bool = False,
    check_vma: bool = False,
    use_self_old: bool = False,
):
    """Build a multi-device GRPO loss using `jax.shard_map`.

    Why this exists:
    - TPU Mosaic (Pallas) kernels are not SPMD auto-partitionable.
    - Wrapping the kernel with `jax.shard_map` makes the multi-device mapping
      explicit, similar to how SplashAttention is invoked in MaxText.

    Current scope:
    - Supports sharding the *batch* axis across `sharding.batch_axes`.
    - Assumes the vocab axis is replicated (`sharding.vocab_axis is None`).
      Vocab sharding is added as a separate step in this repo.
    """
    import functools

    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as PS

    sharding = sharding or GRPOKernelShardingSpec()
    batch_axes = tuple(str(ax) for ax in sharding.batch_axes if str(ax))
    vocab_axis = sharding.vocab_axis

    bwd_impl = str(getattr(cfg, "bwd_impl", "pallas") or "pallas").strip().lower()
    if bwd_impl not in {"pallas", "jax"}:
        raise ValueError("cfg.bwd_impl must be one of: pallas, jax")

    cfg = GRPOKernelConfig(
        block_size=int(cfg.block_size),
        time_block=int(cfg.time_block),
        epsilon_low=float(cfg.epsilon_low),
        epsilon_high=float(cfg.epsilon_high),
        temperature=float(cfg.temperature),
        bwd_impl=bwd_impl,
    )

    logits_spec = PS(batch_axes, None, vocab_axis) if vocab_axis is not None else PS(batch_axes, None, None)
    bt_spec = PS(batch_axes, None)
    b_spec = PS(batch_axes)

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(logits_spec, bt_spec, bt_spec, b_spec),
        out_specs=(bt_spec, bt_spec, bt_spec),
        check_vma=bool(check_vma),
    )
    def _sharded_fwd(logits, chosen_ids, old_per_token_logps, advantages):
        if vocab_axis is None:
            per_token_loss, per_token_logps, lse, _original_vocab = _grpo_pallas_fwd(
                logits=logits,
                chosen_ids=chosen_ids,
                old_per_token_logps=old_per_token_logps,
                advantages=advantages,
                cfg=cfg,
                interpret=interpret,
                debug=debug,
                use_self_old=bool(use_self_old),
            )
            return per_token_loss, per_token_logps, lse

        tp_index = jax.lax.axis_index(vocab_axis)
        vocab_per_shard = int(logits.shape[-1])
        vocab_start = tp_index * vocab_per_shard
        chosen_ids_local = chosen_ids.astype(jnp.int32) - jnp.asarray(vocab_start, dtype=jnp.int32)

        max_local, sum_local, chosen_local = _grpo_pallas_fwd_stats(
            logits=logits,
            chosen_ids=chosen_ids_local,
            cfg=cfg,
            interpret=interpret,
            debug=debug,
        )

        max_global = jax.lax.pmax(max_local, axis_name=vocab_axis)
        sum_global = jax.lax.psum(sum_local * jnp.exp(max_local - max_global), axis_name=vocab_axis)
        lse = max_global + jnp.log(sum_global)
        chosen_logit = jax.lax.psum(chosen_local, axis_name=vocab_axis)

        temperature = float(cfg.temperature)
        per_token_logps = (chosen_logit - lse) / temperature

        if use_self_old:
            old_per_token_logps = jax.lax.stop_gradient(per_token_logps)

        ratio = jnp.exp(per_token_logps - old_per_token_logps.astype(jnp.float32))
        clipped_ratio = jnp.clip(ratio, 1.0 - float(cfg.epsilon_low), 1.0 + float(cfg.epsilon_high))

        advantages = advantages.astype(jnp.float32)
        per_token_loss1 = ratio * advantages[..., None]
        per_token_loss2 = clipped_ratio * advantages[..., None]
        per_token_loss = -jnp.minimum(per_token_loss1, per_token_loss2)
        return per_token_loss.astype(jnp.float32), per_token_logps.astype(jnp.float32), lse.astype(jnp.float32)

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(bt_spec, logits_spec, bt_spec, bt_spec, b_spec, bt_spec, bt_spec),
        out_specs=logits_spec,
        check_vma=bool(check_vma),
    )
    def _sharded_bwd(dloss, logits, chosen_ids, old_per_token_logps, advantages, per_token_logps, lse):
        if vocab_axis is None:
            chosen_ids_local = chosen_ids
            original_vocab = int(logits.shape[-1])
        else:
            tp_index = jax.lax.axis_index(vocab_axis)
            vocab_per_shard = int(logits.shape[-1])
            vocab_start = tp_index * vocab_per_shard
            chosen_ids_local = chosen_ids.astype(jnp.int32) - jnp.asarray(vocab_start, dtype=jnp.int32)
            original_vocab = vocab_per_shard
        if cfg.bwd_impl == "jax":
            return _grpo_jax_bwd(
                dloss=dloss,
                logits=logits,
                chosen_ids=chosen_ids_local,
                old_per_token_logps=old_per_token_logps,
                advantages=advantages,
                per_token_logps=per_token_logps,
                lse=lse,
                cfg=cfg,
                use_self_old=bool(use_self_old),
            )
        return _grpo_pallas_bwd(
            dloss=dloss,
            logits=logits,
            chosen_ids=chosen_ids_local,
            old_per_token_logps=old_per_token_logps,
            advantages=advantages,
            per_token_logps=per_token_logps,
            lse=lse,
            cfg=cfg,
            interpret=interpret,
            debug=debug,
            original_vocab=original_vocab,
            use_self_old=bool(use_self_old),
        )

    @jax.custom_vjp
    def _kernel(logits, chosen_ids, old_per_token_logps, advantages):
        per_token_loss, per_token_logps, _lse = _sharded_fwd(logits, chosen_ids, old_per_token_logps, advantages)
        return per_token_loss, per_token_logps

    def fwd(logits, chosen_ids, old_per_token_logps, advantages):
        per_token_loss, per_token_logps, lse = _sharded_fwd(logits, chosen_ids, old_per_token_logps, advantages)
        outs = (per_token_loss, per_token_logps)
        res = (logits, chosen_ids, old_per_token_logps, advantages, per_token_logps, lse)
        return outs, res

    def bwd(res, g):
        logits, chosen_ids, old_per_token_logps, advantages, per_token_logps, lse = res
        dloss, _dlogp = g
        if cfg.bwd_impl == "jax" and vocab_axis is None:
            dlogits = _grpo_jax_bwd(
                dloss=dloss,
                logits=logits,
                chosen_ids=chosen_ids,
                old_per_token_logps=old_per_token_logps,
                advantages=advantages,
                per_token_logps=per_token_logps,
                lse=lse,
                cfg=cfg,
                use_self_old=bool(use_self_old),
            )
        else:
            dlogits = _sharded_bwd(
                dloss.astype(jnp.float32),
                logits,
                chosen_ids,
                old_per_token_logps,
                advantages,
                per_token_logps,
                lse,
            )
        return (dlogits, None, None, None)

    _kernel.defvjp(fwd, bwd)
    return _kernel


def grpo_per_token_loss_pallas_default_cfg() -> GRPOKernelConfig:
    return GRPOKernelConfig()


def grpo_per_token_loss_pallas(
    *,
    logits: Any,
    chosen_ids: Any,
    old_per_token_logps: Any,
    advantages: Any,
    cfg: GRPOKernelConfig | None = None,
    interpret: bool = False,
    debug: bool = False,
    use_self_old: bool = False,
) -> tuple[Any, Any]:
    if cfg is None:
        cfg = GRPOKernelConfig()
    fn = build_grpo_per_token_loss_pallas(cfg, interpret=interpret, debug=debug, use_self_old=bool(use_self_old))
    return fn(logits, chosen_ids, old_per_token_logps, advantages)


def grpo_per_token_loss_pallas_configured(
    cfg: GRPOKernelConfig,
    *,
    interpret: bool = False,
    debug: bool = False,
):
    fn = build_grpo_per_token_loss_pallas(cfg, interpret=interpret, debug=debug)

    def _fn(*, logits: Any, chosen_ids: Any, old_per_token_logps: Any, advantages: Any) -> tuple[Any, Any]:
        return fn(logits, chosen_ids, old_per_token_logps, advantages)

    return _fn


def grpo_per_token_loss_pallas_sharded(
    *,
    logits: Any,
    chosen_ids: Any,
    old_per_token_logps: Any,
    advantages: Any,
    mesh: Any,
    cfg: GRPOKernelConfig | None = None,
    sharding: GRPOKernelShardingSpec | None = None,
    interpret: bool = False,
    debug: bool = False,
    check_vma: bool = False,
    use_self_old: bool = False,
) -> tuple[Any, Any]:
    """Multi-device GRPO per-token loss/logp via `jax.shard_map`."""
    if cfg is None:
        cfg = GRPOKernelConfig()
    fn = build_grpo_per_token_loss_pallas_sharded(
        cfg,
        mesh=mesh,
        sharding=sharding,
        interpret=interpret,
        debug=debug,
        check_vma=check_vma,
        use_self_old=bool(use_self_old),
    )
    return fn(logits, chosen_ids, old_per_token_logps, advantages)


__all__ = [
    "GRPOKernelConfig",
    "GRPOKernelShardingSpec",
    "grpo_per_token_loss_reference",
    "grpo_per_token_loss_pallas",
    "grpo_per_token_loss_pallas_default_cfg",
    "grpo_per_token_loss_pallas_configured",
    "build_grpo_per_token_loss_pallas_sharded",
    "grpo_per_token_loss_pallas_sharded",
]
