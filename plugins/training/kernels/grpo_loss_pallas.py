from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GRPOKernelConfig:
    block_size: int = 2048
    time_block: int = 8
    epsilon_low: float = 0.2
    epsilon_high: float = 0.2
    temperature: float = 1.0
    # Controls numerical behavior of logp and softmax gradient inside the kernel.
    #
    # - "f32": match the float32 reference implementation (tests/bench)
    # - "bf16": emulate legacy TrainGRPOModule bf16 log_softmax rounding:
    #     - logp is bf16-rounded before temperature scaling
    #     - exp/softmax path in the backward pass uses bf16 math
    compute_dtype: str = "f32"


def _resolve_compute_dtype(value: str):
    import jax.numpy as jnp

    v = str(value or "f32").strip().lower()
    if v in {"f32", "float32"}:
        return jnp.float32
    if v in {"bf16", "bfloat16"}:
        return jnp.bfloat16
    raise ValueError(f"Unsupported compute_dtype={value!r} (expected f32/bf16).")


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def _choose_index_subblock(block_size: int) -> int:
    """Pick a small static sub-block for dynamic-index patterns on TPU Mosaic.

    Mosaic has limitations for some dynamic offset patterns on very wide
    vectors. We implement chosen-token extraction (and the corresponding
    gradient update) using smaller static sub-blocks to avoid those paths.
    """
    block_size = int(block_size)
    for cand in (128, 64, 32, 16, 8):
        if block_size % cand == 0:
            return cand
    return block_size


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

    # Match the kernel's numerics: do the log-softmax in float32 even when
    # model logits are bf16/f16 (otherwise JAX returns bf16/f16 logps and
    # reference comparisons drift on large vocabs).
    logits = logits.astype(jnp.float32)
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


def _logsumexp_stats_pallas_full_vocab(
    *,
    logits: Any,
    cfg: GRPOKernelConfig,
    interpret: bool,
    debug: bool,
):
    """Compute per-(B,T) logsumexp stats over a vocab-multiple slice using Pallas.

    Expects:
      - logits: [B, T, V_full] with V_full % block_size == 0

    Returns:
      - max_val: [B, T] float32
      - sum_exp: [B, T] float32, where sum_exp = sum_v exp(logits_v - max_val)
      - sum_exp_logits: [B, T] float32, where sum_exp_logits = sum_v exp(logits_v - max_val) * logits_v
    """
    import functools

    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    time_block = int(cfg.time_block)
    if time_block <= 0:
        raise ValueError("time_block must be > 0")
    if time_block % 8 != 0:
        raise ValueError("time_block must be divisible by 8")

    softmax_dtype = _resolve_compute_dtype(cfg.compute_dtype)
    compute_dtype = softmax_dtype

    logits, original_time = _pad_time(logits, time_block=time_block, pad_value=0.0)

    batch, time, vocab = (int(logits.shape[0]), int(logits.shape[1]), int(logits.shape[2]))
    block_size = int(cfg.block_size)
    if vocab % block_size != 0:
        raise ValueError("logits vocab must be a multiple of block_size for the full-vocab stats kernel")

    time_blocks = int(time // time_block)
    blocks = int(vocab // block_size)
    if blocks <= 0:
        raise ValueError("stats kernel requires at least 1 full vocab block")

    out_max = jax.ShapeDtypeStruct((batch, time, 1), jnp.float32)
    out_sum = jax.ShapeDtypeStruct((batch, time, 1), jnp.float32)
    out_sum_logits = jax.ShapeDtypeStruct((batch, time, 1), jnp.float32)

    if blocks == 1:
        def kernel_single_block(logits_ref, out_max_ref, out_sum_ref, out_sum_logits_ref):
            logits_tile = logits_ref[0, :, :]
            if compute_dtype == jnp.float32 and logits_tile.dtype != jnp.float32:
                logits_tile = logits_tile.astype(jnp.float32)

            tile_max = jnp.max(logits_tile, axis=-1).astype(compute_dtype)
            tile_exp = jnp.exp(logits_tile - tile_max[:, None])
            tile_sum = jnp.sum(tile_exp, axis=-1).astype(compute_dtype)
            tile_sum_logits = jnp.sum(tile_exp * logits_tile, axis=-1).astype(compute_dtype)

            out_max_ref[0, :, 0] = tile_max.astype(out_max_ref.dtype)
            out_sum_ref[0, :, 0] = tile_sum.astype(out_sum_ref.dtype)
            out_sum_logits_ref[0, :, 0] = tile_sum_logits.astype(out_sum_logits_ref.dtype)

        call = pl.pallas_call(
            functools.partial(kernel_single_block),
            out_shape=(out_max, out_sum, out_sum_logits),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[pl.BlockSpec((1, time_block, block_size), lambda b, t, k: (b, t, k))],
                out_specs=[
                    pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b, t, 0)),
                    pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b, t, 0)),
                    pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b, t, 0)),
                ],
                grid=(batch, time_blocks, 1),
            ),
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel")),
            interpret=interpret,
            debug=bool(debug),
        )
        max3, sum3, sum_logits3 = call(logits)
        max_val = max3[:, :original_time, 0]
        sum_exp = sum3[:, :original_time, 0]
        sum_exp_logits = sum_logits3[:, :original_time, 0]
        return max_val, sum_exp, sum_exp_logits

    def kernel(
        logits_ref,
        out_max_ref,
        out_sum_ref,
        out_sum_logits_ref,
        max_ref,
        sum_ref,
        sum_logits_ref,
        _dummy_ref,
    ):
        pid_k = pl.program_id(2)

        @pl.when(pid_k == 0)
        def init():
            max_ref[:] = jnp.full((time_block,), jnp.finfo(compute_dtype).min, dtype=compute_dtype)
            sum_ref[:] = jnp.zeros((time_block,), dtype=compute_dtype)
            sum_logits_ref[:] = jnp.zeros((time_block,), dtype=compute_dtype)
            _dummy_ref[:] = jnp.zeros((time_block,), dtype=compute_dtype)
            out_max_ref[...] = jnp.zeros_like(out_max_ref)
            out_sum_ref[...] = jnp.zeros_like(out_sum_ref)
            out_sum_logits_ref[...] = jnp.zeros_like(out_sum_logits_ref)

        logits_tile = logits_ref[0, :, :]
        if compute_dtype == jnp.float32 and logits_tile.dtype != jnp.float32:
            logits_tile = logits_tile.astype(jnp.float32)

        prev_max = max_ref[:]
        prev_sum = sum_ref[:]
        prev_sum_logits = sum_logits_ref[:]

        tile_max = jnp.max(logits_tile, axis=-1)
        new_max = jnp.maximum(prev_max, tile_max)

        prev_scale = jnp.exp(prev_max - new_max)
        prev_sum = prev_sum * prev_scale
        prev_sum_logits = prev_sum_logits * prev_scale

        tile_exp = jnp.exp(logits_tile - new_max[:, None])
        tile_sum = jnp.sum(tile_exp, axis=-1).astype(compute_dtype)
        tile_sum_logits = jnp.sum(tile_exp * logits_tile, axis=-1).astype(compute_dtype)

        max_ref[:] = new_max
        sum_ref[:] = prev_sum + tile_sum
        sum_logits_ref[:] = prev_sum_logits + tile_sum_logits

        @pl.when(pid_k == blocks - 1)
        def out():
            out_max_ref[0, :, 0] = max_ref[:].astype(out_max_ref.dtype)
            out_sum_ref[0, :, 0] = sum_ref[:].astype(out_sum_ref.dtype)
            out_sum_logits_ref[0, :, 0] = sum_logits_ref[:].astype(out_sum_logits_ref.dtype)

    call = pl.pallas_call(
        functools.partial(kernel),
        out_shape=(out_max, out_sum, out_sum_logits),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[pl.BlockSpec((1, time_block, block_size), lambda b, t, k: (b, t, k))],
            out_specs=[
                pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b, t, 0)),
                pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b, t, 0)),
                pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b, t, 0)),
            ],
            grid=(batch, time_blocks, blocks),
            scratch_shapes=[
                pltpu.VMEM((time_block,), compute_dtype),
                pltpu.VMEM((time_block,), compute_dtype),
                pltpu.VMEM((time_block,), compute_dtype),
                pltpu.VMEM((time_block,), compute_dtype),
            ],
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
        interpret=interpret,
        debug=bool(debug),
    )

    max3, sum3, sum_logits3 = call(logits)
    max_val = max3[:, :original_time, 0]
    sum_exp = sum3[:, :original_time, 0]
    sum_exp_logits = sum_logits3[:, :original_time, 0]
    return max_val, sum_exp, sum_exp_logits


def _logsumexp_stats(
    *,
    logits: Any,
    cfg: GRPOKernelConfig,
    interpret: bool,
    debug: bool,
):
    """Compute per-(B,T) logsumexp stats without vocab padding.

    This avoids materializing a padded [B,T,V_pad] copy of `logits` (which can be
    ~1+ GiB for Qwen2.5-3B), by:
      1) running Pallas on the largest vocab-multiple prefix, and
      2) handling the small remainder ("tail") with JAX ops, then combining.
    """
    import jax.numpy as jnp

    softmax_dtype = _resolve_compute_dtype(cfg.compute_dtype)
    compute_dtype = softmax_dtype

    vocab = int(logits.shape[-1])
    block_size = int(cfg.block_size)
    if block_size <= 0:
        raise ValueError("block_size must be > 0")

    full_blocks = int(vocab // block_size)
    full_vocab = int(full_blocks * block_size)
    tail_vocab = int(vocab - full_vocab)

    # TPU bf16 log_softmax in JAX tends to use higher-precision internal math.
    # For the degenerate single-block case (V <= block_size), use a small JAX
    # float32 reduction to keep bf16-gradient parity tests stable.
    if full_blocks == 1 and tail_vocab == 0 and compute_dtype == jnp.bfloat16:
        logits_f32 = logits.astype(jnp.float32)
        max_val = jnp.max(logits_f32, axis=-1)
        exp = jnp.exp(logits_f32 - max_val[..., None])
        sum_exp = jnp.sum(exp, axis=-1)
        sum_logits = jnp.sum(exp * logits_f32, axis=-1)
        return max_val.astype(jnp.float32), sum_exp.astype(jnp.float32), sum_logits.astype(jnp.float32)

    if full_blocks > 0:
        max_full, sum_full, sum_logits_full = _logsumexp_stats_pallas_full_vocab(
            logits=logits[:, :, :full_vocab],
            cfg=cfg,
            interpret=interpret,
            debug=debug,
        )
    else:
        max_full = jnp.full(logits.shape[:2], jnp.finfo(compute_dtype).min, dtype=jnp.float32)
        sum_full = jnp.zeros(logits.shape[:2], dtype=jnp.float32)
        sum_logits_full = jnp.zeros(logits.shape[:2], dtype=jnp.float32)

    if tail_vocab > 0:
        tail = logits[:, :, full_vocab:]
        tail_compute = tail.astype(compute_dtype)
        max_tail = jnp.max(tail_compute, axis=-1).astype(jnp.float32)
        exp_tail = jnp.exp(tail_compute - max_tail.astype(compute_dtype)[..., None])
        sum_tail = jnp.sum(exp_tail, axis=-1).astype(jnp.float32)
        sum_logits_tail = jnp.sum(exp_tail * tail_compute, axis=-1).astype(jnp.float32)
    else:
        max_tail = jnp.full(logits.shape[:2], jnp.finfo(compute_dtype).min, dtype=jnp.float32)
        sum_tail = jnp.zeros(logits.shape[:2], dtype=jnp.float32)
        sum_logits_tail = jnp.zeros(logits.shape[:2], dtype=jnp.float32)

    max_full_c = max_full.astype(compute_dtype)
    max_tail_c = max_tail.astype(compute_dtype)
    sum_full_c = sum_full.astype(compute_dtype)
    sum_tail_c = sum_tail.astype(compute_dtype)
    sum_logits_full_c = sum_logits_full.astype(compute_dtype)
    sum_logits_tail_c = sum_logits_tail.astype(compute_dtype)

    max_all = jnp.maximum(max_full_c, max_tail_c)
    scale_full = jnp.exp(max_full_c - max_all)
    scale_tail = jnp.exp(max_tail_c - max_all)
    sum_all = (sum_full_c * scale_full + sum_tail_c * scale_tail).astype(jnp.float32)
    sum_logits_all = (sum_logits_full_c * scale_full + sum_logits_tail_c * scale_tail).astype(jnp.float32)
    return max_all.astype(jnp.float32), sum_all, sum_logits_all


def _grpo_pallas_fwd(
    *,
    logits: Any,
    chosen_ids: Any,
    old_per_token_logps: Any,
    advantages: Any,
    cfg: GRPOKernelConfig,
    interpret: bool,
    debug: bool,
) -> tuple[Any, Any, Any, Any, int]:
    import jax.numpy as jnp

    time_block = int(cfg.time_block)
    if time_block <= 0:
        raise ValueError("time_block must be > 0")
    if time_block % 8 != 0:
        raise ValueError("time_block must be divisible by 8")
    softmax_dtype = _resolve_compute_dtype(cfg.compute_dtype)
    quantize_logp = softmax_dtype == jnp.bfloat16
    compute_dtype = softmax_dtype

    original_vocab = int(logits.shape[-1])
    eps_low = float(cfg.epsilon_low)
    eps_high = float(cfg.epsilon_high)
    temperature = float(cfg.temperature)

    max_val, sum_exp, sum_exp_logits = _logsumexp_stats(logits=logits, cfg=cfg, interpret=interpret, debug=debug)
    lse = max_val + jnp.log(sum_exp)

    chosen = jnp.take_along_axis(logits, chosen_ids[..., None], axis=-1)[..., 0].astype(compute_dtype)
    logp_raw = chosen.astype(jnp.float32) - lse.astype(jnp.float32)
    if quantize_logp:
        logp_raw = logp_raw.astype(jnp.bfloat16).astype(jnp.float32)
    per_token_logps = logp_raw / temperature

    mean_logit = sum_exp_logits.astype(jnp.float32) / sum_exp.astype(jnp.float32)
    entropy = lse.astype(jnp.float32) - mean_logit

    old_logp = old_per_token_logps.astype(jnp.float32)
    ratio = jnp.exp(per_token_logps.astype(jnp.float32) - old_logp)
    clipped_ratio = jnp.clip(ratio, 1.0 - eps_low, 1.0 + eps_high)

    advantages = advantages.astype(jnp.float32)
    loss1 = ratio * advantages[..., None]
    loss2 = clipped_ratio * advantages[..., None]
    per_token_loss = -jnp.minimum(loss1, loss2)

    return (
        per_token_loss.astype(jnp.float32),
        per_token_logps.astype(jnp.float32),
        lse.astype(jnp.float32),
        entropy.astype(jnp.float32),
        original_vocab,
    )


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
) -> Any:
    import functools

    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    time_block = int(cfg.time_block)
    if time_block <= 0:
        raise ValueError("time_block must be > 0")
    if time_block % 8 != 0:
        raise ValueError("time_block must be divisible by 8")
    original_time = int(dloss.shape[1])
    compute_dtype = jnp.float32
    softmax_dtype = _resolve_compute_dtype(cfg.compute_dtype)
    # Match JAX bf16 `log_softmax` gradient more closely by using float32 softmax
    # math in the backward pass, even when forward logp is bf16-quantized.
    use_bf16_softmax = False

    chosen_ids, _ = _pad_time(chosen_ids, time_block=time_block, pad_value=0)
    old_per_token_logps, _ = _pad_time(old_per_token_logps, time_block=time_block, pad_value=0.0)
    per_token_logps, _ = _pad_time(per_token_logps, time_block=time_block, pad_value=0.0)
    lse, _ = _pad_time(lse, time_block=time_block, pad_value=0.0)
    dloss, _ = _pad_time(dloss, time_block=time_block, pad_value=0.0)

    block_size = int(cfg.block_size)
    index_subblock = _choose_index_subblock(block_size)
    num_index_subblocks = int(block_size // index_subblock)

    eps_low = float(cfg.epsilon_low)
    eps_high = float(cfg.epsilon_high)
    temperature = float(cfg.temperature)

    chosen_ids3 = chosen_ids[..., None]
    old_logps3 = old_per_token_logps[..., None]
    logps3 = per_token_logps[..., None]
    lse3 = lse[..., None]
    dloss3 = dloss[..., None]
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

        idx = chosen_ids_ref[0, :, 0].astype(jnp.int32)
        block_start = pid_k * block_size

        lse_val = lse_ref[0, :, 0].astype(jnp.float32)
        if use_bf16_softmax:
            lse_bf16 = lse_val.astype(jnp.bfloat16)

        logp = logps_ref[0, :, 0].astype(compute_dtype)
        old_logp = old_logps_ref[0, :, 0].astype(compute_dtype)
        ratio = jnp.exp(logp - old_logp).astype(jnp.float32)
        clipped_ratio = jnp.clip(ratio, 1.0 - eps_low, 1.0 + eps_high)

        advantage = advantages_ref[pid_b, 0].astype(jnp.float32)
        loss1 = ratio * advantage
        loss2 = clipped_ratio * advantage
        unclipped = loss2 >= loss1

        dlogp = -loss1 * unclipped.astype(jnp.float32)
        dlogp = dlogp * dloss_ref[0, :, 0].astype(jnp.float32)
        if use_bf16_softmax:
            scale_bf16 = (dlogp / temperature).astype(jnp.bfloat16)
        else:
            scale = dlogp / temperature
        lane_ids = jnp.arange(index_subblock, dtype=jnp.int32)[None, :]
        for sb in range(num_index_subblocks):
            if use_bf16_softmax:
                logits_sub = logits_ref[0, :, sb * index_subblock : (sb + 1) * index_subblock].astype(jnp.bfloat16)
                log_softmax_sub = logits_sub - lse_bf16[:, None]
                probs_sub = jnp.exp(log_softmax_sub)
                dlogits_sub = (-probs_sub) * scale_bf16[:, None]
            else:
                logits_sub = logits_ref[0, :, sb * index_subblock : (sb + 1) * index_subblock].astype(jnp.float32)
                log_softmax_sub = logits_sub - lse_val[:, None]
                probs_sub = jnp.exp(log_softmax_sub).astype(jnp.float32)
                dlogits_sub = (-probs_sub) * scale[:, None]

            sb_start = block_start + sb * index_subblock
            offset = idx - sb_start
            onehot = (lane_ids == offset[:, None]).astype(jnp.float32)
            if use_bf16_softmax:
                dlogits_sub = dlogits_sub + onehot.astype(jnp.bfloat16) * scale_bf16[:, None]
                dlogits_ref[0, :, sb * index_subblock : (sb + 1) * index_subblock] = dlogits_sub.astype(dlogits_ref.dtype)
            else:
                dlogits_sub = dlogits_sub + onehot * scale[:, None]
                dlogits_ref[0, :, sb * index_subblock : (sb + 1) * index_subblock] = dlogits_sub.astype(dlogits_ref.dtype)

    vocab = int(original_vocab)
    full_blocks = int(vocab // block_size)
    full_vocab = int(full_blocks * block_size)
    tail_vocab = int(vocab - full_vocab)

    dlogits_full = None
    if full_blocks > 0:
        logits_full, _ = _pad_time(logits[:, :, :full_vocab], time_block=time_block, pad_value=0.0)
        batch, time, _ = (int(logits_full.shape[0]), int(logits_full.shape[1]), int(logits_full.shape[2]))
        time_blocks = int(time // time_block)
        out_dlogits_full = jax.ShapeDtypeStruct((batch, time, full_vocab), logits.dtype)

        call = pl.pallas_call(
            functools.partial(kernel),
            out_shape=out_dlogits_full,
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
                grid=(batch, time_blocks, full_blocks),
            ),
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel")),
            interpret=interpret,
            debug=bool(debug),
        )

        dlogits_full = call(
            logits_full,
            chosen_ids3,
            old_logps3,
            advantages2,
            logps3,
            lse3,
            dloss3,
        )

    dlogits_tail = None
    if tail_vocab > 0:
        logits_tail, _ = _pad_time(logits[:, :, full_vocab:], time_block=time_block, pad_value=0.0)

        lse_val = lse.astype(jnp.float32)
        logp = per_token_logps.astype(jnp.float32)
        old_logp = old_per_token_logps.astype(jnp.float32)
        ratio = jnp.exp(logp - old_logp).astype(jnp.float32)
        clipped_ratio = jnp.clip(ratio, 1.0 - eps_low, 1.0 + eps_high)

        advantage = advantages.astype(jnp.float32)[:, None]
        loss1 = ratio * advantage
        loss2 = clipped_ratio * advantage
        unclipped = loss2 >= loss1

        dlogp = (-loss1) * unclipped.astype(jnp.float32)
        dlogp = dlogp * dloss.astype(jnp.float32)
        scale = dlogp / temperature

        if use_bf16_softmax:
            scale_c = scale.astype(jnp.bfloat16)
            lse_c = lse_val.astype(jnp.bfloat16)
            logits_c = logits_tail.astype(jnp.bfloat16)
            log_softmax = logits_c - lse_c[..., None]
            probs = jnp.exp(log_softmax)
            dlogits_tail = (-probs) * scale_c[..., None]
        else:
            scale_c = scale
            logits_c = logits_tail.astype(jnp.float32)
            log_softmax = logits_c - lse_val[..., None]
            probs = jnp.exp(log_softmax).astype(jnp.float32)
            dlogits_tail = (-probs) * scale_c[..., None]

        idx_tail = (chosen_ids.astype(jnp.int32) - int(full_vocab)).astype(jnp.int32)
        mask = (idx_tail >= 0) & (idx_tail < int(tail_vocab))
        bt = int(dlogits_tail.shape[0] * dlogits_tail.shape[1])
        rows = jnp.arange(bt, dtype=jnp.int32)
        cols = idx_tail.reshape((bt,))
        updates = scale_c.reshape((bt,))
        mask_flat = mask.reshape((bt,))
        cols = jnp.where(mask_flat, cols, 0)
        updates = jnp.where(mask_flat, updates, jnp.asarray(0, dtype=updates.dtype))
        dlogits_tail = dlogits_tail.reshape((bt, int(tail_vocab))).at[rows, cols].add(updates).reshape(
            dlogits_tail.shape
        )
        dlogits_tail = dlogits_tail.astype(logits.dtype)

    if dlogits_full is None:
        dlogits = dlogits_tail
    elif dlogits_tail is None:
        dlogits = dlogits_full
    else:
        dlogits = jnp.concatenate([dlogits_full, dlogits_tail], axis=-1)

    return dlogits[:, :original_time, :vocab]

def build_grpo_per_token_loss_pallas(
    cfg: GRPOKernelConfig,
    *,
    interpret: bool = False,
    debug: bool = False,
):
    import jax
    import jax.numpy as jnp

    cfg = GRPOKernelConfig(
        block_size=int(cfg.block_size),
        time_block=int(cfg.time_block),
        epsilon_low=float(cfg.epsilon_low),
        epsilon_high=float(cfg.epsilon_high),
        temperature=float(cfg.temperature),
        compute_dtype=str(getattr(cfg, "compute_dtype", "f32")),
    )

    @jax.custom_vjp
    def _kernel(logits, chosen_ids, old_per_token_logps, advantages):
        per_token_loss, per_token_logps, _lse, _entropy, _original_vocab = _grpo_pallas_fwd(
            logits=logits,
            chosen_ids=chosen_ids,
            old_per_token_logps=old_per_token_logps,
            advantages=advantages,
            cfg=cfg,
            interpret=interpret,
            debug=debug,
        )
        return per_token_loss, per_token_logps

    def fwd(logits, chosen_ids, old_per_token_logps, advantages):
        per_token_loss, per_token_logps, lse, _entropy, original_vocab = _grpo_pallas_fwd(
            logits=logits,
            chosen_ids=chosen_ids,
            old_per_token_logps=old_per_token_logps,
            advantages=advantages,
            cfg=cfg,
            interpret=interpret,
            debug=debug,
        )
        outs = (per_token_loss, per_token_logps)
        res = (logits, chosen_ids, old_per_token_logps, advantages, per_token_logps, lse, original_vocab)
        return outs, res

    def bwd(res, g):
        logits, chosen_ids, old_per_token_logps, advantages, per_token_logps, lse, original_vocab = res
        dloss, _dlogp = g
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
        )
        return (dlogits, None, None, None)

    _kernel.defvjp(fwd, bwd)
    return _kernel


def build_grpo_per_token_loss_pallas_with_entropy(
    cfg: GRPOKernelConfig,
    *,
    interpret: bool = False,
    debug: bool = False,
):
    """GRPO Pallas loss/logps + entropy (forward-only metric).

    Notes:
      - Entropy output is treated as a metric: its VJP is defined as zero.
      - Loss/logp numerics are controlled by `cfg.compute_dtype` as in the
        original kernel.
    """
    import jax
    import jax.numpy as jnp

    cfg = GRPOKernelConfig(
        block_size=int(cfg.block_size),
        time_block=int(cfg.time_block),
        epsilon_low=float(cfg.epsilon_low),
        epsilon_high=float(cfg.epsilon_high),
        temperature=float(cfg.temperature),
        compute_dtype=str(getattr(cfg, "compute_dtype", "f32")),
    )

    @jax.custom_vjp
    def _kernel(logits, chosen_ids, old_per_token_logps, advantages):
        per_token_loss, per_token_logps, _lse, per_token_entropy, _original_vocab = _grpo_pallas_fwd(
            logits=logits,
            chosen_ids=chosen_ids,
            old_per_token_logps=old_per_token_logps,
            advantages=advantages,
            cfg=cfg,
            interpret=interpret,
            debug=debug,
        )
        return per_token_loss, per_token_logps, jax.lax.stop_gradient(per_token_entropy)

    def fwd(logits, chosen_ids, old_per_token_logps, advantages):
        per_token_loss, per_token_logps, lse, per_token_entropy, original_vocab = _grpo_pallas_fwd(
            logits=logits,
            chosen_ids=chosen_ids,
            old_per_token_logps=old_per_token_logps,
            advantages=advantages,
            cfg=cfg,
            interpret=interpret,
            debug=debug,
        )
        outs = (per_token_loss, per_token_logps, per_token_entropy)
        res = (logits, chosen_ids, old_per_token_logps, advantages, per_token_logps, lse, original_vocab)
        return outs, res

    def bwd(res, g):
        logits, chosen_ids, old_per_token_logps, advantages, per_token_logps, lse, original_vocab = res
        dloss, _dlogp, _dentropy = g
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
        )
        return (dlogits, None, None, None)

    _kernel.defvjp(fwd, bwd)
    return _kernel


def build_grpo_per_token_loss_pallas_on_policy(
    cfg: GRPOKernelConfig,
    *,
    interpret: bool = False,
    debug: bool = False,
):
    """Pallas GRPO loss for the on-policy (old_logps implicit) PPO epoch.

    This matches the common training contract where the first PPO epoch sets:
      old_per_token_logps = stop_gradient(per_token_logps)
    so ratio = exp(logp - stop_grad(logp)) has value 1 but still carries the
    correct gradient signal.
    """
    import jax
    import jax.numpy as jnp

    cfg = GRPOKernelConfig(
        block_size=int(cfg.block_size),
        time_block=int(cfg.time_block),
        epsilon_low=float(cfg.epsilon_low),
        epsilon_high=float(cfg.epsilon_high),
        temperature=float(cfg.temperature),
        compute_dtype=str(getattr(cfg, "compute_dtype", "f32")),
    )
    eps_low = float(cfg.epsilon_low)
    eps_high = float(cfg.epsilon_high)

    def _compute_loss_from_logps(*, per_token_logps, old_per_token_logps, advantages):
        per_token_logps = per_token_logps.astype(jnp.float32)
        old_per_token_logps = old_per_token_logps.astype(jnp.float32)
        ratio = jnp.exp(per_token_logps - old_per_token_logps)
        clipped_ratio = jnp.clip(ratio, 1.0 - eps_low, 1.0 + eps_high)
        advantages = advantages.astype(jnp.float32)
        loss1 = ratio * advantages[..., None]
        loss2 = clipped_ratio * advantages[..., None]
        return (-jnp.minimum(loss1, loss2)).astype(jnp.float32)

    @jax.custom_vjp
    def _kernel(logits, chosen_ids, advantages):
        dummy_old = jnp.zeros(chosen_ids.shape, dtype=jnp.float32)
        _loss_unused, per_token_logps, _lse, _entropy, _original_vocab = _grpo_pallas_fwd(
            logits=logits,
            chosen_ids=chosen_ids,
            old_per_token_logps=dummy_old,
            advantages=advantages,
            cfg=cfg,
            interpret=interpret,
            debug=debug,
        )
        old_per_token_logps = jax.lax.stop_gradient(per_token_logps)
        per_token_loss = _compute_loss_from_logps(
            per_token_logps=per_token_logps,
            old_per_token_logps=old_per_token_logps,
            advantages=advantages,
        )
        return per_token_loss, per_token_logps

    def fwd(logits, chosen_ids, advantages):
        dummy_old = jnp.zeros(chosen_ids.shape, dtype=jnp.float32)
        _loss_unused, per_token_logps, lse, _entropy, original_vocab = _grpo_pallas_fwd(
            logits=logits,
            chosen_ids=chosen_ids,
            old_per_token_logps=dummy_old,
            advantages=advantages,
            cfg=cfg,
            interpret=interpret,
            debug=debug,
        )
        old_per_token_logps = jax.lax.stop_gradient(per_token_logps)
        per_token_loss = _compute_loss_from_logps(
            per_token_logps=per_token_logps,
            old_per_token_logps=old_per_token_logps,
            advantages=advantages,
        )
        outs = (per_token_loss, per_token_logps)
        res = (logits, chosen_ids, old_per_token_logps, advantages, per_token_logps, lse, original_vocab)
        return outs, res

    def bwd(res, g):
        logits, chosen_ids, old_per_token_logps, advantages, per_token_logps, lse, original_vocab = res
        dloss, _dlogp = g
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
        )
        return (dlogits, None, None)

    _kernel.defvjp(fwd, bwd)
    return _kernel


def build_grpo_per_token_loss_pallas_on_policy_with_entropy(
    cfg: GRPOKernelConfig,
    *,
    interpret: bool = False,
    debug: bool = False,
):
    """On-policy Pallas GRPO loss/logps + entropy (forward-only metric)."""
    import jax
    import jax.numpy as jnp

    cfg = GRPOKernelConfig(
        block_size=int(cfg.block_size),
        time_block=int(cfg.time_block),
        epsilon_low=float(cfg.epsilon_low),
        epsilon_high=float(cfg.epsilon_high),
        temperature=float(cfg.temperature),
        compute_dtype=str(getattr(cfg, "compute_dtype", "f32")),
    )
    eps_low = float(cfg.epsilon_low)
    eps_high = float(cfg.epsilon_high)

    def _compute_loss_from_logps(*, per_token_logps, old_per_token_logps, advantages):
        per_token_logps = per_token_logps.astype(jnp.float32)
        old_per_token_logps = old_per_token_logps.astype(jnp.float32)
        ratio = jnp.exp(per_token_logps - old_per_token_logps)
        clipped_ratio = jnp.clip(ratio, 1.0 - eps_low, 1.0 + eps_high)
        advantages = advantages.astype(jnp.float32)
        loss1 = ratio * advantages[..., None]
        loss2 = clipped_ratio * advantages[..., None]
        return (-jnp.minimum(loss1, loss2)).astype(jnp.float32)

    @jax.custom_vjp
    def _kernel(logits, chosen_ids, advantages):
        dummy_old = jnp.zeros(chosen_ids.shape, dtype=jnp.float32)
        _loss_unused, per_token_logps, _lse, per_token_entropy, _original_vocab = _grpo_pallas_fwd(
            logits=logits,
            chosen_ids=chosen_ids,
            old_per_token_logps=dummy_old,
            advantages=advantages,
            cfg=cfg,
            interpret=interpret,
            debug=debug,
        )
        old_per_token_logps = jax.lax.stop_gradient(per_token_logps)
        per_token_loss = _compute_loss_from_logps(
            per_token_logps=per_token_logps,
            old_per_token_logps=old_per_token_logps,
            advantages=advantages,
        )
        return per_token_loss, per_token_logps, jax.lax.stop_gradient(per_token_entropy)

    def fwd(logits, chosen_ids, advantages):
        dummy_old = jnp.zeros(chosen_ids.shape, dtype=jnp.float32)
        _loss_unused, per_token_logps, lse, per_token_entropy, original_vocab = _grpo_pallas_fwd(
            logits=logits,
            chosen_ids=chosen_ids,
            old_per_token_logps=dummy_old,
            advantages=advantages,
            cfg=cfg,
            interpret=interpret,
            debug=debug,
        )
        old_per_token_logps = jax.lax.stop_gradient(per_token_logps)
        per_token_loss = _compute_loss_from_logps(
            per_token_logps=per_token_logps,
            old_per_token_logps=old_per_token_logps,
            advantages=advantages,
        )
        outs = (per_token_loss, per_token_logps, per_token_entropy)
        res = (logits, chosen_ids, old_per_token_logps, advantages, per_token_logps, lse, original_vocab)
        return outs, res

    def bwd(res, g):
        logits, chosen_ids, old_per_token_logps, advantages, per_token_logps, lse, original_vocab = res
        dloss, _dlogp, _dentropy = g
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
        )
        return (dlogits, None, None)

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
) -> tuple[Any, Any]:
    if cfg is None:
        cfg = GRPOKernelConfig()
    fn = build_grpo_per_token_loss_pallas(cfg, interpret=interpret, debug=debug)
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


def grpo_per_token_loss_pallas_on_policy(
    *,
    logits: Any,
    chosen_ids: Any,
    advantages: Any,
    cfg: GRPOKernelConfig | None = None,
    interpret: bool = False,
    debug: bool = False,
) -> tuple[Any, Any]:
    """Compute per-token GRPO loss/logps with implicit old_logps=stop_grad(logps)."""
    if cfg is None:
        cfg = GRPOKernelConfig()
    fn = build_grpo_per_token_loss_pallas_on_policy(cfg, interpret=interpret, debug=debug)
    return fn(logits, chosen_ids, advantages)


def grpo_per_token_loss_pallas_with_entropy(
    *,
    logits: Any,
    chosen_ids: Any,
    old_per_token_logps: Any,
    advantages: Any,
    cfg: GRPOKernelConfig | None = None,
    interpret: bool = False,
    debug: bool = False,
) -> tuple[Any, Any, Any]:
    if cfg is None:
        cfg = GRPOKernelConfig()
    fn = build_grpo_per_token_loss_pallas_with_entropy(cfg, interpret=interpret, debug=debug)
    return fn(logits, chosen_ids, old_per_token_logps, advantages)


def grpo_per_token_loss_pallas_on_policy_with_entropy(
    *,
    logits: Any,
    chosen_ids: Any,
    advantages: Any,
    cfg: GRPOKernelConfig | None = None,
    interpret: bool = False,
    debug: bool = False,
) -> tuple[Any, Any, Any]:
    if cfg is None:
        cfg = GRPOKernelConfig()
    fn = build_grpo_per_token_loss_pallas_on_policy_with_entropy(cfg, interpret=interpret, debug=debug)
    return fn(logits, chosen_ids, advantages)


__all__ = [
    "GRPOKernelConfig",
    "grpo_per_token_loss_reference",
    "grpo_per_token_loss_pallas",
    "grpo_per_token_loss_pallas_default_cfg",
    "grpo_per_token_loss_pallas_configured",
    "grpo_per_token_loss_pallas_on_policy",
    "build_grpo_per_token_loss_pallas_on_policy",
    "build_grpo_per_token_loss_pallas_with_entropy",
    "build_grpo_per_token_loss_pallas_on_policy_with_entropy",
    "grpo_per_token_loss_pallas_with_entropy",
    "grpo_per_token_loss_pallas_on_policy_with_entropy",
]
