from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class EntropyKernelConfig:
    """Pallas tiled per-token entropy over vocab.

    Computes entropy for the categorical distribution:
      p = softmax(logits / temperature)

    without materializing `p` (streams over vocab tiles).
    """

    block_size: int = 2048
    time_block: int = 8
    temperature: float = 1.0


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


def entropy_per_token_reference(logits: Any, *, temperature: float = 1.0) -> Any:
    """Reference per-token entropy (stable logsumexp formulation)."""
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    x = logits.astype(jnp.float32) / temperature
    m = jnp.max(x, axis=-1)
    exp_x = jnp.exp(x - m[..., None])
    sumexp = jnp.sum(exp_x, axis=-1)
    sumexp_x = jnp.sum(exp_x * x, axis=-1)
    lse = m + jnp.log(sumexp)
    return (lse - (sumexp_x / sumexp)).astype(jnp.float32)


def _entropy_pallas_fwd(
    *,
    logits: Any,
    cfg: EntropyKernelConfig,
    interpret: bool,
    debug: bool,
) -> Any:
    import jax.numpy as jnp

    # Two-pass design:
    # 1) Pallas computes per-vocab-tile softmax stats (max,sumexp,sumexp_x) in parallel.
    # 2) JAX reduces across tiles to get global entropy.
    time_block = int(cfg.time_block)

    logits, _original_vocab = _pad_vocab(logits, block_size=cfg.block_size)
    logits, original_time = _pad_time(logits, time_block=time_block, pad_value=0.0)

    max_blocks, sum_blocks, sumx_blocks = _entropy_pallas_fwd_block_stats(
        logits=logits,
        cfg=cfg,
        interpret=interpret,
        debug=debug,
    )

    max_global = jnp.max(max_blocks, axis=-1)
    rescale = jnp.exp(max_blocks - max_global[..., None])
    sum_global = jnp.sum(sum_blocks * rescale, axis=-1)
    sumx_global = jnp.sum(sumx_blocks * rescale, axis=-1)
    lse = max_global + jnp.log(sum_global)
    entropy = lse - (sumx_global / sum_global)
    return entropy[:, :original_time].astype(jnp.float32)


def _entropy_pallas_fwd_block_stats(
    *,
    logits: Any,
    cfg: EntropyKernelConfig,
    interpret: bool,
    debug: bool,
) -> tuple[Any, Any, Any]:
    """Return per-vocab-tile softmax stats for entropy.

    Outputs (all float32):
      - max_blocks: [B, T, K] where K=ceil(V/block_size)
      - sum_blocks: [B, T, K] sum(exp(x - max_blocks)) within each tile
      - sumx_blocks: [B, T, K] sum(exp(x - max_blocks) * x) within each tile
    """
    import functools

    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    time_block = int(cfg.time_block)

    logits, _ = _pad_time(logits, time_block=time_block, pad_value=0.0)

    batch, time, vocab = (int(logits.shape[0]), int(logits.shape[1]), int(logits.shape[2]))
    time_blocks = int(time // time_block)
    block_size = int(cfg.block_size)
    blocks = _ceil_div(vocab, block_size)

    temperature = float(cfg.temperature)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    # Same TPU BlockSpec constraint as GRPO: keep trailing dim `1` by folding
    # the (usually non-128-aligned) `blocks` axis into the leading dimension.
    out_max = jax.ShapeDtypeStruct((batch * blocks, time, 1), jnp.float32)
    out_sum = jax.ShapeDtypeStruct((batch * blocks, time, 1), jnp.float32)
    out_sumx = jax.ShapeDtypeStruct((batch * blocks, time, 1), jnp.float32)

    def kernel(logits_ref, out_max_ref, out_sum_ref, out_sumx_ref):
        x_tile = logits_ref[0, :, :].astype(jnp.float32) / temperature
        tile_max = jnp.max(x_tile, axis=-1)
        exp_tile = jnp.exp(x_tile - tile_max[:, None])
        tile_sum = jnp.sum(exp_tile, axis=-1)
        tile_sumx = jnp.sum(exp_tile * x_tile, axis=-1)

        out_max_ref[0, :, 0] = tile_max.astype(out_max_ref.dtype)
        out_sum_ref[0, :, 0] = tile_sum.astype(out_sum_ref.dtype)
        out_sumx_ref[0, :, 0] = tile_sumx.astype(out_sumx_ref.dtype)

    call = pl.pallas_call(
        functools.partial(kernel),
        out_shape=(out_max, out_sum, out_sumx),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[pl.BlockSpec((1, time_block, block_size), lambda b, t, k: (b, t, k))],
            out_specs=[
                pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b * blocks + k, t, 0)),
                pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b * blocks + k, t, 0)),
                pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b * blocks + k, t, 0)),
            ],
            grid=(batch, time_blocks, blocks),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel")),
        interpret=interpret,
        debug=bool(debug),
    )

    max3, sum3, sumx3 = call(logits)
    max_blocks = max3[:, :, 0].reshape(batch, blocks, time).transpose(0, 2, 1)
    sum_blocks = sum3[:, :, 0].reshape(batch, blocks, time).transpose(0, 2, 1)
    sumx_blocks = sumx3[:, :, 0].reshape(batch, blocks, time).transpose(0, 2, 1)
    return max_blocks, sum_blocks, sumx_blocks


def _entropy_pallas_fwd_stats(
    *,
    logits: Any,
    cfg: EntropyKernelConfig,
    interpret: bool,
    debug: bool,
) -> tuple[Any, Any, Any]:
    """Forward pass that returns local softmax stats for vocab-sharded reduce.

    Returns (all float32):
      - per_token_max: [B, T]
      - per_token_sumexp: [B, T] where sumexp is computed in the stabilized
        space (i.e. Σ exp(x - max)).
      - per_token_sumexp_x: [B, T] where sumexp_x is Σ exp(x - max) * x.
    """
    import jax.numpy as jnp

    logits, _original_vocab = _pad_vocab(logits, block_size=cfg.block_size)
    logits, original_time = _pad_time(logits, time_block=int(cfg.time_block), pad_value=0.0)

    max_blocks, sum_blocks, sumx_blocks = _entropy_pallas_fwd_block_stats(
        logits=logits,
        cfg=cfg,
        interpret=interpret,
        debug=debug,
    )

    max_blocks = max_blocks[:, :original_time, :]
    sum_blocks = sum_blocks[:, :original_time, :]
    sumx_blocks = sumx_blocks[:, :original_time, :]

    max_local = jnp.max(max_blocks, axis=-1)
    rescale = jnp.exp(max_blocks - max_local[..., None])
    sum_local = jnp.sum(sum_blocks * rescale, axis=-1)
    sumx_local = jnp.sum(sumx_blocks * rescale, axis=-1)
    return max_local.astype(jnp.float32), sum_local.astype(jnp.float32), sumx_local.astype(jnp.float32)


@functools.partial(jax.custom_jvp, nondiff_argnames=("cfg", "interpret", "debug"))
def _entropy_per_token_pallas_jvp_safe(
    logits: Any,
    cfg: EntropyKernelConfig | None = None,
    interpret: Any = False,
    debug: bool = False,
) -> Any:
    if cfg is None:
        cfg = EntropyKernelConfig()
    cfg = EntropyKernelConfig(
        block_size=int(cfg.block_size),
        time_block=int(cfg.time_block),
        temperature=float(cfg.temperature),
    )
    return _entropy_pallas_fwd(logits=logits, cfg=cfg, interpret=interpret, debug=debug)


@_entropy_per_token_pallas_jvp_safe.defjvp
def _entropy_per_token_pallas_jvp_safe_jvp(
    cfg: EntropyKernelConfig | None = None,
    interpret: Any = False,
    debug: bool = False,
    primals=None,
    tangents=None,
):
    if primals is None or tangents is None:
        raise TypeError("custom_jvp rule must be called with primals and tangents")
    (logits,) = primals
    out = _entropy_per_token_pallas_jvp_safe(logits, cfg=cfg, interpret=interpret, debug=debug)
    return out, jnp.zeros_like(out)


def entropy_per_token_pallas(
    *,
    logits: Any,
    cfg: EntropyKernelConfig | None = None,
    interpret: Any = False,
    debug: bool = False,
) -> Any:
    return _entropy_per_token_pallas_jvp_safe(logits, cfg=cfg, interpret=interpret, debug=debug)


def entropy_per_token_pallas_sharded_impl(
    logits: Any,
    *,
    mesh: Any,
    cfg: EntropyKernelConfig | None = None,
    batch_axes: tuple[str, ...] = ("dp", "fsdp"),
    vocab_axis: str | None = None,
    interpret: Any = False,
    debug: bool = False,
    check_vma: bool = False,
) -> Any:
    import functools

    from jax.sharding import PartitionSpec as PS

    if cfg is None:
        cfg = EntropyKernelConfig()
    cfg = EntropyKernelConfig(
        block_size=int(cfg.block_size),
        time_block=int(cfg.time_block),
        temperature=float(cfg.temperature),
    )

    batch_axes = tuple(str(ax) for ax in batch_axes if str(ax))
    logits_spec = PS(batch_axes, None, vocab_axis) if vocab_axis is not None else PS(batch_axes, None, None)
    bt_spec = PS(batch_axes, None)

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(logits_spec,),
        out_specs=bt_spec,
        check_vma=bool(check_vma),
    )
    def _sharded_entropy(logits):
        if vocab_axis is None:
            return _entropy_pallas_fwd(logits=logits, cfg=cfg, interpret=interpret, debug=debug)

        max_local, sum_local, sumx_local = _entropy_pallas_fwd_stats(
            logits=logits,
            cfg=cfg,
            interpret=interpret,
            debug=debug,
        )
        max_global = jax.lax.pmax(max_local, axis_name=vocab_axis)
        rescale = jnp.exp(max_local - max_global)
        sum_global = jax.lax.psum(sum_local * rescale, axis_name=vocab_axis)
        sumx_global = jax.lax.psum(sumx_local * rescale, axis_name=vocab_axis)
        lse = max_global + jnp.log(sum_global)
        return (lse - (sumx_global / sum_global)).astype(jnp.float32)

    return _sharded_entropy(logits)


@functools.partial(
    jax.custom_jvp,
    nondiff_argnames=("mesh", "cfg", "batch_axes", "vocab_axis", "interpret", "debug", "check_vma"),
)
def entropy_per_token_pallas_sharded_jvp_safe(
    logits: Any,
    mesh: Any,
    cfg: EntropyKernelConfig | None = None,
    batch_axes: tuple[str, ...] = ("dp", "fsdp"),
    vocab_axis: str | None = None,
    interpret: Any = False,
    debug: bool = False,
    check_vma: bool = False,
) -> Any:
    return entropy_per_token_pallas_sharded_impl(
        logits,
        mesh=mesh,
        cfg=cfg,
        batch_axes=batch_axes,
        vocab_axis=vocab_axis,
        interpret=interpret,
        debug=debug,
        check_vma=check_vma,
    )


@entropy_per_token_pallas_sharded_jvp_safe.defjvp
def _entropy_per_token_pallas_sharded_jvp_safe_jvp(
    mesh: Any,
    cfg: EntropyKernelConfig | None = None,
    batch_axes: tuple[str, ...] = ("dp", "fsdp"),
    vocab_axis: str | None = None,
    interpret: Any = False,
    debug: bool = False,
    check_vma: bool = False,
    primals=None,
    tangents=None,
):
    if primals is None or tangents is None:
        raise TypeError("custom_jvp rule must be called with primals and tangents")
    (logits,) = primals
    out = entropy_per_token_pallas_sharded_jvp_safe(
        logits,
        mesh=mesh,
        cfg=cfg,
        batch_axes=batch_axes,
        vocab_axis=vocab_axis,
        interpret=interpret,
        debug=debug,
        check_vma=check_vma,
    )
    return out, jnp.zeros_like(out)


def entropy_per_token_pallas_sharded(
    *,
    logits: Any,
    mesh: Any,
    cfg: EntropyKernelConfig | None = None,
    batch_axes: tuple[str, ...] = ("dp", "fsdp"),
    vocab_axis: str | None = None,
    interpret: Any = False,
    debug: bool = False,
    check_vma: bool = False,
) -> Any:
    return entropy_per_token_pallas_sharded_jvp_safe(
        logits,
        mesh=mesh,
        cfg=cfg,
        batch_axes=batch_axes,
        vocab_axis=vocab_axis,
        interpret=interpret,
        debug=debug,
        check_vma=check_vma,
    )


__all__ = [
    "EntropyKernelConfig",
    "entropy_per_token_reference",
    "entropy_per_token_pallas",
    "entropy_per_token_pallas_sharded",
]
