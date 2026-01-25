from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CrossEntropyKernelConfig:
    """Pallas tiled cross-entropy over vocab.

    This kernel computes per-token negative log-likelihood (NLL) and per-token
    log-prob for the provided labels, plus a custom VJP that returns `dlogits`
    matching the reference `jax.nn.log_softmax` formulation.

    Notes:
    - Designed for large vocabs by streaming over `block_size` vocab tiles.
    - `temperature` scales logits as `logits / temperature` before softmax.
    """

    block_size: int = 2048
    time_block: int = 8
    ignore_index: int = -100
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


def cross_entropy_per_token_reference(
    logits: Any,
    labels: Any,
    *,
    ignore_index: int = -100,
    temperature: float = 1.0,
) -> tuple[Any, Any]:
    """Reference per-token CE/logp (JAX ops).

    Contract:
      - logits: [B, T, V]
      - labels: [B, T] (int32 ids), with ignore positions marked by `ignore_index`

    Returns:
      - per_token_loss: [B, T] float32 (0 for ignored tokens)
      - per_token_logps: [B, T] float32 (0 for ignored tokens)
    """
    import jax
    import jax.numpy as jnp

    logits = logits.astype(jnp.float32) / float(temperature)
    labels = labels.astype(jnp.int32)
    valid = labels != int(ignore_index)
    safe_labels = jnp.where(valid, labels, 0).astype(jnp.int32)

    per_token_logps = jnp.take_along_axis(
        jax.nn.log_softmax(logits, axis=-1),
        safe_labels[..., None],
        axis=-1,
    )[..., 0]
    per_token_logps = jnp.where(valid, per_token_logps, 0.0)
    per_token_loss = -per_token_logps
    return per_token_loss.astype(jnp.float32), per_token_logps.astype(jnp.float32)


def _ce_pallas_fwd(
    *,
    logits: Any,
    labels: Any,
    cfg: CrossEntropyKernelConfig,
    interpret: bool,
    debug: bool,
) -> tuple[Any, Any, Any, int]:
    import functools

    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    time_block = int(cfg.time_block)

    logits, original_vocab = _pad_vocab(logits, block_size=cfg.block_size)
    logits, original_time = _pad_time(logits, time_block=time_block, pad_value=0.0)
    labels, _ = _pad_time(labels, time_block=time_block, pad_value=int(cfg.ignore_index))

    batch, time, vocab = (int(logits.shape[0]), int(logits.shape[1]), int(logits.shape[2]))
    time_blocks = int(time // time_block)
    block_size = int(cfg.block_size)
    blocks = _ceil_div(vocab, block_size)

    labels3 = labels[..., None]

    out_loss = jax.ShapeDtypeStruct((batch, time, 1), jnp.float32)
    out_logp = jax.ShapeDtypeStruct((batch, time, 1), jnp.float32)
    out_lse = jax.ShapeDtypeStruct((batch, time, 1), jnp.float32)

    ignore_index = int(cfg.ignore_index)
    temperature = float(cfg.temperature)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    def kernel(
        logits_ref,
        labels_ref,
        out_loss_ref,
        out_logp_ref,
        out_lse_ref,
        max_ref,
        sum_ref,
        chosen_ref,
    ):
        pid_k = pl.program_id(2)

        @pl.when(pid_k == 0)
        def init():
            max_ref[:] = jnp.full((time_block,), jnp.finfo(jnp.float32).min, dtype=jnp.float32)
            sum_ref[:] = jnp.zeros((time_block,), dtype=jnp.float32)
            chosen_ref[:] = jnp.zeros((time_block,), dtype=jnp.float32)
            out_loss_ref[...] = jnp.zeros_like(out_loss_ref)
            out_logp_ref[...] = jnp.zeros_like(out_logp_ref)
            out_lse_ref[...] = jnp.zeros_like(out_lse_ref)

        logits_tile = logits_ref[0, :, :].astype(jnp.float32) / temperature
        label_idx = labels_ref[0, :, 0].astype(jnp.int32)

        block_start = pid_k * block_size
        offset = label_idx - block_start
        valid = label_idx != ignore_index
        in_range = valid & (offset >= 0) & (offset < block_size)

        lane_ids = jnp.arange(block_size, dtype=jnp.int32)[None, :]
        onehot = lane_ids == offset[:, None]
        chosen_val = jnp.sum(jnp.where(onehot, logits_tile, 0.0), axis=-1)
        chosen_ref[:] = jnp.where(in_range, chosen_val, chosen_ref[:])

        prev_max = max_ref[:]
        prev_sum = sum_ref[:]
        tile_max = jnp.max(logits_tile, axis=-1)
        new_max = jnp.maximum(prev_max, tile_max)
        prev_sum = prev_sum * jnp.exp(prev_max - new_max)
        tile_sum = jnp.sum(jnp.exp(logits_tile - new_max[:, None]), axis=-1)
        new_sum = prev_sum + tile_sum
        max_ref[:] = new_max
        sum_ref[:] = new_sum

        @pl.when(pid_k == blocks - 1)
        def out():
            lse = max_ref[:] + jnp.log(sum_ref[:])
            logp = chosen_ref[:] - lse
            loss = lse - chosen_ref[:]
            logp = jnp.where(valid, logp, 0.0)
            loss = jnp.where(valid, loss, 0.0)
            out_lse_ref[0, :, 0] = lse.astype(out_lse_ref.dtype)
            out_logp_ref[0, :, 0] = logp.astype(out_logp_ref.dtype)
            out_loss_ref[0, :, 0] = loss.astype(out_loss_ref.dtype)

    call = pl.pallas_call(
        functools.partial(kernel),
        out_shape=(out_loss, out_logp, out_lse),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((1, time_block, block_size), lambda b, t, k: (b, t, k)),
                pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b, t, 0)),
            ],
            out_specs=[
                pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b, t, 0)),
                pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b, t, 0)),
                pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b, t, 0)),
            ],
            grid=(batch, time_blocks, blocks),
            scratch_shapes=[
                pltpu.VMEM((time_block,), jnp.float32),
                pltpu.VMEM((time_block,), jnp.float32),
                pltpu.VMEM((time_block,), jnp.float32),
            ],
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
        interpret=interpret,
        debug=bool(debug),
    )

    per_token_loss3, per_token_logps3, lse3 = call(logits, labels3)
    per_token_loss = per_token_loss3[:, :original_time, 0]
    per_token_logps = per_token_logps3[:, :original_time, 0]
    lse = lse3[:, :original_time, 0]
    return per_token_loss, per_token_logps, lse, original_vocab


def _ce_pallas_bwd(
    *,
    dloss: Any,
    logits: Any,
    labels: Any,
    lse: Any,
    cfg: CrossEntropyKernelConfig,
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
    original_time = int(dloss.shape[1])

    logits, _ = _pad_vocab(logits, block_size=cfg.block_size)
    logits, _ = _pad_time(logits, time_block=time_block, pad_value=0.0)
    labels, _ = _pad_time(labels, time_block=time_block, pad_value=int(cfg.ignore_index))
    lse, _ = _pad_time(lse, time_block=time_block, pad_value=0.0)
    dloss, _ = _pad_time(dloss, time_block=time_block, pad_value=0.0)

    batch, time, vocab = (int(logits.shape[0]), int(logits.shape[1]), int(logits.shape[2]))
    time_blocks = int(time // time_block)
    block_size = int(cfg.block_size)
    blocks = _ceil_div(vocab, block_size)

    ignore_index = int(cfg.ignore_index)
    temperature = float(cfg.temperature)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    out_dlogits = jax.ShapeDtypeStruct((batch, time, vocab), logits.dtype)

    labels3 = labels[..., None]
    lse3 = lse[..., None]
    dloss3 = dloss[..., None]

    def kernel(
        logits_ref,
        labels_ref,
        lse_ref,
        dloss_ref,
        dlogits_ref,
    ):
        pid_k = pl.program_id(2)

        logits_tile = logits_ref[0, :, :].astype(jnp.float32) / temperature

        label_idx = labels_ref[0, :, 0].astype(jnp.int32)
        valid = label_idx != ignore_index

        block_start = pid_k * block_size
        offset = label_idx - block_start

        lane_ids = jnp.arange(block_size, dtype=jnp.int32)[None, :]
        onehot = (lane_ids == offset[:, None]).astype(jnp.float32)
        onehot = onehot * valid.astype(jnp.float32)[:, None]

        lse_val = lse_ref[0, :, 0].astype(jnp.float32)
        probs = jnp.exp(logits_tile - lse_val[:, None])

        scale = dloss_ref[0, :, 0].astype(jnp.float32) * valid.astype(jnp.float32)
        scale = scale / temperature

        dlogits = (probs - onehot) * scale[:, None]
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
                pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b, t, 0)),
            ],
            out_specs=pl.BlockSpec((1, time_block, block_size), lambda b, t, k: (b, t, k)),
            grid=(batch, time_blocks, blocks),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel")),
        interpret=interpret,
        debug=bool(debug),
    )

    dlogits = call(logits, labels3, lse3, dloss3)
    return dlogits[:, :original_time, : int(original_vocab)]


def build_cross_entropy_per_token_pallas(
    cfg: CrossEntropyKernelConfig,
    *,
    interpret: bool = False,
    debug: bool = False,
):
    import jax
    import jax.numpy as jnp

    cfg = CrossEntropyKernelConfig(
        block_size=int(cfg.block_size),
        time_block=int(cfg.time_block),
        ignore_index=int(cfg.ignore_index),
        temperature=float(cfg.temperature),
    )

    @jax.custom_vjp
    def _kernel(logits, labels):
        per_token_loss, per_token_logps, _lse, _original_vocab = _ce_pallas_fwd(
            logits=logits,
            labels=labels,
            cfg=cfg,
            interpret=interpret,
            debug=debug,
        )
        return per_token_loss, per_token_logps

    def fwd(logits, labels):
        per_token_loss, per_token_logps, lse, original_vocab = _ce_pallas_fwd(
            logits=logits,
            labels=labels,
            cfg=cfg,
            interpret=interpret,
            debug=debug,
        )
        outs = (per_token_loss, per_token_logps)
        res = (logits, labels, lse, int(original_vocab))
        return outs, res

    def bwd(res, g):
        logits, labels, lse, original_vocab = res
        dloss, _dlogp = g
        dlogits = _ce_pallas_bwd(
            dloss=dloss.astype(jnp.float32),
            logits=logits,
            labels=labels,
            lse=lse,
            cfg=cfg,
            interpret=interpret,
            debug=debug,
            original_vocab=int(original_vocab),
        )
        return (dlogits, None)

    _kernel.defvjp(fwd, bwd)
    return _kernel


def cross_entropy_per_token_pallas_default_cfg() -> CrossEntropyKernelConfig:
    return CrossEntropyKernelConfig()


def cross_entropy_per_token_pallas(
    *,
    logits: Any,
    labels: Any,
    cfg: CrossEntropyKernelConfig | None = None,
    interpret: bool = False,
    debug: bool = False,
) -> tuple[Any, Any]:
    if cfg is None:
        cfg = CrossEntropyKernelConfig()
    fn = build_cross_entropy_per_token_pallas(cfg, interpret=interpret, debug=debug)
    return fn(logits, labels)


def cross_entropy_per_token_pallas_configured(
    cfg: CrossEntropyKernelConfig,
    *,
    interpret: bool = False,
    debug: bool = False,
):
    fn = build_cross_entropy_per_token_pallas(cfg, interpret=interpret, debug=debug)

    def _fn(*, logits: Any, labels: Any) -> tuple[Any, Any]:
        return fn(logits, labels)

    return _fn


__all__ = [
    "CrossEntropyKernelConfig",
    "cross_entropy_per_token_reference",
    "cross_entropy_per_token_pallas",
    "cross_entropy_per_token_pallas_configured",
    "cross_entropy_per_token_pallas_default_cfg",
]
