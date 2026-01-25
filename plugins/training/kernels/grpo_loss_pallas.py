from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GRPOKernelConfig:
    block_size: int = 2048
    epsilon_low: float = 0.2
    epsilon_high: float = 0.2
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
) -> tuple[Any, Any, Any, int]:
    import functools

    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    time_block = 8

    logits, original_vocab = _pad_vocab(logits, block_size=cfg.block_size)
    logits, original_time = _pad_time(logits, time_block=time_block, pad_value=0.0)
    chosen_ids, _ = _pad_time(chosen_ids, time_block=time_block, pad_value=0)
    old_per_token_logps, _ = _pad_time(old_per_token_logps, time_block=time_block, pad_value=0.0)

    batch, time, vocab = (int(logits.shape[0]), int(logits.shape[1]), int(logits.shape[2]))
    time_blocks = int(time // time_block)
    block_size = int(cfg.block_size)
    blocks = _ceil_div(vocab, block_size)

    chosen_ids3 = chosen_ids[..., None]
    old_logps3 = old_per_token_logps[..., None]

    out_loss = jax.ShapeDtypeStruct((batch, time, 1), jnp.float32)
    out_logp = jax.ShapeDtypeStruct((batch, time, 1), jnp.float32)
    out_lse = jax.ShapeDtypeStruct((batch, time, 1), jnp.float32)

    eps_low = float(cfg.epsilon_low)
    eps_high = float(cfg.epsilon_high)
    temperature = float(cfg.temperature)

    def kernel(
        logits_ref,
        chosen_ids_ref,
        old_logps_ref,
        advantages_ref,
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

        logits_tile = logits_ref[0, :, :].astype(jnp.float32)

        idx = chosen_ids_ref[0, :, 0].astype(jnp.int32)
        block_start = pid_k * block_size
        offset = idx - block_start
        in_range = (offset >= 0) & (offset < block_size)
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
            out_lse_ref[0, :, 0] = lse.astype(out_lse_ref.dtype)

            logp = (chosen_ref[:] - lse) / temperature
            out_logp_ref[0, :, 0] = logp.astype(out_logp_ref.dtype)

            old_logp = old_logps_ref[0, :, 0].astype(jnp.float32)
            ratio = jnp.exp(logp - old_logp)
            clipped_ratio = jnp.clip(ratio, 1.0 - eps_low, 1.0 + eps_high)
            advantage = advantages_ref[0].astype(jnp.float32)
            loss1 = ratio * advantage
            loss2 = clipped_ratio * advantage
            per_token_loss = -jnp.minimum(loss1, loss2)
            out_loss_ref[0, :, 0] = per_token_loss.astype(out_loss_ref.dtype)

    call = pl.pallas_call(
        functools.partial(kernel),
        out_shape=(out_loss, out_logp, out_lse),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((1, time_block, block_size), lambda b, t, k: (b, t * time_block, k * block_size)),
                pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b, t * time_block, 0)),
                pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b, t * time_block, 0)),
                pl.BlockSpec((1,), lambda b, t, k: (b,)),
            ],
            out_specs=[
                pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b, t * time_block, 0)),
                pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b, t * time_block, 0)),
                pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b, t * time_block, 0)),
            ],
            grid=(batch, time_blocks, blocks),
            scratch_shapes=[
                pltpu.VMEM((time_block,), jnp.float32),
                pltpu.VMEM((time_block,), jnp.float32),
                pltpu.VMEM((time_block,), jnp.float32),
            ],
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
        interpret=bool(interpret),
        debug=bool(debug),
    )

    per_token_loss3, per_token_logps3, lse3 = call(logits, chosen_ids3, old_logps3, advantages)
    per_token_loss = per_token_loss3[:, :original_time, 0]
    per_token_logps = per_token_logps3[:, :original_time, 0]
    lse = lse3[:, :original_time, 0]
    return per_token_loss, per_token_logps, lse, original_vocab


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

    time_block = 8
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

    out_dlogits = jax.ShapeDtypeStruct((batch, time, vocab), logits.dtype)

    chosen_ids3 = chosen_ids[..., None]
    old_logps3 = old_per_token_logps[..., None]
    logps3 = per_token_logps[..., None]
    lse3 = lse[..., None]
    dloss3 = dloss[..., None]

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
        old_logp = old_logps_ref[0, :, 0].astype(jnp.float32)
        ratio = jnp.exp(logp - old_logp)
        clipped_ratio = jnp.clip(ratio, 1.0 - eps_low, 1.0 + eps_high)

        advantage = advantages_ref[0].astype(jnp.float32)
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
                pl.BlockSpec((1, time_block, block_size), lambda b, t, k: (b, t * time_block, k * block_size)),
                pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b, t * time_block, 0)),
                pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b, t * time_block, 0)),
                pl.BlockSpec((1,), lambda b, t, k: (b,)),
                pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b, t * time_block, 0)),
                pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b, t * time_block, 0)),
                pl.BlockSpec((1, time_block, 1), lambda b, t, k: (b, t * time_block, 0)),
            ],
            out_specs=pl.BlockSpec((1, time_block, block_size), lambda b, t, k: (b, t * time_block, k * block_size)),
            grid=(batch, time_blocks, blocks),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel")),
        interpret=bool(interpret),
        debug=bool(debug),
    )

    dlogits = call(
        logits,
        chosen_ids3,
        old_logps3,
        advantages,
        logps3,
        lse3,
        dloss3,
    )
    return dlogits[:, :original_time, : int(original_vocab)]

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
        epsilon_low=float(cfg.epsilon_low),
        epsilon_high=float(cfg.epsilon_high),
        temperature=float(cfg.temperature),
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


__all__ = [
    "GRPOKernelConfig",
    "grpo_per_token_loss_reference",
    "grpo_per_token_loss_pallas",
    "grpo_per_token_loss_pallas_default_cfg",
    "grpo_per_token_loss_pallas_configured",
]
