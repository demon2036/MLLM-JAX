from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GRPOLmHeadFusedConfig:
    """Config for a fused GRPO loss over the LM head.

    This kernel consumes:
      - hidden states `h` (shape `[B, T, D]`)
      - LM head weights `W` (shape `[D, V]`)

    and computes GRPO per-token loss + selective logp without materializing the
    full logits tensor `[B, T, V]` by streaming over vocab tiles.

    Notes:
    - Temperature semantics match this repo's GRPO reference:
        logp = log_softmax(logits)[y] / temperature
      (this is NOT the same as scaling logits by 1/temperature before softmax).
    """

    vocab_block_size: int = 2048
    epsilon_low: float = 0.2
    epsilon_high: float = 0.2
    temperature: float = 1.0


def grpo_per_token_loss_lm_head_reference(
    *,
    hidden_states: Any,
    lm_head_kernel: Any,
    chosen_ids: Any,
    old_per_token_logps: Any,
    advantages: Any,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    temperature: float = 1.0,
) -> tuple[Any, Any]:
    """Reference GRPO per-token loss from hidden states (materializes logits)."""
    import jax
    import jax.numpy as jnp

    from plugins.training.kernels.grpo_loss_pallas import grpo_per_token_loss_reference

    logits = jax.lax.dot_general(
        hidden_states,
        lm_head_kernel,
        (((hidden_states.ndim - 1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    logits = logits.astype(hidden_states.dtype)
    return grpo_per_token_loss_reference(
        logits=logits,
        chosen_ids=chosen_ids,
        old_per_token_logps=old_per_token_logps,
        advantages=advantages,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        temperature=temperature,
    )


def _selective_log_softmax_lm_head_streaming(
    *,
    hidden_states: Any,
    lm_head_kernel: Any,
    chosen_ids: Any,
    vocab_block_size: int,
) -> tuple[Any, Any]:
    """Compute selective logp + per-token logsumexp without `[B,T,V]` logits."""
    import jax
    import jax.numpy as jnp

    hidden_states = jnp.asarray(hidden_states)
    lm_head_kernel = jnp.asarray(lm_head_kernel)
    chosen_ids = jnp.asarray(chosen_ids)

    batch, time, hidden = (int(hidden_states.shape[0]), int(hidden_states.shape[1]), int(hidden_states.shape[2]))
    if lm_head_kernel.ndim != 2:
        raise ValueError(f"lm_head_kernel must be rank-2, got {lm_head_kernel.ndim=}")
    if int(lm_head_kernel.shape[0]) != hidden:
        raise ValueError(f"hidden dim mismatch: {hidden_states.shape=} vs {lm_head_kernel.shape=}")

    vocab = int(lm_head_kernel.shape[1])
    block = int(vocab_block_size)
    if block <= 0:
        raise ValueError("vocab_block_size must be > 0")

    # Flatten token dimension to simplify dot + indexing.
    tokens = int(batch * time)
    h2 = hidden_states.reshape(tokens, hidden)

    ids = chosen_ids.astype(jnp.int32).reshape(tokens)
    valid_id = (ids >= 0) & (ids < vocab)
    safe_ids = jnp.where(valid_id, ids, jnp.zeros((), jnp.int32))

    full_blocks = int(vocab // block)
    rem = int(vocab % block)

    def _dot(x, w):
        return jax.lax.dot_general(
            x,
            w,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )

    def scan_body(_, block_id):
        start = block_id * block

        w_blk = jax.lax.dynamic_slice(lm_head_kernel, (0, start), (hidden, block))
        logits = _dot(h2, w_blk).astype(jnp.float32)  # [tokens, block]

        tile_max = jnp.max(logits, axis=-1)  # [tokens]
        tile_sum = jnp.sum(jnp.exp(logits - tile_max[:, None]), axis=-1)  # [tokens]

        in_range = valid_id & (safe_ids >= start) & (safe_ids < start + block)
        offsets = jnp.where(in_range, safe_ids - start, jnp.zeros((), jnp.int32)).astype(jnp.int32)
        chosen_val = jnp.take_along_axis(logits, offsets[:, None], axis=-1)[:, 0]
        tile_chosen = jnp.where(in_range, chosen_val, jnp.zeros((), jnp.float32))
        return None, (tile_max, tile_sum, tile_chosen)

    _, (max_blocks, sum_blocks, chosen_blocks) = jax.lax.scan(
        scan_body,
        None,
        jnp.arange(full_blocks, dtype=jnp.int32),
    )

    if full_blocks:
        max_blocks = max_blocks.transpose(1, 0)  # [tokens, K]
        sum_blocks = sum_blocks.transpose(1, 0)  # [tokens, K]
        chosen_blocks = chosen_blocks.transpose(1, 0)  # [tokens, K]
    else:
        max_blocks = jnp.zeros((tokens, 0), dtype=jnp.float32)
        sum_blocks = jnp.zeros((tokens, 0), dtype=jnp.float32)
        chosen_blocks = jnp.zeros((tokens, 0), dtype=jnp.float32)

    if rem:
        start = full_blocks * block
        w_tail = lm_head_kernel[:, start:]
        logits = _dot(h2, w_tail).astype(jnp.float32)  # [tokens, rem]

        tail_max = jnp.max(logits, axis=-1)
        tail_sum = jnp.sum(jnp.exp(logits - tail_max[:, None]), axis=-1)

        in_range = valid_id & (safe_ids >= start)
        offsets = jnp.where(in_range, safe_ids - start, jnp.zeros((), jnp.int32)).astype(jnp.int32)
        chosen_val = jnp.take_along_axis(logits, offsets[:, None], axis=-1)[:, 0]
        tail_chosen = jnp.where(in_range, chosen_val, jnp.zeros((), jnp.float32))

        max_blocks = jnp.concatenate([max_blocks, tail_max[:, None]], axis=-1)
        sum_blocks = jnp.concatenate([sum_blocks, tail_sum[:, None]], axis=-1)
        chosen_blocks = jnp.concatenate([chosen_blocks, tail_chosen[:, None]], axis=-1)

    max_global = jnp.max(max_blocks, axis=-1)
    sum_global = jnp.sum(sum_blocks * jnp.exp(max_blocks - max_global[:, None]), axis=-1)
    lse = max_global + jnp.log(sum_global)
    chosen = jnp.sum(chosen_blocks, axis=-1)
    logp = chosen - lse

    logp = logp.reshape(batch, time).astype(jnp.float32)
    lse = lse.reshape(batch, time).astype(jnp.float32)
    return logp, lse


def grpo_per_token_loss_fused_lm_head_forward(
    *,
    hidden_states: Any,
    lm_head_kernel: Any,
    chosen_ids: Any,
    old_per_token_logps: Any,
    advantages: Any,
    cfg: GRPOLmHeadFusedConfig | None = None,
    use_self_old: bool = False,
) -> tuple[Any, Any, Any]:
    """Forward-only fused GRPO loss (returns loss, logp, lse).

    This is the forward used by the custom VJP implementation; it is also
    useful for microbenching forward numerics.
    """
    import jax
    import jax.numpy as jnp

    if cfg is None:
        cfg = GRPOLmHeadFusedConfig()
    cfg = GRPOLmHeadFusedConfig(
        vocab_block_size=int(cfg.vocab_block_size),
        epsilon_low=float(cfg.epsilon_low),
        epsilon_high=float(cfg.epsilon_high),
        temperature=float(cfg.temperature),
    )

    logp, lse = _selective_log_softmax_lm_head_streaming(
        hidden_states=hidden_states,
        lm_head_kernel=lm_head_kernel,
        chosen_ids=chosen_ids,
        vocab_block_size=int(cfg.vocab_block_size),
    )
    per_token_logps = logp / float(cfg.temperature)

    if use_self_old:
        old_per_token_logps = jax.lax.stop_gradient(per_token_logps)
    else:
        old_per_token_logps = jnp.asarray(old_per_token_logps, dtype=jnp.float32)

    ratio = jnp.exp(per_token_logps - old_per_token_logps)
    clipped_ratio = jnp.clip(ratio, 1.0 - float(cfg.epsilon_low), 1.0 + float(cfg.epsilon_high))

    advantages = jnp.asarray(advantages, dtype=jnp.float32)
    per_token_loss1 = ratio * advantages[..., None]
    per_token_loss2 = clipped_ratio * advantages[..., None]
    per_token_loss = -jnp.minimum(per_token_loss1, per_token_loss2)

    return per_token_loss.astype(jnp.float32), per_token_logps.astype(jnp.float32), lse.astype(jnp.float32)


def build_grpo_per_token_loss_fused_lm_head(
    cfg: GRPOLmHeadFusedConfig,
    *,
    use_self_old: bool = False,
):
    """Build a fused GRPO per-token loss with a custom VJP.

    Backward returns gradients for:
      - hidden_states
      - lm_head_kernel

    and returns None for non-differentiable inputs (ids/old_logps/advantages).
    """
    import jax
    import jax.numpy as jnp

    cfg = GRPOLmHeadFusedConfig(
        vocab_block_size=int(cfg.vocab_block_size),
        epsilon_low=float(cfg.epsilon_low),
        epsilon_high=float(cfg.epsilon_high),
        temperature=float(cfg.temperature),
    )

    block = int(cfg.vocab_block_size)
    eps_low = float(cfg.epsilon_low)
    eps_high = float(cfg.epsilon_high)
    temperature = float(cfg.temperature)
    use_self_old = bool(use_self_old)
    if block <= 0:
        raise ValueError("cfg.vocab_block_size must be > 0")
    if temperature <= 0:
        raise ValueError("cfg.temperature must be > 0")

    def _dot(x, w):
        return jax.lax.dot_general(
            x,
            w,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )

    @jax.custom_vjp
    def _kernel(hidden_states, lm_head_kernel, chosen_ids, old_per_token_logps, advantages):
        per_loss, per_logps, _lse = grpo_per_token_loss_fused_lm_head_forward(
            hidden_states=hidden_states,
            lm_head_kernel=lm_head_kernel,
            chosen_ids=chosen_ids,
            old_per_token_logps=old_per_token_logps,
            advantages=advantages,
            cfg=cfg,
            use_self_old=use_self_old,
        )
        return per_loss, per_logps

    def fwd(hidden_states, lm_head_kernel, chosen_ids, old_per_token_logps, advantages):
        per_loss, per_logps, lse = grpo_per_token_loss_fused_lm_head_forward(
            hidden_states=hidden_states,
            lm_head_kernel=lm_head_kernel,
            chosen_ids=chosen_ids,
            old_per_token_logps=old_per_token_logps,
            advantages=advantages,
            cfg=cfg,
            use_self_old=use_self_old,
        )
        outs = (per_loss, per_logps)
        res = (hidden_states, lm_head_kernel, chosen_ids, old_per_token_logps, advantages, per_logps, lse)
        return outs, res

    def bwd(res, g):
        hidden_states, lm_head_kernel, chosen_ids, old_per_token_logps, advantages, per_logps, lse = res
        dloss, _dlogps = g

        hidden_states = jnp.asarray(hidden_states)
        lm_head_kernel = jnp.asarray(lm_head_kernel)
        chosen_ids = jnp.asarray(chosen_ids)
        old_per_token_logps = jnp.asarray(old_per_token_logps)
        advantages = jnp.asarray(advantages)
        per_logps = jnp.asarray(per_logps)
        lse = jnp.asarray(lse)
        dloss = jnp.asarray(dloss)

        batch, time, hidden = (int(hidden_states.shape[0]), int(hidden_states.shape[1]), int(hidden_states.shape[2]))
        vocab = int(lm_head_kernel.shape[1])

        tokens = int(batch * time)
        h2 = hidden_states.reshape(tokens, hidden)
        ids = chosen_ids.astype(jnp.int32).reshape(tokens)
        valid_id = (ids >= 0) & (ids < vocab)
        safe_ids = jnp.where(valid_id, ids, jnp.zeros((), jnp.int32))

        lse2 = lse.reshape(tokens).astype(jnp.float32)
        logp2 = per_logps.reshape(tokens).astype(jnp.float32)
        dloss2 = dloss.reshape(tokens).astype(jnp.float32)

        if use_self_old:
            old2 = jax.lax.stop_gradient(logp2)
        else:
            old2 = old_per_token_logps.reshape(tokens).astype(jnp.float32)

        ratio = jnp.exp(logp2 - old2)
        clipped_ratio = jnp.clip(ratio, 1.0 - eps_low, 1.0 + eps_high)
        adv2 = advantages.astype(jnp.float32).reshape(batch, 1)
        adv_tok = jnp.broadcast_to(adv2, (batch, time)).reshape(tokens)

        loss1 = ratio * adv_tok
        loss2 = clipped_ratio * adv_tok
        unclipped = loss2 >= loss1

        dlogp = (-loss1) * unclipped.astype(jnp.float32)
        dlogp = dlogp * dloss2
        scale = dlogp / float(temperature)

        full_blocks = int(vocab // block)
        rem = int(vocab % block)

        def scan_body(dh_carry, block_id):
            start = block_id * block
            w_blk = jax.lax.dynamic_slice(lm_head_kernel, (0, start), (hidden, block))
            logits = _dot(h2, w_blk).astype(jnp.float32)  # [tokens, block]
            probs = jnp.exp(logits - lse2[:, None])

            dlogits_soft = (-probs) * scale[:, None]
            dh = dh_carry + _dot(dlogits_soft, w_blk.T).astype(jnp.float32)
            dW_soft = _dot(h2.T, dlogits_soft).astype(jnp.float32)

            in_range = valid_id & (safe_ids >= start) & (safe_ids < start + block)
            offsets = jnp.where(in_range, safe_ids - start, jnp.zeros((), jnp.int32)).astype(jnp.int32)
            in_range_f = in_range.astype(jnp.float32)

            # +scale at chosen index -> dh += scale * W[:, y]
            w_sel = jnp.take(w_blk.T, offsets, axis=0).astype(jnp.float32)
            dh = dh + w_sel * (scale * in_range_f)[:, None]

            # dW[:, y] += h * scale
            updates = (h2.astype(jnp.float32) * (scale * in_range_f)[:, None]).astype(jnp.float32)
            seg = _segment_sum(updates, offsets, num_segments=block)  # [block, hidden]
            dW = dW_soft + seg.T
            return dh, dW

        dh0 = jnp.zeros((tokens, hidden), dtype=jnp.float32)
        dh, dW_tiles = jax.lax.scan(
            scan_body,
            dh0,
            jnp.arange(full_blocks, dtype=jnp.int32),
        )

        dW_full = dW_tiles.reshape(full_blocks, hidden, block).transpose(1, 0, 2).reshape(hidden, full_blocks * block)

        if rem:
            start = full_blocks * block
            w_tail = lm_head_kernel[:, start:]
            logits = _dot(h2, w_tail).astype(jnp.float32)  # [tokens, rem]
            probs = jnp.exp(logits - lse2[:, None])

            dlogits_soft = (-probs) * scale[:, None]
            dh = dh + _dot(dlogits_soft, w_tail.T).astype(jnp.float32)
            dW_tail = _dot(h2.T, dlogits_soft).astype(jnp.float32)

            in_range = valid_id & (safe_ids >= start)
            offsets = jnp.where(in_range, safe_ids - start, jnp.zeros((), jnp.int32)).astype(jnp.int32)
            in_range_f = in_range.astype(jnp.float32)

            w_sel = jnp.take(w_tail.T, offsets, axis=0).astype(jnp.float32)
            dh = dh + w_sel * (scale * in_range_f)[:, None]

            updates = (h2.astype(jnp.float32) * (scale * in_range_f)[:, None]).astype(jnp.float32)
            seg = _segment_sum(updates, offsets, num_segments=rem)  # [rem, hidden]
            dW_tail = dW_tail + seg.T
            dW = jnp.concatenate([dW_full, dW_tail], axis=1)
        else:
            dW = dW_full

        dh = dh.reshape(batch, time, hidden).astype(hidden_states.dtype)
        dW = dW.astype(lm_head_kernel.dtype)

        return (dh, dW, None, None, None)

    _kernel.defvjp(fwd, bwd)
    return _kernel


def grpo_per_token_loss_fused_lm_head(
    *,
    hidden_states: Any,
    lm_head_kernel: Any,
    chosen_ids: Any,
    old_per_token_logps: Any,
    advantages: Any,
    cfg: GRPOLmHeadFusedConfig | None = None,
    use_self_old: bool = False,
) -> tuple[Any, Any]:
    if cfg is None:
        cfg = GRPOLmHeadFusedConfig()
    fn = build_grpo_per_token_loss_fused_lm_head(cfg, use_self_old=use_self_old)
    return fn(hidden_states, lm_head_kernel, chosen_ids, old_per_token_logps, advantages)


__all__ = [
    "GRPOLmHeadFusedConfig",
    "grpo_per_token_loss_lm_head_reference",
    "grpo_per_token_loss_fused_lm_head_forward",
    "build_grpo_per_token_loss_fused_lm_head",
    "grpo_per_token_loss_fused_lm_head",
]
