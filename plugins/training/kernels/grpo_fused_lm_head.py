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


def _segment_sum(updates: Any, segment_ids: Any, *, num_segments: int) -> Any:
    import jax.numpy as jnp

    num_segments = int(num_segments)
    if num_segments <= 0:
        raise ValueError("num_segments must be > 0")
    out = jnp.zeros((num_segments,) + updates.shape[1:], dtype=updates.dtype)
    return out.at[segment_ids].add(updates)


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

    ids = chosen_ids.astype(jnp.int32)
    valid_id = (ids >= 0) & (ids < vocab)
    safe_ids = jnp.where(valid_id, ids, jnp.zeros((), jnp.int32))

    full_blocks = int(vocab // block)
    rem = int(vocab % block)

    def _dot_bt(x, w):
        return jax.lax.dot_general(
            x,
            w,
            (((x.ndim - 1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )

    neg_inf = jnp.asarray(-jnp.inf, dtype=jnp.float32)

    def _logits_f32(x, w):
        # Match the repo reference rounding: compute dot in f32 then cast to
        # the hidden dtype (typically bf16) before any softmax stats.
        logits = _dot_bt(x, w).astype(x.dtype)
        return logits.astype(jnp.float32)

    def scan_stats(carry, block_id):
        m, sumexp, chosen = carry
        start = block_id * block

        w_blk = jax.lax.dynamic_slice(lm_head_kernel, (0, start), (hidden, block))
        logits = _logits_f32(hidden_states, w_blk)

        tile_max = jnp.max(logits, axis=-1)
        new_m = jnp.maximum(m, tile_max)
        sumexp = sumexp * jnp.exp(m - new_m) + jnp.sum(jnp.exp(logits - new_m[..., None]), axis=-1)
        m = new_m

        in_range = valid_id & (safe_ids >= start) & (safe_ids < start + block)
        offsets = jnp.where(in_range, safe_ids - start, jnp.zeros((), jnp.int32)).astype(jnp.int32)
        chosen_val = jnp.take_along_axis(logits, offsets[..., None], axis=-1)[..., 0]
        chosen = jnp.where(in_range, chosen_val, chosen)
        return (m, sumexp, chosen), None

    init_m = jnp.full((batch, time), neg_inf, dtype=jnp.float32)
    init_sumexp = jnp.zeros((batch, time), dtype=jnp.float32)
    init_chosen = jnp.zeros((batch, time), dtype=jnp.float32)
    (m, sumexp, chosen), _ = jax.lax.scan(
        scan_stats,
        (init_m, init_sumexp, init_chosen),
        jnp.arange(full_blocks, dtype=jnp.int32),
    )

    if rem:
        start = full_blocks * block
        w_tail = lm_head_kernel[:, start:]
        logits = _logits_f32(hidden_states, w_tail)

        tile_max = jnp.max(logits, axis=-1)
        new_m = jnp.maximum(m, tile_max)
        sumexp = sumexp * jnp.exp(m - new_m) + jnp.sum(jnp.exp(logits - new_m[..., None]), axis=-1)
        m = new_m

        in_range = valid_id & (safe_ids >= start)
        offsets = jnp.where(in_range, safe_ids - start, jnp.zeros((), jnp.int32)).astype(jnp.int32)
        chosen_val = jnp.take_along_axis(logits, offsets[..., None], axis=-1)[..., 0]
        chosen = jnp.where(in_range, chosen_val, chosen)

    lse = m + jnp.log(sumexp)
    logp = chosen - lse
    return logp.astype(jnp.float32), lse.astype(jnp.float32)


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
        ids = chosen_ids.astype(jnp.int32)
        valid_id = (ids >= 0) & (ids < vocab)
        safe_ids = jnp.where(valid_id, ids, jnp.zeros((), jnp.int32))

        lse_f32 = lse.astype(jnp.float32)
        lse2 = lse_f32.reshape(tokens)
        logp2 = per_logps.astype(jnp.float32).reshape(tokens)
        dloss2 = dloss.astype(jnp.float32).reshape(tokens)

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
        scale_valid = scale * valid_id.reshape(tokens).astype(jnp.float32)

        full_blocks = int(vocab // block)
        rem = int(vocab % block)

        def scan_body(carry, block_id):
            dh_carry, dW_carry = carry
            start = block_id * block
            w_blk = jax.lax.dynamic_slice(lm_head_kernel, (0, start), (hidden, block))
            logits = jax.lax.dot_general(
                hidden_states,
                w_blk,
                (((2,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            ).astype(hidden_states.dtype).astype(jnp.float32)
            probs = jnp.exp(logits - lse_f32[..., None])

            probs2 = probs.reshape(tokens, block)
            dlogits2 = probs2 * (-scale_valid[:, None])

            in_range = valid_id & (safe_ids >= start) & (safe_ids < start + block)
            offsets = jnp.where(in_range, safe_ids - start, jnp.zeros((), jnp.int32)).astype(jnp.int32).reshape(tokens)
            dlogits2 = dlogits2.at[jnp.arange(tokens), offsets].add(scale_valid * in_range.reshape(tokens).astype(jnp.float32))

            dlogits = dlogits2.reshape(batch, time, block)
            dh_update = jax.lax.dot_general(
                dlogits,
                w_blk,
                (((2,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            ).astype(dh_carry.dtype)
            dh = dh_carry + dh_update

            dW_tile = jax.lax.dot_general(
                h2,
                dlogits2.astype(lm_head_kernel.dtype),
                (((0,), (0,)), ((), ())),
                preferred_element_type=lm_head_kernel.dtype,
            )
            dW_carry = jax.lax.dynamic_update_slice(dW_carry, dW_tile.astype(lm_head_kernel.dtype), (0, start))
            return (dh, dW_carry), None

        dh0 = jnp.zeros((batch, time, hidden), dtype=hidden_states.dtype)
        dW0 = jnp.zeros((hidden, vocab), dtype=lm_head_kernel.dtype)
        (dh, dW), _ = jax.lax.scan(
            scan_body,
            (dh0, dW0),
            jnp.arange(full_blocks, dtype=jnp.int32),
        )

        if rem:
            start = full_blocks * block
            w_tail = lm_head_kernel[:, start:]
            logits = jax.lax.dot_general(
                hidden_states,
                w_tail,
                (((2,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            ).astype(hidden_states.dtype).astype(jnp.float32)
            probs = jnp.exp(logits - lse_f32[..., None])

            probs2 = probs.reshape(tokens, rem)
            dlogits2 = probs2 * (-scale_valid[:, None])

            in_range = valid_id & (safe_ids >= start)
            offsets = jnp.where(in_range, safe_ids - start, jnp.zeros((), jnp.int32)).astype(jnp.int32).reshape(tokens)
            dlogits2 = dlogits2.at[jnp.arange(tokens), offsets].add(scale_valid * in_range.reshape(tokens).astype(jnp.float32))

            dlogits = dlogits2.reshape(batch, time, rem)
            dh_update = jax.lax.dot_general(
                dlogits,
                w_tail,
                (((2,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            ).astype(dh.dtype)
            dh = dh + dh_update

            dW_tail = jax.lax.dot_general(
                h2,
                dlogits2.astype(lm_head_kernel.dtype),
                (((0,), (0,)), ((), ())),
                preferred_element_type=lm_head_kernel.dtype,
            )
            dW = jax.lax.dynamic_update_slice(dW, dW_tail.astype(lm_head_kernel.dtype), (0, start))

        dh = dh.astype(hidden_states.dtype)
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
