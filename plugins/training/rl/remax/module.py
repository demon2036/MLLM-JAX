from __future__ import annotations

from typing import Any, Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp


def _as_float32(x: Any) -> jnp.ndarray:
    return jnp.asarray(x, dtype=jnp.float32)


def _reverse_cumsum(x: jnp.ndarray) -> jnp.ndarray:
    if x.ndim != 2:
        raise ValueError(f"reverse_cumsum expects rank-2 [B, T], got shape={x.shape}")
    return jnp.cumsum(x[:, ::-1], axis=1)[:, ::-1]


def _last_token_mask(completion_mask: jnp.ndarray) -> jnp.ndarray:
    if completion_mask.ndim != 2:
        raise ValueError(f"completion_mask must be rank-2 [B, T], got shape={completion_mask.shape}")

    mask = _as_float32(completion_mask)
    bsz, seq_len = mask.shape
    del bsz

    positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    last_idx = jnp.max(jnp.where(mask > 0, positions, -jnp.ones_like(positions)), axis=1)

    # For rows with an empty completion (e.g. baseline rows), last_idx=-1 and one_hot is all zeros.
    return jax.nn.one_hot(last_idx, seq_len, dtype=jnp.float32)


def _discounted_terminal_returns(
    *,
    terminal_reward: jnp.ndarray,
    completion_mask: jnp.ndarray,
    gamma: float,
) -> jnp.ndarray:
    """Compute per-token discounted terminal returns (no accumulation of per-step rewards).

    Matches the official ReMax return shape where each completion token gets the
    terminal scalar reward discounted by `gamma**k`, where k is the number of
    remaining completion tokens including the current one.

    - terminal_reward: [B] scalar terminal rewards (e.g. advantage)
    - completion_mask: [B, T] 1 for completion tokens, 0 otherwise

    Returns [B, T]. Tokens with mask==0 get 0.
    """
    if completion_mask.ndim != 2:
        raise ValueError(f"completion_mask must be rank-2 [B, T], got shape={completion_mask.shape}")

    mask = _as_float32(completion_mask)
    remaining = jnp.cumsum(mask[:, ::-1], axis=1)[:, ::-1]
    gamma_f = float(gamma)
    if gamma_f <= 0:
        raise ValueError("gamma must be > 0")

    discounted = _as_float32(terminal_reward)[:, None] * jnp.power(gamma_f, remaining)
    return discounted * mask


class ReMaxPolicyGradientModule(nn.Module):
    """ReMax policy-gradient loss with greedy baseline advantages and KL shaping.

    Inputs (batch keys)
    -------------------
    - input_ids: int32 [B, L]
    - attention_mask: int32 [B, L]
    - labels: int32 [B, L] completion mask (1 for completion tokens)
    - advantages: float32 [B] (already baseline-subtracted)
    - total_valid_token_count: optional float scalar

    Notes
    -----
    - Baseline rows should have labels==0, so they contribute no gradients.
    - `returns_style` selects how we spread the scalar outcome reward across tokens:

      - "official": discounted terminal reward + token-local KL shaping
        (matches DeepSpeed-Chat style reference).

      - "verl": construct token-level rewards as:
          token_level_scores = advantage placed on the last completion token
          token_level_rewards = token_level_scores - beta * KL(token)
        then compute per-token returns via reverse-cumsum.
    """

    model: Any
    pad_token_id: int
    ref_model: Any | None = None
    kl_coef: float = 0.0
    gamma: float = 1.0
    returns_style: str = "official"

    def __call__(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        advantages = _as_float32(inputs["advantages"]).reshape((-1,))
        advantages = jax.lax.stop_gradient(advantages)

        logits, _ = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        ref_logits = None
        kl_coef = float(self.kl_coef)
        if kl_coef != 0.0:
            if self.ref_model is None:
                raise ValueError("kl_coef != 0 requires ref_model")
            ref_logits, _ = self.ref_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            ref_logits = jax.lax.stop_gradient(ref_logits)

        chosen_ids = input_ids[:, 1:]
        completion_mask = _as_float32(labels[:, 1:])

        log_probs = jax.nn.log_softmax(_as_float32(logits[:, :-1, :]), axis=-1)
        per_token_logps = jnp.take_along_axis(
            log_probs,
            chosen_ids[..., None],
            axis=-1,
        )[..., 0]

        probs = jnp.exp(log_probs)
        token_entropy = -jnp.sum(probs * log_probs, axis=-1)

        kl_log_ratio = jnp.zeros_like(per_token_logps, dtype=jnp.float32)
        kl_shaping = jnp.zeros_like(per_token_logps, dtype=jnp.float32)
        if ref_logits is not None:
            ref_per_token_logps = jnp.take_along_axis(
                jax.nn.log_softmax(_as_float32(ref_logits[:, :-1, :]), axis=-1),
                chosen_ids[..., None],
                axis=-1,
            )[..., 0]
            kl_log_ratio = jax.lax.stop_gradient(per_token_logps - ref_per_token_logps)
            kl_shaping = -kl_coef * kl_log_ratio

        style_raw = str(self.returns_style or "").strip().lower()
        style_aliases = {
            "": "official",
            "official": "official",
            "paper": "official",
            "remax": "official",
            "discounted_terminal": "official",
            "terminal_discount": "official",
            "verl": "verl",
            "reverse_cumsum": "verl",
            "cumsum": "verl",
        }
        returns_style = style_aliases.get(style_raw, style_raw)

        if returns_style == "official":
            terminal_returns = _discounted_terminal_returns(
                terminal_reward=advantages,
                completion_mask=completion_mask,
                gamma=float(self.gamma),
            )
            returns = terminal_returns + kl_shaping * completion_mask
        elif returns_style == "verl":
            last_token = _last_token_mask(completion_mask)
            token_level_scores = advantages[:, None] * last_token
            token_level_rewards = (token_level_scores + kl_shaping) * completion_mask
            returns = _reverse_cumsum(token_level_rewards)
        else:
            raise ValueError(f"Unsupported returns_style={self.returns_style!r}; expected 'official' or 'verl'.")

        returns = jax.lax.stop_gradient(returns)

        total_valid_token_count = _as_float32(inputs.get("total_valid_token_count", completion_mask.sum()))
        total_valid_token_count = jnp.maximum(total_valid_token_count, 1.0)

        policy_loss = -jnp.sum(returns * per_token_logps * completion_mask) / total_valid_token_count

        entropy = jnp.sum(token_entropy * completion_mask) / total_valid_token_count

        return_mean = jnp.sum(returns * completion_mask) / total_valid_token_count
        kl_mean = jnp.sum(kl_log_ratio * completion_mask) / total_valid_token_count
        baseline_fraction = _as_float32(inputs.get("is_baseline", jnp.zeros((advantages.shape[0],), dtype=jnp.int32))).mean()
        returns_style_id = jnp.asarray(1.0 if returns_style == "verl" else 0.0, dtype=jnp.float32)

        return {
            "loss": policy_loss,
            "policy_loss": policy_loss,
            "entropy": entropy,
            "return_mean": return_mean,
            "adv_mean": jnp.mean(advantages),
            "kl_log_ratio_mean": kl_mean,
            "kl_coef": jnp.asarray(kl_coef, dtype=jnp.float32),
            "gamma": jnp.asarray(float(self.gamma), dtype=jnp.float32),
            "returns_style_id": returns_style_id,
            "baseline_fraction": baseline_fraction,
            "per_token_logps": jax.lax.stop_gradient(per_token_logps),
        }


__all__ = ["ReMaxPolicyGradientModule"]
