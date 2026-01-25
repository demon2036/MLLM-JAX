from __future__ import annotations

from typing import Any, Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp


class MiniOneRecGrpoModule(nn.Module):
    """GRPO-style PPO loss (+ optional KL penalty to a frozen reference policy)."""

    model: Any
    ref_model: Any | None = None
    temperature: float = 1.0
    epsilon_low: float = 0.2
    epsilon_high: float = 0.3
    beta: float = 0.0

    def __call__(self, inputs: Mapping[str, Any]):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        logits, _cache = self.model(input_ids=input_ids, attention_mask=attention_mask)

        ref_logits = None
        if float(self.beta) != 0.0:
            if self.ref_model is None:
                raise ValueError("beta != 0 requires ref_model")
            ref_logits, _ref_cache = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
            ref_logits = jax.lax.stop_gradient(ref_logits)

        chosen_ids = input_ids[:, 1:]
        mask_loss = labels[:, 1:].astype(jnp.float32)

        log_probs = jax.nn.log_softmax(logits[..., :-1, :], axis=-1)
        per_token_logps = jnp.take_along_axis(log_probs, chosen_ids[..., None], axis=-1)[..., 0] / float(self.temperature)

        entropy = None
        try:
            probs = jax.nn.softmax(logits[..., :-1, :] / float(self.temperature), axis=-1)
            token_entropy = -jnp.sum(probs * jax.lax.log(probs + 1e-9), axis=-1)
            entropy = (token_entropy * mask_loss).sum() / (mask_loss.sum() + 1e-8)
        except Exception:
            entropy = None

        old_per_token_logps = inputs.get("old_per_token_logps")
        if old_per_token_logps is None:
            old_per_token_logps = jax.lax.stop_gradient(per_token_logps)

        ratio = jnp.exp(per_token_logps - old_per_token_logps)
        clipped_ratio = jnp.clip(ratio, 1.0 - float(self.epsilon_low), 1.0 + float(self.epsilon_high))

        advantages = inputs["advantages"]
        if advantages.ndim == 1:
            advantages = advantages[:, None]
        if advantages.shape[1] == 1:
            advantages = jnp.broadcast_to(advantages, per_token_logps.shape)
        if advantages.shape != per_token_logps.shape:
            raise ValueError(f"advantages shape {advantages.shape} does not match per_token_logps {per_token_logps.shape}")

        per_token_loss1 = ratio * advantages
        per_token_loss2 = clipped_ratio * advantages
        per_token_ppo = jnp.minimum(per_token_loss1, per_token_loss2)
        per_token_loss = -per_token_ppo

        total_valid_token_count = inputs.get("total_valid_token_count", mask_loss.sum())
        policy_loss = (per_token_loss * mask_loss).sum() / (total_valid_token_count + 1e-8)

        kl_loss = jnp.asarray(0.0, dtype=jnp.float32)
        if float(self.beta) != 0.0 and ref_logits is not None:
            ref_log_probs = jax.nn.log_softmax(ref_logits[..., :-1, :], axis=-1)
            ref_per_token_logps = jnp.take_along_axis(ref_log_probs, chosen_ids[..., None], axis=-1)[..., 0]
            per_token_kl = (per_token_logps - ref_per_token_logps).astype(jnp.float32)
            kl_loss = (per_token_kl * mask_loss).sum() / (total_valid_token_count + 1e-8)

        loss = policy_loss + float(self.beta) * kl_loss

        out = {
            "loss": loss,
            "policy_loss": policy_loss,
            "kl_loss": kl_loss,
            "per_token_logps": jax.lax.stop_gradient(per_token_logps),
        }
        if entropy is not None:
            out["entropy"] = entropy
        return out


__all__ = ["MiniOneRecGrpoModule"]

