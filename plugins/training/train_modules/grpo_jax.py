from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp


class TrainGRPOModuleJax(nn.Module):
    """TrainGRPOModule variant that keeps numerics aligned with the Pallas kernel.

    Key differences vs the legacy upstream module:
    - Compute selective log-softmax via float32 logsumexp (large-vocab stability + better parity),
      while avoiding blockwise scans that can inflate TPU compile-time HBM.
    """

    model: Any
    pad_token_id: float
    num_pre_Q: int
    ref_model: Any = None
    beta: float = 0.04
    temperature: float = 1.0
    max_lengths: float = 2048
    epsilon_low: float = 0.2
    epsilon_high: float = 0.3
    entropy_threshold: float = 0.3

    def __call__(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        logits, _ = self.model(input_ids=input_ids, attention_mask=attention_mask)

        if self.beta != 0 and self.ref_model is not None:
            ref_logits, _ = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
            _ = jax.lax.stop_gradient(ref_logits)
        elif self.beta != 0 and self.ref_model is None:
            print("Warning: beta is non-zero but ref_model not provided. KL penalty calculation will be skipped.")

        chosen_ids = input_ids[:, 1:]
        mask_loss = labels[:, 1:]

        logits_time = logits[:, :-1, :]
        logits_time_f32 = logits_time.astype(jnp.float32)
        lse = jax.nn.logsumexp(logits_time_f32, axis=-1)
        chosen_logits = jnp.take_along_axis(
            logits_time_f32,
            chosen_ids.astype(jnp.int32)[..., None],
            axis=-1,
        )[..., 0]
        per_token_logps = (chosen_logits - lse) / float(self.temperature)

        if "old_per_token_logps" in inputs:
            old_per_token_logps = inputs["old_per_token_logps"].astype(jnp.float32)
        else:
            old_per_token_logps = jax.lax.stop_gradient(per_token_logps)

        ratio = jnp.exp(per_token_logps - old_per_token_logps)
        clipped_ratio = jnp.clip(ratio, 1.0 - float(self.epsilon_low), 1.0 + float(self.epsilon_high))

        advantages = inputs["advantages"].astype(jnp.float32)
        loss1 = ratio * advantages[..., None]
        loss2 = clipped_ratio * advantages[..., None]
        per_token_loss = -jnp.minimum(loss1, loss2)

        probs = jax.nn.softmax(logits_time / float(self.temperature), axis=-1)
        token_entropy = -jnp.sum(probs * jax.lax.log(probs + 1e-9), axis=-1)

        masked_token_entropy = token_entropy * mask_loss
        sum_entropy_per_sample = masked_token_entropy.sum(axis=-1)
        avg_entropy_per_sample = sum_entropy_per_sample / mask_loss.sum(axis=-1)

        valid_mask_for_metric = mask_loss == 1
        cum_valid = jnp.cumsum(valid_mask_for_metric, axis=-1)
        entropy_calc_mask = jnp.logical_and(
            valid_mask_for_metric,
            jnp.logical_and(cum_valid >= 4, cum_valid <= 100),
        )
        masked_token_entropy_truncated = token_entropy * entropy_calc_mask
        sum_entropy_per_sample_truncated = masked_token_entropy_truncated.sum(axis=-1)
        avg_entropy_per_sample_truncated = sum_entropy_per_sample_truncated / entropy_calc_mask.sum(axis=-1)

        total_valid_token_count = inputs.get("total_valid_token_count", mask_loss.sum())
        loss = (per_token_loss * mask_loss).sum() / total_valid_token_count

        return {
            "loss": loss,
            "per_token_logps": jax.lax.stop_gradient(per_token_logps),
            "entropy": avg_entropy_per_sample,
            "entropy_loss": avg_entropy_per_sample_truncated,
        }


__all__ = ["TrainGRPOModuleJax"]
