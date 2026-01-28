from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from plugins.training.kernels.grpo_loss_pallas import (
    GRPOKernelConfig,
    grpo_per_token_loss_pallas,
    grpo_per_token_loss_pallas_on_policy,
)


class TrainGRPOModulePallas(nn.Module):
    """TrainGRPOModule variant that uses the Pallas GRPO kernel for policy loss."""

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

    pallas_block_size: int = 2048
    pallas_time_block: int = 8

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

        kernel_cfg = GRPOKernelConfig(
            block_size=int(self.pallas_block_size),
            time_block=int(self.pallas_time_block),
            epsilon_low=float(self.epsilon_low),
            epsilon_high=float(self.epsilon_high),
            temperature=float(self.temperature),
        )
        advantages = inputs["advantages"]

        if "old_per_token_logps" in inputs:
            old_per_token_logps = inputs["old_per_token_logps"]
            per_token_loss, per_token_logps = grpo_per_token_loss_pallas(
                logits=logits[..., :-1, :],
                chosen_ids=chosen_ids,
                old_per_token_logps=old_per_token_logps,
                advantages=advantages,
                cfg=kernel_cfg,
                interpret=False,
                debug=False,
            )
        else:
            per_token_loss, per_token_logps = grpo_per_token_loss_pallas_on_policy(
                logits=logits[..., :-1, :],
                chosen_ids=chosen_ids,
                advantages=advantages,
                cfg=kernel_cfg,
                interpret=False,
                debug=False,
            )

        probs = jax.nn.softmax(logits[..., :-1, :] / self.temperature, axis=-1)
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


__all__ = ["TrainGRPOModulePallas"]

