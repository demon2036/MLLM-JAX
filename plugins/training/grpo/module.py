from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from plugins.training.kernels.grpo_loss_pallas import GRPOKernelConfig, grpo_per_token_loss_pallas


class TrainGRPOModulePallas(nn.Module):
    """GRPO loss module using a Pallas kernel for per-token loss+logp."""

    model: Any
    pad_token_id: float
    num_pre_Q: int
    ref_model: Any = None
    beta: float = 0.0
    temperature: float = 1.0
    max_lengths: float = 2048
    epsilon_low: float = 0.2
    epsilon_high: float = 0.3
    entropy_threshold: float = 0.3
    kernel_cfg: GRPOKernelConfig = GRPOKernelConfig()
    kernel_interpret: bool = False
    kernel_debug: bool = False

    def __call__(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        logits, _ = self.model(input_ids=input_ids, attention_mask=attention_mask)

        if self.beta != 0 and self.ref_model is not None:
            ref_logits, _ = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
            ref_logits = jax.lax.stop_gradient(ref_logits)
            _ = ref_logits
        elif self.beta != 0 and self.ref_model is None:
            print("Warning: beta is non-zero but ref_model not provided. KL penalty calculation will be skipped.")

        chosen_ids = input_ids[:, 1:]
        mask_loss = labels[:, 1:]

        logits_for_loss = logits[..., :-1, :]

        if "old_per_token_logps" in inputs:
            old_per_token_logps = inputs["old_per_token_logps"]
        else:
            old_per_token_logps = jnp.take_along_axis(
                jax.nn.log_softmax(logits_for_loss, axis=-1),
                chosen_ids[..., None],
                axis=-1,
            )[..., 0] / float(self.temperature)
            old_per_token_logps = jax.lax.stop_gradient(old_per_token_logps)

        per_token_loss, per_token_logps = grpo_per_token_loss_pallas(
            logits=logits_for_loss,
            chosen_ids=chosen_ids,
            old_per_token_logps=old_per_token_logps,
            advantages=inputs["advantages"],
            cfg=GRPOKernelConfig(
                block_size=int(self.kernel_cfg.block_size),
                time_block=int(self.kernel_cfg.time_block),
                epsilon_low=float(self.epsilon_low),
                epsilon_high=float(self.epsilon_high),
                temperature=float(self.temperature),
            ),
            interpret=bool(self.kernel_interpret),
            debug=bool(self.kernel_debug),
        )

        total_valid_token_count = inputs.get("total_valid_token_count", mask_loss.sum())
        loss = (per_token_loss * mask_loss).sum() / total_valid_token_count

        probs = jax.nn.softmax(logits_for_loss / float(self.temperature), axis=-1)
        token_entropy = -jnp.sum(probs * jax.lax.log(probs + 1e-9), axis=-1)

        masked_token_entropy = token_entropy * mask_loss
        sum_entropy_per_sample = masked_token_entropy.sum(axis=-1)
        avg_entropy_per_sample = sum_entropy_per_sample / mask_loss.sum(axis=-1)

        valid_mask_for_metric = mask_loss == 1
        cum_valid = jnp.cumsum(valid_mask_for_metric, axis=-1)
        entropy_calc_mask = jnp.logical_and(valid_mask_for_metric, jnp.logical_and(cum_valid >= 4, cum_valid <= 100))
        masked_token_entropy_truncated = token_entropy * entropy_calc_mask
        sum_entropy_per_sample_truncated = masked_token_entropy_truncated.sum(axis=-1)
        avg_entropy_per_sample_truncated = sum_entropy_per_sample_truncated / entropy_calc_mask.sum(axis=-1)

        return {
            "loss": loss,
            "per_token_logps": jax.lax.stop_gradient(per_token_logps),
            "entropy": avg_entropy_per_sample,
            "entropy_loss": avg_entropy_per_sample_truncated,
        }


__all__ = ["TrainGRPOModulePallas"]
