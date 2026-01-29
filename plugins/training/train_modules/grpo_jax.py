from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp


class TrainGRPOModuleJax(nn.Module):
    """TrainGRPOModule variant that keeps numerics aligned with the Pallas kernel.

    Key differences vs the legacy upstream module:
    - Compute selective log-softmax via a float32, vocab-blocked logsumexp update.
      This matches the Pallas kernel's reduction order more closely than a single
      `jax.nn.logsumexp(..., axis=-1)` over the full vocab, improving parity for
      RL runs that are sensitive to tiny numerical drift.
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
    vocab_block_size: int = 2048

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
        block_size = int(self.vocab_block_size)
        if block_size <= 0:
            raise ValueError("vocab_block_size must be > 0")

        vocab = int(logits_time.shape[-1])
        full_blocks = int(vocab // block_size)
        remainder = int(vocab % block_size)

        max_acc = jnp.full(logits_time.shape[:2], jnp.finfo(jnp.float32).min, dtype=jnp.float32)
        sum_acc = jnp.zeros(logits_time.shape[:2], dtype=jnp.float32)

        def _update_block(max_sum, logits_block):
            block_f32 = logits_block.astype(jnp.float32)
            block_max = jnp.max(block_f32, axis=-1)
            max_prev, sum_prev = max_sum
            max_new = jnp.maximum(max_prev, block_max)
            sum_prev = sum_prev * jnp.exp(max_prev - max_new)
            block_sum = jnp.sum(jnp.exp(block_f32 - max_new[..., None]), axis=-1)
            sum_new = sum_prev + block_sum
            return max_new, sum_new

        def _scan_body(i, max_sum):
            start = i * block_size
            logits_block = jax.lax.dynamic_slice_in_dim(logits_time, start, block_size, axis=-1)
            return _update_block(max_sum, logits_block)

        if full_blocks > 0:
            max_acc, sum_acc = jax.lax.fori_loop(0, full_blocks, _scan_body, (max_acc, sum_acc))
        if remainder > 0:
            start = full_blocks * block_size
            logits_block = jax.lax.dynamic_slice_in_dim(logits_time, start, remainder, axis=-1)
            max_acc, sum_acc = _update_block((max_acc, sum_acc), logits_block)

        lse = max_acc + jnp.log(sum_acc)
        chosen_logits = jnp.take_along_axis(
            logits_time,
            chosen_ids.astype(jnp.int32)[..., None],
            axis=-1,
        )[..., 0].astype(jnp.float32)
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
