from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp


class TrainGRPOModuleJax(nn.Module):
    """TrainGRPOModule variant that keeps numerics aligned with the Pallas kernel.

    Key differences vs the legacy upstream module:
    - Compute selective log-softmax in float32 (large-vocab stability + better parity).
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

    log_softmax_block_size: int = 2048

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

        def _pad_vocab(logits_in: jnp.ndarray, *, block_size: int) -> jnp.ndarray:
            vocab = int(logits_in.shape[-1])
            block_size = int(block_size)
            if block_size <= 0:
                raise ValueError("log_softmax_block_size must be > 0")
            pad = (-vocab) % block_size
            if pad == 0:
                return logits_in
            return jnp.pad(
                logits_in,
                ((0, 0), (0, 0), (0, pad)),
                constant_values=jnp.finfo(logits_in.dtype).min,
            )

        def _selective_log_softmax_blockwise(
            *,
            logits_in: jnp.ndarray,
            chosen_ids_in: jnp.ndarray,
            temperature: float,
            block_size: int,
        ) -> jnp.ndarray:
            logits_in = _pad_vocab(logits_in, block_size=block_size)
            block_size = int(block_size)
            batch, time, vocab = (int(logits_in.shape[0]), int(logits_in.shape[1]), int(logits_in.shape[2]))
            blocks = int(vocab // block_size)

            max_ref = jnp.full((batch, time), jnp.finfo(jnp.float32).min, dtype=jnp.float32)
            sum_ref = jnp.zeros((batch, time), dtype=jnp.float32)
            chosen_ref = jnp.zeros((batch, time), dtype=jnp.float32)

            def body(block_idx, carry):
                max_local, sum_local, chosen_local = carry
                start = jnp.asarray(block_idx, dtype=jnp.int32) * jnp.asarray(block_size, dtype=jnp.int32)
                logits_block = jax.lax.dynamic_slice(
                    logits_in,
                    (0, 0, start),
                    (batch, time, block_size),
                ).astype(jnp.float32)

                idx = chosen_ids_in.astype(jnp.int32) - start
                in_range = (idx >= 0) & (idx < block_size)
                idx_clipped = jnp.clip(idx, 0, block_size - 1)
                chosen_val = jnp.take_along_axis(logits_block, idx_clipped[..., None], axis=-1)[..., 0]
                chosen_local = jnp.where(in_range, chosen_val, chosen_local)

                tile_max = jnp.max(logits_block, axis=-1)
                new_max = jnp.maximum(max_local, tile_max)
                sum_local = sum_local * jnp.exp(max_local - new_max) + jnp.sum(
                    jnp.exp(logits_block - new_max[..., None]),
                    axis=-1,
                )
                max_local = new_max
                return max_local, sum_local, chosen_local

            max_ref, sum_ref, chosen_ref = jax.lax.fori_loop(0, blocks, body, (max_ref, sum_ref, chosen_ref))

            lse = max_ref + jnp.log(sum_ref)
            per_token_logps_out = (chosen_ref - lse) / float(temperature)
            return per_token_logps_out.astype(jnp.float32)

        logits_time = logits[:, :-1, :]
        per_token_logps = _selective_log_softmax_blockwise(
            logits_in=logits_time,
            chosen_ids_in=chosen_ids,
            temperature=float(self.temperature),
            block_size=int(self.log_softmax_block_size),
        )

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
