from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp


def _masked_mean(values: jax.Array, mask: jax.Array) -> jax.Array:
    mask_f = mask.astype(jnp.float32)
    denom = jnp.maximum(mask_f.sum(), 1.0)
    return (values.astype(jnp.float32) * mask_f).sum() / denom


class TrainSftModule(nn.Module):
    """Causal-LM supervised fine-tuning loss for SID SFT."""

    model: Any
    label_ignore_id: int = -100

    def __call__(self, inputs: dict[str, jax.Array]) -> dict[str, jax.Array]:
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        logits, _cache = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Shift for next-token prediction (matches HF causal LM loss).
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        shift_attention = attention_mask[:, 1:]

        valid = jnp.logical_and(shift_labels != int(self.label_ignore_id), shift_attention.astype(bool))
        safe_labels = jnp.where(valid, shift_labels, 0).astype(jnp.int32)

        log_probs = jax.nn.log_softmax(shift_logits.astype(jnp.float32), axis=-1)
        chosen = jnp.take_along_axis(log_probs, safe_labels[..., None], axis=-1)[..., 0]
        per_token_loss = -chosen

        loss = _masked_mean(per_token_loss, valid)
        token_count = valid.astype(jnp.int32).sum()
        return {
            "loss": loss,
            "token_count": token_count,
        }


__all__ = ["TrainSftModule"]

