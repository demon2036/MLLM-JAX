from __future__ import annotations

from typing import Any, Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp


class MiniOneRecGrpoModule(nn.Module):
    """GRPO-style policy-gradient loss (+ optional KL penalty to a reference policy).

    This matches the core math used in MiniOneRec's `minionerec_trainer.py`:
    - policy gradient via `exp(logp - stop_gradient(logp)) * advantage`
    - KL term: `exp(ref_logp - logp) - (ref_logp - logp) - 1`

    Note: we rely on the caller-provided `labels` as a completion-token mask
    (1 for completion tokens, 0 for prompt/pad), so the prompt tokens do not
    contribute to the loss.
    """

    model: Any
    ref_model: Any | None = None
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
        completion_mask = labels[:, 1:].astype(jnp.float32)

        log_probs = jax.nn.log_softmax(logits[..., :-1, :], axis=-1)
        per_token_logps = jnp.take_along_axis(log_probs, chosen_ids[..., None], axis=-1)[..., 0]

        advantages = inputs["advantages"]
        if advantages.ndim == 1:
            advantages = advantages[:, None]  # [B, 1]

        # Policy gradient trick (exactly like upstream): forward value is 1, but gradient is d(logp).
        pg_ratio = jnp.exp(per_token_logps - jax.lax.stop_gradient(per_token_logps))
        per_token_pg = pg_ratio * advantages

        per_token_kl = jnp.asarray(0.0, dtype=jnp.float32)
        if float(self.beta) != 0.0 and ref_logits is not None:
            ref_log_probs = jax.nn.log_softmax(ref_logits[..., :-1, :], axis=-1)
            ref_per_token_logps = jnp.take_along_axis(ref_log_probs, chosen_ids[..., None], axis=-1)[..., 0]
            delta = (ref_per_token_logps - per_token_logps).astype(jnp.float32)
            per_token_kl = jnp.exp(delta) - delta - 1.0

        per_token_loss = -(per_token_pg - float(self.beta) * per_token_kl)

        denom = completion_mask.sum(axis=1) + 1e-8
        loss = ((per_token_loss * completion_mask).sum(axis=1) / denom).mean()
        kl = ((per_token_kl * completion_mask).sum(axis=1) / denom).mean()
        completion_length = completion_mask.sum(axis=1).mean()

        out = {
            "loss": loss,
            "kl": kl,
            "completion_length": completion_length,
            "per_token_logps": jax.lax.stop_gradient(per_token_logps),
        }
        return out


__all__ = ["MiniOneRecGrpoModule"]
