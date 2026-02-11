from __future__ import annotations

from typing import Any, Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp

from MLLM_JAX.language.qwen2.modular_qwen2 import Qwen2Model
from plugins.training.rl.token_focus import build_token_focus_mask


class PPOActorCriticModule(nn.Module):
    """Actor-critic PPO module with a value head (Qwen2 backbone)."""

    config: Any
    jax_config: Any
    temperature: float = 1.0
    epsilon_low: float = 0.2
    epsilon_high: float = 0.2
    value_coef: float = 0.5
    value_clip_range: float | None = 0.2
    entropy_coef: float = 0.0
    gradient_checkpointing: bool = True
    token_focus_enabled: bool = False
    token_focus_prob_threshold: float = 0.3
    token_focus_max_tokens_per_sequence: int = 10
    token_focus_use_old_logps: bool = True

    def setup(self) -> None:
        dtype = self.jax_config.dtype
        param_dtype = self.jax_config.param_dtype
        if bool(self.gradient_checkpointing):
            self.model = nn.remat(Qwen2Model)(self.config, jax_config=self.jax_config)
            self.lm_head = nn.remat(nn.Dense)(
                self.config.vocab_size,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
            )
        else:
            self.model = Qwen2Model(self.config, jax_config=self.jax_config)
            self.lm_head = nn.Dense(
                self.config.vocab_size,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
            )
        self.value_head = nn.Dense(
            1,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
        )

    def _forward(self, input_ids, attention_mask):
        _, seq_len = input_ids.shape
        position_ids = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
        hidden_states, cache = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=None,
            cache=None,
        )
        logits = self.lm_head(hidden_states)
        values = self.value_head(hidden_states).squeeze(-1)
        return logits, values, cache

    def value(self, input_ids, attention_mask):
        _, values, _ = self._forward(input_ids, attention_mask)
        return values

    def __call__(self, inputs: Mapping[str, Any]):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        logits, values, _ = self._forward(input_ids, attention_mask)

        chosen_ids = input_ids[:, 1:]
        mask_loss = labels[:, 1:].astype(jnp.float32)

        per_token_logps = jnp.take_along_axis(
            jax.nn.log_softmax(logits[..., :-1, :], axis=-1),
            chosen_ids[..., None],
            axis=-1,
        )[..., 0] / float(self.temperature)

        probs = jax.nn.softmax(logits[..., :-1, :] / float(self.temperature), axis=-1)
        token_entropy = -jnp.sum(probs * jax.lax.log(probs + 1e-9), axis=-1)
        entropy = (token_entropy * mask_loss).sum() / (mask_loss.sum() + 1e-8)

        if "old_per_token_logps" in inputs:
            old_per_token_logps = inputs["old_per_token_logps"]
        else:
            old_per_token_logps = jax.lax.stop_gradient(per_token_logps)

        ratio = jnp.exp(per_token_logps - old_per_token_logps)
        clipped_ratio = jnp.clip(ratio, 1.0 - float(self.epsilon_low), 1.0 + float(self.epsilon_high))

        advantages = inputs["advantages"]
        if advantages.ndim == 1:
            advantages = advantages[:, None]
        if advantages.shape != per_token_logps.shape:
            if advantages.shape[1] == 1:
                advantages = jnp.broadcast_to(advantages, per_token_logps.shape)
            else:
                raise ValueError(f"advantages shape {advantages.shape} does not match per_token_logps {per_token_logps.shape}")

        per_token_loss1 = ratio * advantages
        per_token_loss2 = clipped_ratio * advantages
        per_token_ppo_loss = jnp.minimum(per_token_loss1, per_token_loss2)
        per_token_loss = -per_token_ppo_loss

        total_valid_token_count = inputs.get("total_valid_token_count", mask_loss.sum())

        focus_logps = old_per_token_logps if bool(self.token_focus_use_old_logps) else per_token_logps
        focus_logps = jax.lax.stop_gradient(focus_logps)

        focus_mask, selected_counts = (
            build_token_focus_mask(
                focus_logps,
                mask_loss,
                prob_threshold=float(self.token_focus_prob_threshold),
                max_tokens_per_sequence=int(self.token_focus_max_tokens_per_sequence),
            )
            if bool(self.token_focus_enabled)
            else (jnp.ones_like(mask_loss, dtype=jnp.float32), mask_loss.sum(axis=-1).astype(jnp.int32))
        )
        selected_counts_f = selected_counts.astype(jnp.float32)

        eligible_counts = jnp.zeros_like(selected_counts, dtype=jnp.int32)
        if bool(self.token_focus_enabled):
            logp_threshold = jnp.log(jnp.asarray(float(self.token_focus_prob_threshold), dtype=focus_logps.dtype))
            eligible = jnp.logical_and(mask_loss > 0, focus_logps < logp_threshold)
            eligible_counts = eligible.astype(jnp.int32).sum(axis=-1)
        eligible_counts_f = eligible_counts.astype(jnp.float32)
        if bool(self.token_focus_enabled):
            eligible_fraction = eligible_counts_f.sum() / (mask_loss.sum() + 1e-8)
        else:
            eligible_fraction = jnp.asarray(0.0, dtype=jnp.float32)

        mask_policy = mask_loss * focus_mask
        if bool(self.token_focus_enabled):
            policy_denom = jnp.maximum(mask_policy.sum(), 1.0)
        else:
            policy_denom = total_valid_token_count

        policy_loss = (per_token_loss * mask_policy).sum() / policy_denom

        value_pred = values[:, :-1]
        returns = inputs.get("returns")
        if returns is None:
            returns = advantages + value_pred
        if returns.ndim == 1:
            returns = returns[:, None]
        if returns.shape != value_pred.shape:
            if returns.shape[1] == 1:
                returns = jnp.broadcast_to(returns, value_pred.shape)
            else:
                raise ValueError(f"returns shape {returns.shape} does not match value_pred {value_pred.shape}")

        old_values = inputs.get("old_values", inputs.get("values"))
        if old_values is None:
            old_values = jax.lax.stop_gradient(value_pred)
        if old_values.ndim == 1:
            old_values = old_values[:, None]
        if old_values.shape != value_pred.shape:
            if old_values.shape[1] == 1:
                old_values = jnp.broadcast_to(old_values, value_pred.shape)
            else:
                raise ValueError(f"old_values shape {old_values.shape} does not match value_pred {value_pred.shape}")

        value_pred = value_pred.astype(jnp.float32)
        returns = returns.astype(jnp.float32)
        old_values = old_values.astype(jnp.float32)

        if self.value_clip_range is not None:
            value_clip = float(self.value_clip_range)
            value_clipped = old_values + jnp.clip(value_pred - old_values, -value_clip, value_clip)
            value_loss = jnp.maximum((value_pred - returns) ** 2, (value_clipped - returns) ** 2)
        else:
            value_loss = (value_pred - returns) ** 2

        value_loss = (value_loss * mask_loss).sum() / total_valid_token_count

        loss = policy_loss + float(self.value_coef) * value_loss - float(self.entropy_coef) * entropy

        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "value_pred_mean": jnp.mean(value_pred),
            "return_mean": jnp.mean(returns),
            "per_token_logps": jax.lax.stop_gradient(per_token_logps),
            "token_focus/enabled": jnp.asarray(1.0 if bool(self.token_focus_enabled) else 0.0, dtype=jnp.float32),
            "token_focus/prob_threshold": jnp.asarray(float(self.token_focus_prob_threshold), dtype=jnp.float32),
            "token_focus/max_tokens_per_sequence": jnp.asarray(
                float(self.token_focus_max_tokens_per_sequence), dtype=jnp.float32
            ),
            "token_focus/use_old_logps": jnp.asarray(1.0 if bool(self.token_focus_use_old_logps) else 0.0, dtype=jnp.float32),
            "token_focus/eligible_fraction": eligible_fraction,
            "token_focus/eligible_tokens_mean": eligible_counts_f.mean(),
            "token_focus/eligible_tokens_min": eligible_counts_f.min(),
            "token_focus/eligible_tokens_max": eligible_counts_f.max(),
            "token_focus/selected_tokens_mean": selected_counts_f.mean(),
            "token_focus/selected_tokens_min": selected_counts_f.min(),
            "token_focus/selected_tokens_max": selected_counts_f.max(),
            "token_focus/empty_seq_fraction": (selected_counts == 0).astype(jnp.float32).mean(),
            "token_focus/selected_fraction": selected_counts_f.sum() / (mask_loss.sum() + 1e-8),
        }


__all__ = ["PPOActorCriticModule"]
