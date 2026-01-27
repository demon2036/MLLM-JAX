from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from plugins.training.kernels.entropy_pallas import (
    EntropyKernelConfig,
    entropy_per_token_pallas,
    entropy_per_token_pallas_sharded,
)
from plugins.training.kernels.grpo_loss_pallas import (
    GRPOKernelConfig,
    GRPOKernelShardingSpec,
    grpo_per_token_loss_pallas,
    grpo_per_token_loss_pallas_sharded,
)
from plugins.training.kernels.grpo_fused_lm_head import (
    GRPOLmHeadFusedConfig,
    grpo_per_token_loss_fused_lm_head,
)


class TrainGRPOModulePallas(nn.Module):
    """GRPO loss module using a Pallas kernel for per-token loss+logp."""

    model: Any
    pad_token_id: float
    num_pre_Q: int
    mesh: Any | None = None
    ref_model: Any = None
    beta: float = 0.0
    temperature: float = 1.0
    max_lengths: float = 2048
    epsilon_low: float = 0.2
    epsilon_high: float = 0.3
    entropy_threshold: float = 0.3
    kernel_cfg: GRPOKernelConfig = GRPOKernelConfig()
    kernel_sharding: GRPOKernelShardingSpec = GRPOKernelShardingSpec()
    kernel_interpret: bool = False
    kernel_debug: bool = False
    kernel_check_vma: bool = False

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

        # We compute the GRPO loss on the usual `[B, T-1]` positions (aligned
        # with `input_ids[:, 1:]`), but we feed the kernel a `[B, T]` logits
        # tensor so `T` can be chosen to align with kernel tiling (e.g. `T % 8 == 0`)
        # without requiring a full-logits padding copy inside the kernel.
        logits_for_loss = logits[..., :-1, :]
        logits_for_kernel = logits
        chosen_ids_for_kernel = jnp.pad(chosen_ids, ((0, 0), (0, 1)), constant_values=0)

        use_self_old = "old_per_token_logps" not in inputs
        if use_self_old:
            old_per_token_logps = jnp.zeros_like(mask_loss, dtype=jnp.float32)
        else:
            old_per_token_logps = inputs["old_per_token_logps"].astype(jnp.float32)
        old_per_token_logps_for_kernel = jnp.pad(old_per_token_logps, ((0, 0), (0, 1)), constant_values=0.0)

        kernel_cfg = GRPOKernelConfig(
            block_size=int(self.kernel_cfg.block_size),
            time_block=int(self.kernel_cfg.time_block),
            epsilon_low=float(self.epsilon_low),
            epsilon_high=float(self.epsilon_high),
            temperature=float(self.temperature),
            bwd_impl=str(getattr(self.kernel_cfg, "bwd_impl", "pallas") or "pallas"),
        )

        if self.mesh is None:
            per_token_loss, per_token_logps = grpo_per_token_loss_pallas(
                logits=logits_for_kernel,
                chosen_ids=chosen_ids_for_kernel,
                old_per_token_logps=old_per_token_logps_for_kernel,
                advantages=inputs["advantages"],
                cfg=kernel_cfg,
                interpret=bool(self.kernel_interpret),
                debug=bool(self.kernel_debug),
                use_self_old=bool(use_self_old),
            )
        else:
            per_token_loss, per_token_logps = grpo_per_token_loss_pallas_sharded(
                logits=logits_for_kernel,
                chosen_ids=chosen_ids_for_kernel,
                old_per_token_logps=old_per_token_logps_for_kernel,
                advantages=inputs["advantages"],
                mesh=self.mesh,
                cfg=kernel_cfg,
                sharding=self.kernel_sharding,
                interpret=bool(self.kernel_interpret),
                debug=bool(self.kernel_debug),
                check_vma=bool(self.kernel_check_vma),
                use_self_old=bool(use_self_old),
            )

        per_token_loss = per_token_loss[:, : mask_loss.shape[1]]
        per_token_logps = per_token_logps[:, : mask_loss.shape[1]]

        total_valid_token_count = inputs.get("total_valid_token_count", mask_loss.sum())
        loss = (per_token_loss * mask_loss).sum() / total_valid_token_count

        entropy_cfg = EntropyKernelConfig(
            block_size=int(self.kernel_cfg.block_size),
            time_block=int(self.kernel_cfg.time_block),
            temperature=float(self.temperature),
        )
        if self.mesh is None:
            token_entropy = entropy_per_token_pallas(
                logits=logits_for_loss,
                cfg=entropy_cfg,
                interpret=bool(self.kernel_interpret),
                debug=bool(self.kernel_debug),
            )
        else:
            token_entropy = entropy_per_token_pallas_sharded(
                logits=logits_for_loss,
                mesh=self.mesh,
                cfg=entropy_cfg,
                batch_axes=tuple(self.kernel_sharding.batch_axes),
                vocab_axis=self.kernel_sharding.vocab_axis,
                interpret=bool(self.kernel_interpret),
                debug=bool(self.kernel_debug),
            )
        token_entropy = jax.lax.stop_gradient(token_entropy)

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

class TrainGRPOModuleFusedLmHead(nn.Module):
    """GRPO loss module fused with the LM head projection (no logits tensor)."""

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
    fused_cfg: GRPOLmHeadFusedConfig = GRPOLmHeadFusedConfig()

    def __call__(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        seq_len = int(input_ids.shape[1])
        position_ids = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
        hidden_states, _ = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=None,
            cache=None,
        )

        if self.beta != 0 and self.ref_model is not None:
            # Keep the ref forward for API parity; KL is not used in this repo's loss.
            _ref_hidden, _ = self.ref_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=None,
                cache=None,
            )
            _ = jax.lax.stop_gradient(_ref_hidden)
        elif self.beta != 0 and self.ref_model is None:
            print("Warning: beta is non-zero but ref_model not provided. KL penalty calculation will be skipped.")

        # Align with `[B, T-1]` loss positions.
        hidden_for_loss = hidden_states[:, :-1, :]
        chosen_ids = input_ids[:, 1:]
        mask_loss = labels[:, 1:].astype(jnp.float32)

        use_self_old = "old_per_token_logps" not in inputs
        if use_self_old:
            old_per_token_logps = jnp.zeros_like(mask_loss, dtype=jnp.float32)
        else:
            old_per_token_logps = inputs["old_per_token_logps"].astype(jnp.float32)

        params = self.variables.get("params", {})
        model_params = params.get("model", params)
        lm_head_kernel = model_params["lm_head"]["kernel"]

        fused_cfg = GRPOLmHeadFusedConfig(
            vocab_block_size=int(self.fused_cfg.vocab_block_size),
            epsilon_low=float(self.epsilon_low),
            epsilon_high=float(self.epsilon_high),
            temperature=float(self.temperature),
        )

        per_token_loss, per_token_logps = grpo_per_token_loss_fused_lm_head(
            hidden_states=hidden_for_loss,
            lm_head_kernel=lm_head_kernel,
            chosen_ids=chosen_ids,
            old_per_token_logps=old_per_token_logps,
            advantages=inputs["advantages"],
            cfg=fused_cfg,
            use_self_old=bool(use_self_old),
        )

        total_valid_token_count = inputs.get("total_valid_token_count", mask_loss.sum())
        loss = (per_token_loss * mask_loss).sum() / total_valid_token_count

        # Entropy metrics are monitoring-only; keep them cheap here to preserve the
        # memory win from avoiding full `[B,T,V]` logits materialization.
        batch = int(input_ids.shape[0])
        entropy = jnp.zeros((batch,), dtype=jnp.float32)
        entropy_loss = jnp.zeros((batch,), dtype=jnp.float32)

        return {
            "loss": loss,
            "per_token_logps": jax.lax.stop_gradient(per_token_logps),
            "entropy": entropy,
            "entropy_loss": entropy_loss,
        }


__all__ = ["TrainGRPOModulePallas", "TrainGRPOModuleFusedLmHead"]
