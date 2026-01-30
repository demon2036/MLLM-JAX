from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as PS

from plugins.training.kernels.grpo_loss_pallas import (
    GRPOKernelConfig,
    grpo_per_token_loss_pallas,
    grpo_per_token_loss_pallas_on_policy,
    grpo_per_token_loss_pallas_on_policy_with_entropy,
    grpo_per_token_loss_pallas_with_entropy,
)


class TrainGRPOModulePallas(nn.Module):
    """TrainGRPOModule variant that uses the Pallas GRPO kernel for policy loss."""

    model: Any
    mesh: Any
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
    pallas_compute_dtype: str = "f32"
    pallas_bwd_output_alias_logits: bool = False

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

        # NOTE: We keep the external contract (and `old_per_token_logps`) as
        # length `L-1` aligned with `input_ids[:, 1:]`, but we call the kernel
        # on the full `L` logits to avoid padding/copying `logits[:, :-1, :]`
        # back to a multiple of `time_block` inside the kernel.
        chosen_ids = input_ids[:, 1:]
        chosen_ids_full = jnp.pad(chosen_ids, ((0, 0), (0, 1)), constant_values=int(self.pad_token_id))

        mask_loss = labels[:, 1:]

        kernel_cfg = GRPOKernelConfig(
            block_size=int(self.pallas_block_size),
            time_block=int(self.pallas_time_block),
            epsilon_low=float(self.epsilon_low),
            epsilon_high=float(self.epsilon_high),
            temperature=float(self.temperature),
            bwd_output_alias_logits=bool(self.pallas_bwd_output_alias_logits),
            compute_dtype=str(self.pallas_compute_dtype),
        )
        advantages = inputs["advantages"]

        def _call_kernel_on_policy(local_logits, local_chosen_ids, local_advantages):
            return grpo_per_token_loss_pallas_on_policy(
                logits=local_logits,
                chosen_ids=local_chosen_ids,
                advantages=local_advantages,
                cfg=kernel_cfg,
                interpret=False,
                debug=False,
            )

        def _call_kernel_on_policy_with_entropy(local_logits, local_chosen_ids, local_advantages):
            return grpo_per_token_loss_pallas_on_policy_with_entropy(
                logits=local_logits,
                chosen_ids=local_chosen_ids,
                advantages=local_advantages,
                cfg=kernel_cfg,
                interpret=False,
                debug=False,
            )

        def _call_kernel(local_logits, local_chosen_ids, local_old_logps, local_advantages):
            return grpo_per_token_loss_pallas(
                logits=local_logits,
                chosen_ids=local_chosen_ids,
                old_per_token_logps=local_old_logps,
                advantages=local_advantages,
                cfg=kernel_cfg,
                interpret=False,
                debug=False,
            )

        def _call_kernel_with_entropy(local_logits, local_chosen_ids, local_old_logps, local_advantages):
            return grpo_per_token_loss_pallas_with_entropy(
                logits=local_logits,
                chosen_ids=local_chosen_ids,
                old_per_token_logps=local_old_logps,
                advantages=local_advantages,
                cfg=kernel_cfg,
                interpret=False,
                debug=False,
            )

        fuse_entropy_metrics = float(self.temperature) == 1.0

        if "old_per_token_logps" in inputs:
            old_per_token_logps = inputs["old_per_token_logps"]
            old_per_token_logps_full = jnp.pad(old_per_token_logps, ((0, 0), (0, 1)), constant_values=0.0)
            if fuse_entropy_metrics:
                per_token_loss, per_token_logps, token_entropy = shard_map(
                    _call_kernel_with_entropy,
                    mesh=self.mesh,
                    in_specs=(
                        PS(("dp", "fsdp"), None, "tp"),
                        PS(("dp", "fsdp"), None),
                        PS(("dp", "fsdp"), None),
                        PS(("dp", "fsdp"),),
                    ),
                    out_specs=(
                        PS(("dp", "fsdp"), None),
                        PS(("dp", "fsdp"), None),
                        PS(("dp", "fsdp"), None),
                    ),
                    check_rep=False,
                )(
                    logits,
                    chosen_ids_full,
                    old_per_token_logps_full,
                    advantages,
                )
            else:
                per_token_loss, per_token_logps = shard_map(
                    _call_kernel,
                    mesh=self.mesh,
                    in_specs=(
                        PS(("dp", "fsdp"), None, "tp"),
                        PS(("dp", "fsdp"), None),
                        PS(("dp", "fsdp"), None),
                        PS(("dp", "fsdp"),),
                    ),
                    out_specs=(
                        PS(("dp", "fsdp"), None),
                        PS(("dp", "fsdp"), None),
                    ),
                    check_rep=False,
                )(
                    logits,
                    chosen_ids_full,
                    old_per_token_logps_full,
                    advantages,
                )
        else:
            if fuse_entropy_metrics:
                per_token_loss, per_token_logps, token_entropy = shard_map(
                    _call_kernel_on_policy_with_entropy,
                    mesh=self.mesh,
                    in_specs=(
                        PS(("dp", "fsdp"), None, "tp"),
                        PS(("dp", "fsdp"), None),
                        PS(("dp", "fsdp"),),
                    ),
                    out_specs=(
                        PS(("dp", "fsdp"), None),
                        PS(("dp", "fsdp"), None),
                        PS(("dp", "fsdp"), None),
                    ),
                    check_rep=False,
                )(
                    logits,
                    chosen_ids_full,
                    advantages,
                )
            else:
                per_token_loss, per_token_logps = shard_map(
                    _call_kernel_on_policy,
                    mesh=self.mesh,
                    in_specs=(
                        PS(("dp", "fsdp"), None, "tp"),
                        PS(("dp", "fsdp"), None),
                        PS(("dp", "fsdp"),),
                    ),
                    out_specs=(
                        PS(("dp", "fsdp"), None),
                        PS(("dp", "fsdp"), None),
                    ),
                    check_rep=False,
                )(
                    logits,
                    chosen_ids_full,
                    advantages,
                )

        per_token_loss = per_token_loss[:, :-1]
        per_token_logps = per_token_logps[:, :-1]
        if fuse_entropy_metrics:
            token_entropy = token_entropy[:, :-1]

        if not fuse_entropy_metrics:
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
