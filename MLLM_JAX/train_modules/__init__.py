from __future__ import annotations

from typing import Callable, Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import torch
from chex import Array, ArrayTree
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

"""
avg_entropy_grouped = avg_entropy_per_sample.reshape(-1, groups)
# Ranks within each group (0=lowest entropy, groups-1=highest)
ranks_grouped = jnp.argsort(jnp.argsort(avg_entropy_grouped, axis=1), axis=1)
denom_grp = jnp.maximum(groups - 1.0, 0)  # Avoid division by zero if groups=1
entropy_scores_grouped = -1.0 + 2.0 * ranks_grouped.astype(jnp.float32) / denom_grp
entropy_scores = entropy_scores_grouped.reshape(-1)  # Shape [B]

# --- Apply High Entropy Threshold ---
# Create mask for samples with average entropy > threshold
high_entropy_mask = (avg_entropy_per_sample > entropy_threshold)  # Shape [B]

# Determine the final entropy score: 1.0 if entropy > threshold, otherwise use rank-based score
final_entropy_scores = jnp.where(
    high_entropy_mask,
    jnp.ones_like(entropy_scores),  # Use 1.0 for high entropy samples
    entropy_scores  # Use rank-based score otherwise
)  # Shape [B]

# 2. Combine original rewards with weighted entropy score
modified_rewards = rewards #+ alpha * final_entropy_scores  # Shape [B]
# 3. Apply standard GRPO normalization to the *modified* rewards
mean_grouped_mod_rewards = modified_rewards.reshape(-1, groups).mean(axis=1)
std_grouped_mod_rewards = modified_rewards.reshape(-1, groups).std(axis=1)
mean_grouped_mod_rewards = jnp.repeat(mean_grouped_mod_rewards, groups, axis=0)
std_grouped_mod_rewards = jnp.repeat(std_grouped_mod_rewards, groups, axis=0)
advantages = (modified_rewards - mean_grouped_mod_rewards) / (std_grouped_mod_rewards + 1e-4)
advantages = advantages + alpha * (rewards - rewards.mean()) / (rewards.std() + 1e-4)

group_max = advantages.reshape(-1, groups).max(axis=1)
scale_factors = 1.0 / (group_max + 1e-6)
scale_factors = jnp.repeat(scale_factors, groups, axis=0)
advantages = jnp.where(advantages > 0, advantages * scale_factors, advantages)

group_min = advantages.reshape(-1, groups).min(axis=1)
scale_factors_neg = -1.0 / (group_min + 1e-6)
scale_factors_neg = jnp.repeat(scale_factors_neg, groups, axis=0)

scale_factors_neg2 = -0.25/ (group_min + 1e-6)
scale_factors_neg2 = jnp.repeat(scale_factors_neg2, groups, axis=0)
scale_factors_neg=jnp.where(avg_entropy_per_sample<0.4,scale_factors_neg,scale_factors_neg2)
advantages = jnp.where(advantages < 0, advantages * scale_factors_neg, advantages)

return advantages
"""





def get_advantages(rewards,groups,alpha=0.2,avg_entropy_per_sample=None,entropy_threshold=0.4,per_token_kl=None):
    # if per_token_kl is not None:
    #     rewards=rewards-0.001* jnp.sum(per_token_kl,axis=-1)
    # mean_grouped_mod_rewards = rewards.reshape(-1, groups).mean(axis=1)
    # std_grouped_mod_rewards = rewards.reshape(-1, groups).std(axis=1)
    # mean_grouped_mod_rewards = jnp.repeat(mean_grouped_mod_rewards, groups, axis=0)
    # std_grouped_mod_rewards = jnp.repeat(std_grouped_mod_rewards, groups, axis=0)
    # advantages = (rewards - mean_grouped_mod_rewards) / (std_grouped_mod_rewards + 1e-4)

    mean=rewards.mean()
    std=rewards.std()

    advantages=(rewards-mean)/std

    return advantages





class TrainGRPOModule(nn.Module):
    model: Any
    pad_token_id: float
    num_pre_Q: int # Still present, purpose unclear in this snippet
    ref_model: Any = None
    beta: float = 0.04
    temperature: float = 1.0
    max_lengths: float = 2048 # Still present, not used in final loss below
    epsilon_low: float = 0.2
    epsilon_high: float = 0.3
    entropy_threshold: float = 0.3 # Used only for monitoring metrics now

    def __call__(self, inputs) -> ArrayTree:
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        labels = inputs['labels']
        # Original rewards/advantages are ignored for the PPO loss calculation now
        # rewards = inputs['rewards']
        # original_advantages = inputs.get("advantages", None) # We won't use this

        # --- Model Forward Pass ---
        logits, _ = self.model(input_ids=input_ids,
                               attention_mask=attention_mask) # Ignoring cache

        if self.beta != 0 and self.ref_model is not None:
            ref_logits, _ = self.ref_model(input_ids=input_ids,
                                           attention_mask=attention_mask)
            ref_logits = jax.lax.stop_gradient(ref_logits)
        elif self.beta != 0 and self.ref_model is None:
            print("Warning: beta is non-zero but ref_model not provided. KL penalty calculation will be skipped.")
            # Handle this case, maybe raise error or set beta=0 effective immediately?
            # For now, KL part will just not run correctly below if ref_model is None

        # --- Calculate Log Probs and Mask ---
        chosen_ids = input_ids[:, 1:]  # (B, L-1)
        # *** IMPORTANT: Use pad_token_id to create the mask correctly ***
        mask_loss = labels[:, 1:] # Shape: [B, L-1]
        # Avoid division by zero for counts

        # Log probs of chosen tokens under current policy
        per_token_logps = jnp.take_along_axis(  # [B, L-1]
            jax.nn.log_softmax(logits[..., :-1, :], axis=-1), chosen_ids[..., None], axis=-1
        )[..., 0] / self.temperature

        # Log probs under reference policy (if applicable)
        if self.beta != 0 and self.ref_model is not None:
            ref_per_token_logps = jnp.take_along_axis(  # [B, L-1]
                jax.nn.log_softmax(ref_logits[..., :-1, :], axis=-1), chosen_ids[..., None], axis=-1
            )[..., 0]

        # --- Calculate Entropy and Rank-Based Advantages ---
        # 1. Per-token entropy
        probs = jax.nn.softmax(logits[..., :-1, :] / self.temperature, axis=-1)
        # Add epsilon for numerical stability inside log
        token_entropy = -jnp.sum(probs * jax.lax.log(probs + 1e-9), axis=-1)  # Shape: [B, L-1]

        # 2. Average entropy per sample (using the correct mask)
        masked_token_entropy = token_entropy * mask_loss
        sum_entropy_per_sample = masked_token_entropy.sum(axis=-1) # Shape: [B]
        avg_entropy_per_sample = sum_entropy_per_sample / mask_loss.sum(axis=-1) # Shape: [B]

        # --- Monitoring Metrics (Optional: Original entropy/entropy_loss calculation) ---
        # These are NOT part of the main loss calculation anymore, just for logging.
        # Create the specific mask for tokens 4-100 (inclusive)
        valid_mask_for_metric = (mask_loss == 1)
        cum_valid = jnp.cumsum(valid_mask_for_metric, axis=-1)
        entropy_calc_mask = jnp.logical_and(
            valid_mask_for_metric,
            jnp.logical_and(cum_valid >= 4, cum_valid <= 100)
        )
        # Average entropy within the masked region (tokens 4-100)
        masked_token_entropy_truncated = token_entropy * entropy_calc_mask
        sum_entropy_per_sample_truncated = masked_token_entropy_truncated.sum(axis=-1)  # Shape: [B]
        avg_entropy_per_sample_truncated = sum_entropy_per_sample_truncated / entropy_calc_mask.sum(axis=-1)  # Shape: [B]

        # --- PPO Loss Calculation (using the NEW rank_based_advantages) ---
        # Get old log probs (from previous iteration or stop grad)
        if "old_per_token_logps" in inputs:
            old_per_token_logps = inputs["old_per_token_logps"]
            # print(f'Using provided old_per_token_logps {old_per_token_logps.shape=}') # Removed print
        else:
            old_per_token_logps = jax.lax.stop_gradient(per_token_logps)
            # print(f'Using calculated old_per_token_logps {old_per_token_logps.shape=}') # Removed print

        # PPO ratio and clipping
        ratio = jnp.exp(per_token_logps - old_per_token_logps)
        clipped_ratio = jnp.clip(ratio, 1.0 - self.epsilon_low, 1.0 + self.epsilon_high)

        # Advantages shape [B] -> [B, 1] for broadcasting
        adv_broadcast=inputs['advantages'][...,None]
        per_token_loss1 = ratio * adv_broadcast
        per_token_loss2 = clipped_ratio * adv_broadcast
        per_token_ppo_loss = jnp.minimum(per_token_loss1, per_token_loss2) # Shape: [B, L-1]

        per_token_loss=-per_token_ppo_loss
        total_valid_token_count = inputs.get("total_valid_token_count", mask_loss.sum())
        loss = ((per_token_loss * mask_loss).sum()) / total_valid_token_count


        # --- Return Dictionary ---
        # Stop gradient on values returned only for monitoring or next step's input
        return {
            "loss": loss, # The main loss to minimize
            'per_token_logps': jax.lax.stop_gradient(per_token_logps), # Needed for next iteration's old_logps
            # Monitoring outputs related to original entropy calculation:
            'entropy': avg_entropy_per_sample,
            'entropy_loss': avg_entropy_per_sample_truncated,
            # 'per_token_kl':per_token_kl_metrics
        }