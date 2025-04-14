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

CRITERION_COLLECTION = {
    "ce": optax.softmax_cross_entropy,
    "bce": lambda x, y: optax.sigmoid_binary_cross_entropy(x, y > 0).mean(-1),
}


class TrainSFTModule(nn.Module):
    model: Any
    pad_token_id: float
    label_smoothing: float = 0.0
    criterion: Callable[[Array, Array], Array] = CRITERION_COLLECTION["ce"]

    def __call__(self, inputs) -> ArrayTree:
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        labels = inputs['labels']

        logits, cache = self.model(input_ids=input_ids[:, :-1],
                                   attention_mask=attention_mask[:, :-1])

        targets = jax.nn.one_hot(labels[:, 1:], logits.shape[-1])
        loss = optax.softmax_cross_entropy(logits, targets)
        mask_loss = labels[:, 1:] != self.pad_token_id
        loss = loss * mask_loss
        loss_s = loss.sum() / (mask_loss.astype(jnp.float32).sum())

        targets = targets == targets.max(-1, keepdims=True)

        # Instead of directly comparing the maximum classes of predicted logits with the
        # given one-hot labels, we will check if the predicted classes are within the
        # label set. This approach is equivalent to traditional methods in single-label
        # classification and also supports multi-label tasks.
        preds = jax.lax.top_k(logits, k=5)[1]
        accs = jnp.take_along_axis(targets, preds, axis=-1)
        return {"loss": loss_s, }


# def selective_log_softmax_jax_inner(logits,index):
#     return jnp.take_along_axis(  # [B, S]
#         jax.nn.log_softmax(logits, axis=-1), index[..., None], axis=-1
#     )[..., 0]
#
#
# def selective_log_softmax_jax(logits: jnp.ndarray, index: jnp.ndarray,mesh) -> jnp.ndarray:
#     """
#     A memory-efficient JAX implementation for `log(softmax(logits))[index]`.
#
#     Equivalent to: `jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), index[..., None], axis=-1).squeeze(axis=-1)`
#     but avoids materializing the full log_softmax result.
#
#     Args:
#         logits: Logits tensor of shape `(..., num_classes)`.
#         index: Index tensor of shape `(...)`, specifying the positions to gather
#                from the log-softmax output along the last axis.
#
#     Returns:
#         Gathered log probabilities with the same shape as `index`.
#
#     """
#
#
#     return shard_map(jax.vmap(selective_log_softmax_jax_inner),mesh=mesh,
#                      in_specs=P(['dp','fsdp']),out_specs=P(['dp','fsdp'])
#                      )(logits,index)


def selective_log_softmax_jax(logits: jnp.ndarray, index: jnp.ndarray) -> jnp.ndarray:
    """
    A memory-efficient JAX implementation for `log(softmax(logits))[index]`.

    Equivalent to: `jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), index[..., None], axis=-1).squeeze(axis=-1)`
    but avoids materializing the full log_softmax result.

    Args:
        logits: Logits tensor of shape `(..., num_classes)`.
        index: Index tensor of shape `(...)`, specifying the positions to gather
               from the log-softmax output along the last axis.

    Returns:
        Gathered log probabilities with the same shape as `index`.
    """
    # Ensure index has the same number of dimensions as logits for broadcasting subtraction later
    # Or ensure index can be broadcasted correctly against logits dimensions before the last one.
    # If index is (...) and logits is (..., V), ensure (...) matches the prefix shape.

    # Gather the logits corresponding to the target indices: shape (...)
    # Add a dimension to index for take_along_axis, then remove it
    selected_logits = jnp.take_along_axis(logits, index[..., None], axis=-1).squeeze(axis=-1)

    # Calculate the logsumexp normalization factor across the last dimension: shape (...)
    logsumexp_logits = jax.scipy.special.logsumexp(logits, axis=-1)

    # Apply the identity: log_softmax(x)_i = x_i - logsumexp(x)
    per_token_logps = selected_logits - logsumexp_logits

    return per_token_logps


class TrainGRPOModule(nn.Module):
    model: Any
    pad_token_id: float
    num_pre_Q: int # Still present, purpose unclear in this snippet
    ref_model: Any = None
    beta: float = 0.04
    temperature: float = 1.0
    max_lengths: float = 2048 # Still present, not used in final loss below
    epsilon_low: float = 0.2
    epsilon_high: float = 0.28
    entropy_threshold: float = 0.3 # Used only for monitoring metrics now

    def __call__(self, inputs) -> ArrayTree:
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        labels = inputs['labels']
        # Original rewards/advantages are ignored for the PPO loss calculation now
        # rewards = inputs['rewards']
        # original_advantages = inputs.get("advantages", None) # We won't use this

        B = input_ids.shape[0] # Get batch size

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
        mask_loss = (labels[:, 1:] != self.pad_token_id).astype(jnp.float32) # Shape: [B, L-1]
        # Avoid division by zero for counts
        total_valid_token_count = jnp.maximum(mask_loss.sum(), 1e-6)

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

        # 3. Rank samples based on average entropy
        # argsort(argsort) gives ranks: 0 for lowest entropy, B-1 for highest
        ranks = jnp.argsort(jnp.argsort(avg_entropy_per_sample)) # Shape: [B]

        # 4. Scale ranks to advantage range [-1, 1]
        # Handle edge case B=1 separately to avoid division by zero and define behavior

        # Linear scaling: rank 0 -> -1, rank B-1 -> +1
        rank_based_advantages = -1.0 + 2.0 * ranks.astype(jnp.float32) / (B - 1.0) # Shape: [B]

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

        # Use rank_based_advantages, broadcasted to per-token shape [B, L-1]
        # Advantages shape [B] -> [B, 1] for broadcasting
        adv_broadcast = (0.1*rank_based_advantages+inputs['advantages'])[:, None]

        per_token_loss1 = ratio * adv_broadcast
        per_token_loss2 = clipped_ratio * adv_broadcast
        per_token_ppo_loss = jnp.minimum(per_token_loss1, per_token_loss2) # Shape: [B, L-1]

        # --- KL Penalty (Optional) ---
        if self.beta != 0 and self.ref_model is not None:
             # Using the reverse KL formulation from original code:
             # D_KL_rev(Policy || Ref) related term? Check literature if this is standard.
             # Formula: exp(log_ref - log_policy) - (log_ref - log_policy) - 1
            per_token_kl = jnp.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            # Combine: Loss = -(PPO - beta * KL) = beta * KL - PPO
            per_token_loss = self.beta * per_token_kl - per_token_ppo_loss
        else:
            # Loss = -PPO (negate because we minimize loss, PPO maximizes objective)
             per_token_loss = -per_token_ppo_loss

        # --- Final Loss Averaging ---
        # Apply mask and average over all valid tokens in the batch
        masked_loss = per_token_loss * mask_loss
        loss = masked_loss.sum() / total_valid_token_count

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
        sum_entropy_metric = (token_entropy * entropy_calc_mask).sum()
        count_entropy_metric = jnp.maximum(entropy_calc_mask.sum(), 1e-6)
        avg_entropy_metric = sum_entropy_metric / count_entropy_metric

        # Loss/Penalty associated with low entropy within that masked region
        low_entropy_mask = jnp.logical_and(entropy_calc_mask, token_entropy < self.entropy_threshold)
        sum_entropy_loss_metric = (token_entropy * low_entropy_mask).sum()
        count_entropy_loss_metric = jnp.maximum(low_entropy_mask.sum(), 1e-6)
        entropy_loss_metric = sum_entropy_loss_metric / count_entropy_loss_metric # More like 'low_entropy_penalty_metric'

        # --- Return Dictionary ---
        # Stop gradient on values returned only for monitoring or next step's input
        return {
            "loss": loss, # The main loss to minimize
            'per_token_logps': jax.lax.stop_gradient(per_token_logps), # Needed for next iteration's old_logps
            # Monitoring outputs related to original entropy calculation:
            'entropy': avg_entropy_metric,
            'entropy_loss': entropy_loss_metric,
        }