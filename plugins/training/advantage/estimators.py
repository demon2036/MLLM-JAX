from __future__ import annotations

from typing import Any

import numpy as np

from plugins.training.advantage.grpo import compute_grpo_advantages_by_group_id


def _as_1d_float32(x: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be rank-1 [B], got shape={arr.shape}")
    if arr.size <= 0:
        raise ValueError(f"{name} must be non-empty")
    return arr


def _as_1d_group_ids(x: Any, *, name: str, expected_size: int) -> np.ndarray:
    arr = np.asarray(x).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be rank-1 [B], got shape={arr.shape}")
    if int(arr.size) != int(expected_size):
        raise ValueError(f"{name} must have shape [B] with B={expected_size}, got size={arr.size}")
    return arr


def _maybe_clip(advantages: np.ndarray, clip_range: float | None) -> np.ndarray:
    if clip_range is None:
        return advantages
    clip = float(clip_range)
    if clip <= 0:
        raise ValueError("clip_range must be > 0 when set")
    return np.clip(advantages, -clip, clip)


def compute_global_normalized_advantages(
    *,
    rewards: Any,
    eps: float = 1e-4,
    clip_range: float | None = None,
) -> np.ndarray:
    """REINFORCE-style baseline: center+whiten rewards globally.

    Returns float32 advantages of shape [B].
    """
    rewards_np = _as_1d_float32(rewards, name="rewards")
    eps_f = float(eps)
    if eps_f <= 0:
        raise ValueError("eps must be > 0")

    mean = float(rewards_np.mean())
    std = float(rewards_np.std())
    advantages = (rewards_np - mean) / (std + eps_f)
    return _maybe_clip(advantages.astype(np.float32), clip_range)


def compute_rloo_advantages_by_group_id(
    *,
    rewards: Any,
    group_ids: Any,
    eps: float = 1e-4,
    whiten: bool = True,
    clip_range: float | None = None,
) -> np.ndarray:
    """RLOO (Reinforce Leave-One-Out) per-group baseline.

    For each sample i in a prompt-group:
      baseline_i = mean(rewards of the other samples in the same group)
      advantage_i = reward_i - baseline_i

    If a group has size 1, the leave-one-out baseline is undefined; we fall back
    to the group mean (which equals the reward itself), resulting in advantage 0.
    """
    rewards_np = _as_1d_float32(rewards, name="rewards")
    group_ids_np = _as_1d_group_ids(group_ids, name="group_ids", expected_size=int(rewards_np.size))

    eps_f = float(eps)
    if eps_f <= 0:
        raise ValueError("eps must be > 0")

    _unique, inv = np.unique(group_ids_np, return_inverse=True)
    num_groups = int(inv.max()) + 1 if inv.size else 0
    if num_groups <= 0:
        raise ValueError("No groups found (empty batch?)")

    group_sum = np.bincount(inv, weights=rewards_np, minlength=num_groups).astype(np.float32)
    group_count = np.bincount(inv, minlength=num_groups).astype(np.int32)

    count_per_sample = group_count[inv].astype(np.float32)
    denom = np.maximum(count_per_sample - 1.0, 1.0)
    loo_mean = (group_sum[inv] - rewards_np) / denom
    group_mean = group_sum[inv] / np.maximum(count_per_sample, 1.0)
    baseline = np.where(count_per_sample > 1.0, loo_mean, group_mean).astype(np.float32)

    advantages = (rewards_np - baseline).astype(np.float32)

    if whiten:
        mean = float(advantages.mean())
        std = float(advantages.std())
        advantages = (advantages - mean) / (std + eps_f)

    return _maybe_clip(advantages.astype(np.float32), clip_range)


def compute_dapo_advantages_by_group_id(
    *,
    rewards: Any,
    group_ids: Any,
    eps: float = 1e-4,
    alpha: float = 0.2,
    clip_range: float | None = None,
) -> np.ndarray:
    """DAPO-style mixed advantages (group-normalized + global-normalized).

    This follows a common pattern in GRPO variants: keep the per-prompt-group
    normalization, but blend in a global baseline term to stabilize scaling
    across batches.
    """
    rewards_np = _as_1d_float32(rewards, name="rewards")
    group_ids_np = _as_1d_group_ids(group_ids, name="group_ids", expected_size=int(rewards_np.size))

    eps_f = float(eps)
    if eps_f <= 0:
        raise ValueError("eps must be > 0")

    alpha_f = float(alpha)
    if alpha_f < 0:
        raise ValueError("alpha must be >= 0")

    grpo_adv = compute_grpo_advantages_by_group_id(rewards=rewards_np, group_ids=group_ids_np, eps=eps_f)
    global_adv = compute_global_normalized_advantages(rewards=rewards_np, eps=eps_f, clip_range=None)
    mixed = grpo_adv.astype(np.float32) + alpha_f * global_adv.astype(np.float32)
    return _maybe_clip(mixed.astype(np.float32), clip_range)


def compute_reinforce_plus_plus_advantages_by_group_id(
    *,
    rewards: Any,
    group_ids: Any,
    eps: float = 1e-4,
    clip_range: float | None = None,
) -> np.ndarray:
    """REINFORCE++ (pragmatic): RLOO baseline + global whitening.

    In many RLHF implementations, "REINFORCE++" refers to a variance-reduced
    outcome policy-gradient baseline with whitening and (optionally) advantage
    clipping. In this repo we implement it as:
      advantage = whiten(reward - leave_one_out_mean(reward within prompt group))
    """
    return compute_rloo_advantages_by_group_id(
        rewards=rewards,
        group_ids=group_ids,
        eps=eps,
        whiten=True,
        clip_range=clip_range,
    )


def build_token_rewards_from_final(
    *,
    rewards: Any,
    completion_mask: Any,
) -> np.ndarray:
    rewards_np = _as_1d_float32(rewards, name="rewards")
    mask = np.asarray(completion_mask, dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError(f"completion_mask must be rank-2 [B, T], got shape={mask.shape}")
    if int(mask.shape[0]) != int(rewards_np.size):
        raise ValueError(f"completion_mask batch must match rewards (B={rewards_np.size}), got {mask.shape[0]}")
    token_rewards = np.zeros_like(mask, dtype=np.float32)
    positions = np.where(mask > 0, np.arange(mask.shape[1], dtype=np.int32), -1)
    last_indices = positions.max(axis=1)
    for i, idx in enumerate(last_indices):
        if int(idx) >= 0:
            token_rewards[i, int(idx)] = rewards_np[i]
    return token_rewards


def compute_gae_advantages(
    *,
    rewards: Any,
    values: Any,
    completion_mask: Any,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    normalize: bool = True,
    eps: float = 1e-4,
    clip_range: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE advantages and returns (token-level).

    - rewards: shape [B] (sequence-level reward)
    - values: shape [B, T] (token-level values)
    - completion_mask: shape [B, T] (1 for completion tokens)
    """
    rewards_np = _as_1d_float32(rewards, name="rewards")
    values_np = np.asarray(values, dtype=np.float32)
    mask = np.asarray(completion_mask, dtype=np.float32)
    if values_np.ndim != 2:
        raise ValueError(f"values must be rank-2 [B, T], got shape={values_np.shape}")
    if mask.ndim != 2:
        raise ValueError(f"completion_mask must be rank-2 [B, T], got shape={mask.shape}")
    if int(values_np.shape[0]) != int(rewards_np.size):
        raise ValueError(f"values batch must match rewards (B={rewards_np.size}), got {values_np.shape[0]}")
    if values_np.shape != mask.shape:
        raise ValueError(f"values and completion_mask must match shape, got {values_np.shape} vs {mask.shape}")

    token_rewards = build_token_rewards_from_final(rewards=rewards_np, completion_mask=mask)
    advantages = np.zeros_like(values_np, dtype=np.float32)
    last_gae = np.zeros((values_np.shape[0],), dtype=np.float32)
    next_values = np.zeros((values_np.shape[0],), dtype=np.float32)

    gamma_f = float(gamma)
    lambda_f = float(gae_lambda)
    for t in reversed(range(values_np.shape[1])):
        mask_t = mask[:, t]
        delta = token_rewards[:, t] + gamma_f * next_values - values_np[:, t]
        last_gae = (delta + gamma_f * lambda_f * last_gae) * mask_t
        advantages[:, t] = last_gae
        next_values = values_np[:, t] * mask_t

    returns = advantages + values_np

    if normalize:
        valid = mask > 0
        if valid.any():
            adv_valid = advantages[valid]
            mean = float(adv_valid.mean())
            std = float(adv_valid.std())
            advantages = np.where(valid, (advantages - mean) / (std + float(eps)), 0.0)

    if clip_range is not None:
        clip = float(clip_range)
        if clip <= 0:
            raise ValueError("clip_range must be > 0 when set")
        advantages = np.clip(advantages, -clip, clip).astype(np.float32)

    return advantages.astype(np.float32), returns.astype(np.float32)


__all__ = [
    "compute_global_normalized_advantages",
    "compute_rloo_advantages_by_group_id",
    "compute_dapo_advantages_by_group_id",
    "compute_reinforce_plus_plus_advantages_by_group_id",
    "build_token_rewards_from_final",
    "compute_gae_advantages",
]
