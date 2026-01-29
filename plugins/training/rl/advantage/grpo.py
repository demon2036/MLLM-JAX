from __future__ import annotations

from typing import Any

import numpy as np


def compute_grpo_advantages_by_group_id(
    *,
    rewards: Any,
    group_ids: Any,
    eps: float = 1e-4,
) -> np.ndarray:
    """Compute GRPO-style advantages within each `group_id`.

    This avoids brittle reshapes like `rewards.reshape(-1, group_size)` by using
    an explicit grouping vector. It supports variable group sizes.

    Parameters
    ----------
    rewards:
        Array-like shape [B].
    group_ids:
        Array-like shape [B] (int-like). Samples with the same group id share a baseline.
    eps:
        Numerical stabilizer for division.
    """
    rewards_np = np.asarray(rewards, dtype=np.float32)
    group_ids_np = np.asarray(group_ids)

    if rewards_np.ndim != 1:
        raise ValueError(f"rewards must be rank-1 [B], got shape={rewards_np.shape}")
    if group_ids_np.shape != rewards_np.shape:
        raise ValueError(f"group_ids must have same shape as rewards, got {group_ids_np.shape} vs {rewards_np.shape}")

    # Map potentially non-contiguous ids -> [0, num_groups).
    _unique, inv = np.unique(group_ids_np, return_inverse=True)
    num_groups = int(inv.max()) + 1 if inv.size else 0
    if num_groups <= 0:
        raise ValueError("No groups found (empty batch?)")

    group_sum = np.bincount(inv, weights=rewards_np, minlength=num_groups).astype(np.float32)
    group_count = np.bincount(inv, minlength=num_groups).astype(np.float32)
    group_count = np.maximum(group_count, 1.0)
    group_mean = group_sum / group_count

    centered = rewards_np - group_mean[inv]
    group_var = np.bincount(inv, weights=centered * centered, minlength=num_groups).astype(np.float32) / group_count
    group_std = np.sqrt(group_var)

    advantages = centered / (group_std[inv] + float(eps))
    return advantages.astype(np.float32)


__all__ = ["compute_grpo_advantages_by_group_id"]

