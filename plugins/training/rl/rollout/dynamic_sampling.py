from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DynamicSamplingConfig:
    enabled: bool = False
    trigger: str = "homogeneous_group"
    metric: str = "acc"
    homogeneity_threshold: float = 1.0
    min_unique_reward_values: int = 2
    max_extra_roll_rounds: int = 10
    target_valid_groups: int | None = None
    fallback_policy: str = "keep_last"


@dataclass(frozen=True)
class DynamicSamplingSummary:
    total_groups: int
    initially_homogeneous_groups: int
    remaining_homogeneous_groups: int
    extra_rounds: int
    hit_budget: bool
    dropped_groups: int


def _normalize_metric_values(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    if vals.size == 0:
        return vals
    # Round slightly for robust equality under small float jitter.
    return np.round(vals, 6)


def _group_homogeneous_flags(
    *,
    values_by_group: np.ndarray,
    homogeneity_threshold: float,
    min_unique_reward_values: int,
) -> np.ndarray:
    arr = np.asarray(values_by_group, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"values_by_group must be rank-2 [G, K], got shape={arr.shape}")

    g = int(arr.shape[0])
    flags = np.zeros(g, dtype=bool)
    for i in range(g):
        row = _normalize_metric_values(arr[i])
        if row.size == 0:
            flags[i] = True
            continue
        unique, counts = np.unique(row, return_counts=True)
        num_unique = int(unique.size)
        dominant_frac = float(counts.max()) / float(row.size)
        diverse = (num_unique >= int(min_unique_reward_values)) and (dominant_frac < float(homogeneity_threshold))
        flags[i] = not diverse
    return flags


def homogeneous_group_flags(
    *,
    rewards: Any,
    metric_values: Any | None,
    n: int,
    metric: str,
    homogeneity_threshold: float,
    min_unique_reward_values: int,
) -> np.ndarray:
    rewards_np = np.asarray(rewards, dtype=np.float32).reshape(-1)
    if rewards_np.size % int(n) != 0:
        raise ValueError(f"rewards size must be divisible by n={n}, got size={rewards_np.size}")
    group_count = int(rewards_np.size // int(n))

    if metric == "seq_reward":
        source = rewards_np
    elif metric == "acc":
        if metric_values is None:
            source = rewards_np
        else:
            source = np.asarray(metric_values, dtype=np.float32).reshape(-1)
            if source.size != rewards_np.size:
                raise ValueError(
                    "metric_values for metric='acc' must match rewards size, "
                    f"got {source.size} vs {rewards_np.size}"
                )
    else:
        raise ValueError(f"Unsupported metric={metric!r}; expected 'acc' or 'seq_reward'")

    grouped = source.reshape(group_count, int(n))
    return _group_homogeneous_flags(
        values_by_group=grouped,
        homogeneity_threshold=float(homogeneity_threshold),
        min_unique_reward_values=int(min_unique_reward_values),
    )


def select_completion_indices(
    *,
    keep_groups_mask: np.ndarray,
    n: int,
) -> np.ndarray:
    keep = np.asarray(keep_groups_mask, dtype=bool).reshape(-1)
    base = np.arange(int(keep.size), dtype=np.int32)
    kept_groups = base[keep]
    if kept_groups.size == 0:
        return np.asarray([], dtype=np.int32)
    offsets = np.arange(int(n), dtype=np.int32)
    return (kept_groups[:, None] * int(n) + offsets[None, :]).reshape(-1)


def build_dynamic_sampling_summary(
    *,
    initial_homogeneous: np.ndarray,
    final_homogeneous: np.ndarray,
    extra_rounds: int,
    max_extra_roll_rounds: int,
    dropped_groups: int,
) -> DynamicSamplingSummary:
    initial = np.asarray(initial_homogeneous, dtype=bool).reshape(-1)
    final = np.asarray(final_homogeneous, dtype=bool).reshape(-1)
    return DynamicSamplingSummary(
        total_groups=int(initial.size),
        initially_homogeneous_groups=int(initial.sum()),
        remaining_homogeneous_groups=int(final.sum()),
        extra_rounds=int(extra_rounds),
        hit_budget=(int(extra_rounds) >= int(max_extra_roll_rounds)) and bool(final.any()),
        dropped_groups=int(dropped_groups),
    )


__all__ = [
    "DynamicSamplingConfig",
    "DynamicSamplingSummary",
    "build_dynamic_sampling_summary",
    "homogeneous_group_flags",
    "select_completion_indices",
]

