from __future__ import annotations

import math

import numpy as np

from projects.minionerec.sft.metrics import normalize_sid_text


def build_rank_penalties(num_generations: int) -> np.ndarray:
    """Rank-aware penalties used in upstream `rl.py` (negative, sum=-1)."""
    k = int(num_generations)
    if k <= 0:
        raise ValueError("num_generations must be > 0")
    raw = np.asarray([-1.0 / math.log2(i + 2) for i in range(k)], dtype=np.float32)
    denom = float(raw.sum())
    if denom == 0.0:
        raise ValueError("Invalid rank penalty normalization (sum==0)")
    return (-raw / denom).astype(np.float32)


def compute_ranking_rewards(
    *,
    predictions: list[list[str]],
    targets: list[str],
    rank_penalties: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (reward, correct) per completion.

    Matches upstream `rl.py` behavior for `reward_type=ranking`:
    - If a group contains the correct item, incorrect ranks get a negative penalty, correct rank gets +1.
    - If a group contains no correct item, all rewards are 0.
    """
    if len(predictions) != len(targets):
        raise ValueError("predictions/targets length mismatch")
    k = int(rank_penalties.shape[0])
    if k <= 0:
        raise ValueError("Empty rank_penalties")

    rewards = np.zeros((len(predictions), k), dtype=np.float32)
    correct = np.zeros((len(predictions), k), dtype=np.float32)
    for i, (preds, target) in enumerate(zip(predictions, targets, strict=True)):
        if len(preds) != k:
            raise ValueError(f"Expected {k} predictions per prompt, got {len(preds)}")
        target_norm = normalize_sid_text(target)
        hit = None
        preds_norm = [normalize_sid_text(p) for p in preds]
        for j, p in enumerate(preds_norm):
            if p == target_norm:
                hit = j
                break
        if hit is None:
            continue
        rewards[i, :] = rank_penalties
        rewards[i, hit] = 1.0
        correct[i, hit] = 1.0
    return rewards.reshape(-1), correct.reshape(-1)


__all__ = ["build_rank_penalties", "compute_ranking_rewards"]

