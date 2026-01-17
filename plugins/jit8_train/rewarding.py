from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Mapping

import numpy as np


def compute_weighted_rewards(
    *,
    reward_funcs: Sequence[Callable[..., float]],
    reward_weights: Sequence[float],
    inputs: Sequence[Mapping[str, Any]],
    answers: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-function rewards and the weighted sum reward per sample.

    The contract intentionally mirrors the logic in the original `test_jit8.py` loop:
    - Each reward function is applied per (input, answer).
    - Exceptions are caught per-sample and mapped to reward=-1 (to keep training moving).
    """
    if len(reward_funcs) != len(reward_weights):
        raise ValueError(
            f"reward_funcs and reward_weights must have same length, got {len(reward_funcs)} vs {len(reward_weights)}"
        )
    if len(inputs) != len(answers):
        raise ValueError(f"inputs and answers must have same length, got {len(inputs)} vs {len(answers)}")

    rewards_per_func = np.zeros((len(reward_funcs), len(answers)), dtype=np.float32)
    for i, (weight, reward_func) in enumerate(zip(reward_weights, reward_funcs)):
        weight_f = float(weight)
        for j, (inp, ans) in enumerate(zip(inputs, answers)):
            try:
                rewards_per_func[i, j] = weight_f * float(reward_func(inp, ans))
            except Exception as e:
                print(e)
                rewards_per_func[i, j] = -1.0

    rewards = rewards_per_func.sum(axis=0)
    return rewards_per_func, rewards


__all__ = ["compute_weighted_rewards"]

