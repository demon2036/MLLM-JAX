from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from plugins.api.training import RewardResult
from plugins.training.rl.reward.weighted import compute_weighted_rewards


@dataclass(frozen=True)
class WeightedRewardModule:
    """Weighted reward composition over multiple reward functions."""

    reward_funcs: Sequence[Callable[..., float]]
    reward_weights: Sequence[float]

    def compute(self, *, inputs: Sequence[Mapping[str, Any]], answers: Sequence[str]) -> RewardResult:
        rewards_per_func, rewards = compute_weighted_rewards(
            reward_funcs=self.reward_funcs,
            reward_weights=self.reward_weights,
            inputs=inputs,
            answers=answers,
        )
        return RewardResult(rewards=rewards, rewards_per_func=rewards_per_func)


__all__ = ["WeightedRewardModule"]
