"""Reward phase (trajectory -> scalar rewards)."""

from plugins.training.rl.reward.modules import WeightedRewardModule
from plugins.training.rl.reward.weighted import compute_weighted_rewards

__all__ = ["WeightedRewardModule", "compute_weighted_rewards"]
