"""Reward phase (trajectory -> scalar rewards)."""

from plugins.training.reward.modules import WeightedRewardModule
from plugins.training.reward.weighted import compute_weighted_rewards

__all__ = ["WeightedRewardModule", "compute_weighted_rewards"]
