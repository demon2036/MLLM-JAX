"""Reusable training modules (plugins-first).

This package contains small, composable building blocks for the RL training
pipeline phases:
- rollout
- reward
- advantages
- update
"""

from plugins.training.modules.grpo_sync import (
    GRPOSyncRollout,
    GroupIdGRPOAdvantageModule,
    PPOUpdateModule,
    WeightedRewardModule,
)

__all__ = [
    "GRPOSyncRollout",
    "GroupIdGRPOAdvantageModule",
    "PPOUpdateModule",
    "WeightedRewardModule",
]
