"""Backward-compatible imports for training modules.

New code should import phase modules directly:
- `plugins.training.rollout.modules`
- `plugins.training.reward.modules`
- `plugins.training.advantage.modules`
- `plugins.training.update.modules`
"""

from plugins.training.advantage.modules import GroupIdGRPOAdvantageModule
from plugins.training.reward.modules import WeightedRewardModule
from plugins.training.rollout.modules import GRPOSyncRollout, RolloutBackendModule
from plugins.training.update.modules import PPOUpdateModule

__all__ = [
    "GRPOSyncRollout",
    "GroupIdGRPOAdvantageModule",
    "PPOUpdateModule",
    "RolloutBackendModule",
    "WeightedRewardModule",
]
