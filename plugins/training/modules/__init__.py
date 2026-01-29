"""Backward-compatible imports for training modules.

New code should import phase modules directly:
- `plugins.training.rollout.modules`
- `plugins.training.reward.modules`
- `plugins.training.advantage.modules`
- `plugins.training.update.modules`
"""

from plugins.training.rl.advantage.modules import GroupIdGRPOAdvantageModule
from plugins.training.rl.reward.modules import WeightedRewardModule
from plugins.training.rl.rollout.modules import GRPOSyncRollout, RolloutBackendModule
from plugins.training.rl.update.modules import PPOUpdateModule

__all__ = [
    "GRPOSyncRollout",
    "GroupIdGRPOAdvantageModule",
    "PPOUpdateModule",
    "RolloutBackendModule",
    "WeightedRewardModule",
]
