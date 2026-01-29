"""Advantage estimation phase (rewards -> advantages)."""

from plugins.training.rl.advantage.grpo import compute_grpo_advantages_by_group_id
from plugins.training.rl.advantage.modules import CallableAdvantageModule, GroupIdGRPOAdvantageModule

__all__ = [
    "CallableAdvantageModule",
    "GroupIdGRPOAdvantageModule",
    "compute_grpo_advantages_by_group_id",
]
