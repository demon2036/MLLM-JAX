"""Advantage estimation phase (rewards -> advantages)."""

from plugins.training.advantage.grpo import compute_grpo_advantages_by_group_id
from plugins.training.advantage.modules import CallableAdvantageModule, GroupIdGRPOAdvantageModule

__all__ = [
    "CallableAdvantageModule",
    "GroupIdGRPOAdvantageModule",
    "compute_grpo_advantages_by_group_id",
]
