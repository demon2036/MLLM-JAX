from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from plugins.training.advantage.grpo import compute_grpo_advantages_by_group_id
from plugins.training.api import AdvantageResult


@dataclass(frozen=True)
class CallableAdvantageModule:
    """Advantage module wrapper around a callable.

    Prefer `GroupIdGRPOAdvantageModule` for GRPO-style training to avoid
    brittle `group_size` reshapes.
    """

    fn: Callable[..., Any]

    def compute(
        self,
        *,
        rewards: Any,
        group_ids: Any,
        mean_global: float | None = None,
        std_global: float | None = None,
    ) -> AdvantageResult:
        raise NotImplementedError(
            "CallableAdvantageModule is deprecated for GRPO in this repo. "
            "Use GroupIdGRPOAdvantageModule instead."
        )


@dataclass(frozen=True)
class GroupIdGRPOAdvantageModule:
    """GRPO advantages computed within each `group_id` bucket.

    This avoids relying on a positional reshape like `rewards.reshape(-1, group_size)`,
    which is easy to misconfigure and can silently corrupt training.
    """

    eps: float = 1e-4

    def compute(
        self,
        *,
        rewards: Any,
        group_ids: Any,
        mean_global: float | None = None,
        std_global: float | None = None,
    ) -> AdvantageResult:
        advantages = compute_grpo_advantages_by_group_id(
            rewards=rewards,
            group_ids=group_ids,
            eps=float(self.eps),
        )
        return AdvantageResult(
            advantages=advantages,
            mean_global=mean_global,
            std_global=std_global,
        )


__all__ = ["CallableAdvantageModule", "GroupIdGRPOAdvantageModule"]

