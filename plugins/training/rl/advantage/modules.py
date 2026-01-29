from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from plugins.training.rl.advantage.grpo import compute_grpo_advantages_by_group_id
from plugins.training.rl.advantage.estimators import (
    compute_dapo_advantages_by_group_id,
    compute_global_normalized_advantages,
    compute_reinforce_plus_plus_advantages_by_group_id,
    compute_rloo_advantages_by_group_id,
)
from plugins.api.training import AdvantageResult


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
    clip_range: float | None = None

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
        if self.clip_range is not None:
            import numpy as np

            clip = float(self.clip_range)
            if clip <= 0:
                raise ValueError("clip_range must be > 0 when set")
            advantages = np.clip(advantages, -clip, clip).astype(np.float32)
        return AdvantageResult(
            advantages=advantages,
            mean_global=mean_global,
            std_global=std_global,
        )


@dataclass(frozen=True)
class GlobalNormAdvantageModule:
    """Global mean/std normalization of per-sequence rewards (REINFORCE baseline)."""

    eps: float = 1e-4
    clip_range: float | None = None

    def compute(
        self,
        *,
        rewards: Any,
        group_ids: Any,
        mean_global: float | None = None,
        std_global: float | None = None,
    ) -> AdvantageResult:
        del group_ids, mean_global, std_global
        advantages = compute_global_normalized_advantages(
            rewards=rewards,
            eps=float(self.eps),
            clip_range=self.clip_range,
        )
        return AdvantageResult(advantages=advantages)


@dataclass(frozen=True)
class RLOOAdvantageModule:
    """Leave-one-out baseline within each prompt group (RLOO)."""

    eps: float = 1e-4
    whiten: bool = True
    clip_range: float | None = None

    def compute(
        self,
        *,
        rewards: Any,
        group_ids: Any,
        mean_global: float | None = None,
        std_global: float | None = None,
    ) -> AdvantageResult:
        del mean_global, std_global
        advantages = compute_rloo_advantages_by_group_id(
            rewards=rewards,
            group_ids=group_ids,
            eps=float(self.eps),
            whiten=bool(self.whiten),
            clip_range=self.clip_range,
        )
        return AdvantageResult(advantages=advantages)


@dataclass(frozen=True)
class DAPOAdvantageModule:
    """Mixed GRPO + global baseline advantages (DAPO-style)."""

    eps: float = 1e-4
    alpha: float = 0.2
    clip_range: float | None = None

    def compute(
        self,
        *,
        rewards: Any,
        group_ids: Any,
        mean_global: float | None = None,
        std_global: float | None = None,
    ) -> AdvantageResult:
        del mean_global, std_global
        advantages = compute_dapo_advantages_by_group_id(
            rewards=rewards,
            group_ids=group_ids,
            eps=float(self.eps),
            alpha=float(self.alpha),
            clip_range=self.clip_range,
        )
        return AdvantageResult(advantages=advantages)


@dataclass(frozen=True)
class ReinforcePlusPlusAdvantageModule:
    """REINFORCE++: RLOO baseline + whitening (+ optional clipping)."""

    eps: float = 1e-4
    clip_range: float | None = None

    def compute(
        self,
        *,
        rewards: Any,
        group_ids: Any,
        mean_global: float | None = None,
        std_global: float | None = None,
    ) -> AdvantageResult:
        del mean_global, std_global
        advantages = compute_reinforce_plus_plus_advantages_by_group_id(
            rewards=rewards,
            group_ids=group_ids,
            eps=float(self.eps),
            clip_range=self.clip_range,
        )
        return AdvantageResult(advantages=advantages)


__all__ = [
    "CallableAdvantageModule",
    "DAPOAdvantageModule",
    "GlobalNormAdvantageModule",
    "GroupIdGRPOAdvantageModule",
    "ReinforcePlusPlusAdvantageModule",
    "RLOOAdvantageModule",
]
