from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from plugins.api.training import UpdateResult
from plugins.training.rl.update.ppo import ppo_update


@dataclass(frozen=True)
class PPOUpdateModule:
    """PPO/GRPO update loop wrapper."""

    def update(
        self,
        *,
        state: Any,
        batch: Mapping[str, Any],
        total_valid_token_count: Any,
        train_step: Callable[[Any, Any], tuple[Any, Mapping[str, Any]]],
        slice_data: Callable[[Any, int, int], Any],
        grad_accum_steps: int,
        ppo_steps: int,
    ) -> UpdateResult:
        state, batch_out, last_meta, entropy = ppo_update(
            state=state,
            datas=batch,
            total_valid_token_count=total_valid_token_count,
            train_step=train_step,
            slice_data=slice_data,
            grad_accum_steps=int(grad_accum_steps),
            ppo_steps=int(ppo_steps),
        )
        return UpdateResult(state=state, batch=batch_out, last_meta=last_meta, entropy=entropy)


@dataclass(frozen=True)
class PolicyGradientUpdateModule:
    """Policy-gradient update loop wrapper (no value head requirement)."""

    def update(
        self,
        *,
        state: Any,
        batch: Mapping[str, Any],
        total_valid_token_count: Any,
        train_step: Callable[[Any, Any], tuple[Any, Mapping[str, Any]]],
        slice_data: Callable[[Any, int, int], Any],
        grad_accum_steps: int,
        ppo_steps: int,
    ) -> UpdateResult:
        state, batch_out, last_meta, entropy = ppo_update(
            state=state,
            datas=batch,
            total_valid_token_count=total_valid_token_count,
            train_step=train_step,
            slice_data=slice_data,
            grad_accum_steps=int(grad_accum_steps),
            ppo_steps=int(ppo_steps),
        )
        return UpdateResult(state=state, batch=batch_out, last_meta=last_meta, entropy=entropy)


__all__ = ["PPOUpdateModule", "PolicyGradientUpdateModule"]
