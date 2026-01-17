from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from plugins.training.grpo.advantages import compute_grpo_advantages_by_group_id
from plugins.training.grpo.rewarding import compute_weighted_rewards
from plugins.training.grpo.sampling import generate_answers_and_training_batch
from plugins.training.grpo.update import ppo_update
from plugins.training.api import (
    AdvantageResult,
    RewardResult,
    RolloutResult,
    UpdateResult,
)


@dataclass(frozen=True)
class GRPOSyncRollout:
    """Synchronous GRPO rollout: prompts -> answers + batch."""

    def rollout(
        self,
        *,
        prompts: Sequence[str],
        sampler: Any,
        params: Any,
        system_prompt: str,
        global_length: int,
        max_length_sample: int,
    ) -> RolloutResult:
        chat_prompts, answers, batch = generate_answers_and_training_batch(
            prompts=list(prompts),
            sampler=sampler,
            params=params,
            system_prompt=system_prompt,
            global_length=int(global_length),
            max_length_sample=int(max_length_sample),
        )
        return RolloutResult(chat_prompts=chat_prompts, answers=answers, batch=batch)


@dataclass(frozen=True)
class WeightedRewardModule:
    """Weighted reward composition over multiple reward functions."""

    reward_funcs: Sequence[Callable[..., float]]
    reward_weights: Sequence[float]

    def compute(self, *, inputs: Sequence[Mapping[str, Any]], answers: Sequence[str]) -> RewardResult:
        rewards_per_func, rewards = compute_weighted_rewards(
            reward_funcs=self.reward_funcs,
            reward_weights=self.reward_weights,
            inputs=inputs,
            answers=answers,
        )
        return RewardResult(rewards=rewards, rewards_per_func=rewards_per_func)


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
