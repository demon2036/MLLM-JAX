from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence


class RolloutSampler(Protocol):
    """Minimal sampler protocol needed by GRPO-style rollouts."""

    tokenizer: Any

    def find_ceil(self, length: int) -> int | None: ...

    def generate(  # noqa: D401 - signature mirrors existing sampler
        self,
        input_ids,
        attention_mask,
        position_ids,
        prefill_length: int,
        *,
        max_length: int,
        params: Any,
    ) -> Mapping[str, Any]: ...


class RewardFunction(Protocol):
    """Reward function contract used by GRPO-style runners."""

    def __call__(self, item: Mapping[str, Any], answer: str, **kwargs: Any) -> float: ...


class GRPOWorkflow(Protocol):
    """Workflow-like contract: prompts -> (answers, rollout batch)."""

    def collect(
        self,
        *,
        prompts: Sequence[str],
        sampler: RolloutSampler,
        params: Any,
        system_prompt: str,
        global_length: int,
        max_length_sample: int,
    ) -> tuple[list[str], list[str], dict[str, Any]]: ...


Batch = Mapping[str, Any]


@dataclass(frozen=True)
class RolloutResult:
    """Outputs of the rollout phase (prompt -> completion -> training batch)."""

    chat_prompts: list[str]
    answers: list[str]
    batch: dict[str, Any]


@dataclass(frozen=True)
class RewardResult:
    """Outputs of the reward phase."""

    rewards: Any
    rewards_per_func: Any | None = None


@dataclass(frozen=True)
class AdvantageResult:
    """Outputs of the advantage computation phase."""

    advantages: Any
    mean_global: float | None = None
    std_global: float | None = None


@dataclass(frozen=True)
class UpdateResult:
    """Outputs of the update phase (gradient computation + parameter update)."""

    state: Any
    batch: dict[str, Any]
    last_meta: Mapping[str, Any]
    entropy: Any | None = None


class RolloutModule(Protocol):
    """Module that produces a rollout batch (trajectory) from prompts."""

    def rollout(
        self,
        *,
        prompts: Sequence[str],
        sampler: RolloutSampler,
        params: Any,
        system_prompt: str,
        global_length: int,
        max_length_sample: int,
    ) -> RolloutResult: ...


class RewardModule(Protocol):
    """Module that computes rewards for each sampled completion."""

    def compute(
        self,
        *,
        inputs: Sequence[Mapping[str, Any]],
        answers: Sequence[str],
    ) -> RewardResult: ...


class AdvantageModule(Protocol):
    """Module that converts rewards into advantages."""

    def compute(
        self,
        *,
        rewards: Any,
        group_ids: Any,
        mean_global: float | None = None,
        std_global: float | None = None,
    ) -> AdvantageResult: ...


class UpdateModule(Protocol):
    """Module that applies optimizer updates for a batch (supports PPO/GRPO loops)."""

    def update(
        self,
        *,
        state: Any,
        batch: Batch,
        total_valid_token_count: Any,
        train_step: Callable[[Any, Any], tuple[Any, Mapping[str, Any]]],
        slice_data: Callable[[Any, int, int], Any],
        grad_accum_steps: int,
        ppo_steps: int,
    ) -> UpdateResult: ...


__all__ = [
    "AdvantageModule",
    "AdvantageResult",
    "Batch",
    "GRPOWorkflow",
    "RewardFunction",
    "RewardModule",
    "RewardResult",
    "RolloutModule",
    "RolloutResult",
    "RolloutSampler",
    "UpdateModule",
    "UpdateResult",
]

