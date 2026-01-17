from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence


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
    """Reward function contract used by `jit8_train`."""

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

