from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from plugins.training.api import RolloutResult, RolloutSampler
from plugins.training.grpo.sampling import generate_answers_and_training_batch


@dataclass(frozen=True)
class NaiveSamplerRolloutBackend:
    """In-process rollout backend backed by the repo's existing Sampler API."""

    sampler: RolloutSampler

    def rollout(
        self,
        *,
        prompts: Sequence[str],
        params: Any,
        system_prompt: str,
        global_length: int,
        max_length_sample: int,
    ) -> RolloutResult:
        chat_prompts, answers, batch = generate_answers_and_training_batch(
            prompts=list(prompts),
            sampler=self.sampler,
            params=params,
            system_prompt=system_prompt,
            global_length=int(global_length),
            max_length_sample=int(max_length_sample),
        )
        return RolloutResult(chat_prompts=chat_prompts, answers=answers, batch=batch)

