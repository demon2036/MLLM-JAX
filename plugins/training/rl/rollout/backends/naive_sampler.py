from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from plugins.api.training import RolloutResult, RolloutSampler
from plugins.training.rl.rollout.sampling import generate_answers_and_training_batch


@dataclass
class NaiveSamplerRolloutBackend:
    """In-process rollout backend backed by the repo's existing Sampler API."""

    sampler: RolloutSampler
    _last_synced_params: Any | None = None

    def sync_weights(self, params: Any) -> None:
        # The naive backend uses the same in-process JAX sampler; passing `params`
        # into `rollout()` is already sufficient for on-policy rollouts. We still
        # expose this hook so the runner can treat all backends uniformly.
        self._last_synced_params = params

    def flush_cache(self) -> None:
        # No external KV/cache to release for the naive sampler.
        return

    def release_weights(self) -> None:
        # Drop the cached reference so we don't keep training buffers alive
        # across the (donated) update step.
        self._last_synced_params = None

    def rollout(
        self,
        *,
        prompts: Sequence[str],
        params: Any,
        system_prompt: str,
        global_length: int,
        max_length_sample: int,
    ) -> RolloutResult:
        if self._last_synced_params is not None:
            params = self._last_synced_params
        chat_prompts, answers, batch = generate_answers_and_training_batch(
            prompts=list(prompts),
            sampler=self.sampler,
            params=params,
            system_prompt=system_prompt,
            global_length=int(global_length),
            max_length_sample=int(max_length_sample),
        )
        return RolloutResult(chat_prompts=chat_prompts, answers=answers, batch=batch)
