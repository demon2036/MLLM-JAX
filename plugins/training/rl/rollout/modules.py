from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from plugins.api.training import RolloutResult, RolloutSampler
from plugins.training.rl.rollout.backends.base import RolloutBackend
from plugins.training.rl.rollout.sampling import generate_answers_and_training_batch


@dataclass(frozen=True)
class GRPOSyncRollout:
    """Synchronous rollout: prompts -> answers + batch (sampler-backed)."""

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


@dataclass
class RolloutBackendModule:
    """Adapter: treat a `RolloutBackend` as a `RolloutModule`.

    The backend owns any generation engine state (in-process sampler, external
    engine client, etc). This wrapper exists so runners can uniformly use the
    4-phase module interfaces (rollout/reward/advantage/update).

    Notes
    -----
    - `sampler` is accepted for API compatibility, but ignored because the
      backend already owns it (if needed).
    - Optional backend hooks are exposed as no-ops when unsupported.
    """

    backend: RolloutBackend

    def initialize(self) -> None:
        fn = getattr(self.backend, "initialize", None)
        if fn is not None:
            fn()

    def shutdown(self) -> None:
        fn = getattr(self.backend, "shutdown", None)
        if fn is not None:
            fn()

    def sync_weights(self, params: Any) -> None:
        fn = getattr(self.backend, "sync_weights", None)
        if fn is not None:
            fn(params)

    def flush_cache(self) -> None:
        fn = getattr(self.backend, "flush_cache", None)
        if fn is not None:
            fn()

    def release_weights(self) -> None:
        fn = getattr(self.backend, "release_weights", None)
        if fn is not None:
            fn()

    def rollout(
        self,
        *,
        prompts: Sequence[str],
        sampler: RolloutSampler,
        params: Any,
        system_prompt: str,
        global_length: int,
        max_length_sample: int,
    ) -> RolloutResult:
        return self.backend.rollout(
            prompts=prompts,
            params=params,
            system_prompt=system_prompt,
            global_length=int(global_length),
            max_length_sample=int(max_length_sample),
        )


__all__ = ["GRPOSyncRollout", "RolloutBackendModule"]
