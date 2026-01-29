from __future__ import annotations

from typing import Any

from plugins.api.training import RolloutSampler
from plugins.training.rl.rollout.backends.base import RolloutBackend
from plugins.training.rl.rollout.backends.naive_sampler import NaiveSamplerRolloutBackend


SUPPORTED_ROLLOUT_BACKENDS = ("naive",)


def create_rollout_backend(
    *,
    name: str,
    sampler: RolloutSampler | None = None,
    tokenizer: Any | None = None,
    model_path: str | None = None,
    **_kwargs: Any,
) -> RolloutBackend:
    key = str(name).strip().lower()
    if key in {"naive", "naive_sampler", "sampler"}:
        if sampler is None:
            raise ValueError("rollout.backend='naive' requires a sampler.")
        return NaiveSamplerRolloutBackend(sampler=sampler)
    raise ValueError(f"Unknown rollout.backend={name!r}. Supported backends: {SUPPORTED_ROLLOUT_BACKENDS}")
