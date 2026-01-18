from __future__ import annotations

from typing import Any

from plugins.training.api import RolloutSampler
from plugins.training.rollout_backends.base import RolloutBackend
from plugins.training.rollout_backends.naive_sampler import NaiveSamplerRolloutBackend


SUPPORTED_ROLLOUT_BACKENDS = ("naive",)


def create_rollout_backend(*, name: str, sampler: RolloutSampler, **_kwargs: Any) -> RolloutBackend:
    key = str(name).strip().lower()
    if key in {"naive", "naive_sampler", "sampler"}:
        return NaiveSamplerRolloutBackend(sampler=sampler)
    raise ValueError(f"Unknown rollout.backend={name!r}. Supported backends: {SUPPORTED_ROLLOUT_BACKENDS}")

