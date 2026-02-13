from __future__ import annotations

from typing import Any

from plugins.api.training import RolloutSampler
from plugins.training.rl.rollout.backends.base import RolloutBackend
from plugins.training.rl.rollout.backends.naive_sampler import NaiveSamplerRolloutBackend
from plugins.training.rl.rollout.backends.remax_mixed_naive import ReMaxMixedNaiveRolloutBackend


SUPPORTED_ROLLOUT_BACKENDS = ("naive", "remax_mixed_naive")


def create_rollout_backend(
    *,
    name: str,
    sampler: RolloutSampler | None = None,
    tokenizer: Any | None = None,
    model_path: str | None = None,
    **_kwargs: Any,
) -> RolloutBackend:
    del tokenizer, model_path

    key = str(name).strip().lower()
    if key in {"naive", "naive_sampler", "sampler"}:
        if sampler is None:
            raise ValueError("rollout.backend='naive' requires a sampler.")
        return NaiveSamplerRolloutBackend(sampler=sampler)

    if key in {"remax_mixed_naive", "remax_mixed"}:
        if sampler is None:
            raise ValueError("rollout.backend='remax_mixed_naive' requires a sampler.")
        group_size = _kwargs.get("group_size")
        if group_size is None:
            raise ValueError("rollout.backend='remax_mixed_naive' requires group_size (rollout.n).")
        baseline_position = int(_kwargs.get("baseline_position", 0))
        return ReMaxMixedNaiveRolloutBackend(
            sampler=sampler,
            group_size=int(group_size),
            baseline_position=baseline_position,
        )

    raise ValueError(f"Unknown rollout.backend={name!r}. Supported backends: {SUPPORTED_ROLLOUT_BACKENDS}")
