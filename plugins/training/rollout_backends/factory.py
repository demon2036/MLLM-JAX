from __future__ import annotations

from typing import Any

from plugins.training.api import RolloutSampler
from plugins.training.rollout_backends.base import RolloutBackend
from plugins.training.rollout_backends.naive_sampler import NaiveSamplerRolloutBackend
from plugins.training.rollout_backends.sglang_jax_engine import SglangJaxRolloutBackend


SUPPORTED_ROLLOUT_BACKENDS = ("naive", "sglang")


def create_rollout_backend(
    *,
    name: str,
    sampler: RolloutSampler | None = None,
    tokenizer: Any | None = None,
    model_path: str | None = None,
    mesh_shape: str | None = None,
    **_kwargs: Any,
) -> RolloutBackend:
    key = str(name).strip().lower()
    if key in {"naive", "naive_sampler", "sampler"}:
        if sampler is None:
            raise ValueError("rollout.backend='naive' requires a sampler.")
        return NaiveSamplerRolloutBackend(sampler=sampler)
    if key in {"sglang", "sglang-jax", "sglang_jax"}:
        if sampler is None:
            raise ValueError("rollout.backend='sglang' requires a sampler.")
        if model_path is None:
            raise ValueError("rollout.backend='sglang' requires model_path.")
        return SglangJaxRolloutBackend(sampler=sampler, model_path=str(model_path), mesh_shape=mesh_shape)
    raise ValueError(f"Unknown rollout.backend={name!r}. Supported backends: {SUPPORTED_ROLLOUT_BACKENDS}")
