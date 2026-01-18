from __future__ import annotations

from typing import Any

from plugins.training.api import RolloutSampler
from plugins.training.rollout_backends.base import RolloutBackend
from plugins.training.rollout_backends.naive_sampler import NaiveSamplerRolloutBackend
from plugins.training.rollout_backends.sglang_jax import SglangJaxRolloutBackend


SUPPORTED_ROLLOUT_BACKENDS = ("naive", "sglang_jax")


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
    if key in {"sglang_jax", "sglang-jax", "sglang"}:
        if model_path is None or str(model_path).strip() == "":
            raise ValueError("rollout.backend='sglang_jax' requires model_path (pass `cfg.model_path` from the runner).")
        if tokenizer is None:
            if sampler is None:
                raise ValueError("rollout.backend='sglang_jax' requires tokenizer (or a sampler with `.tokenizer`).")
            tokenizer = sampler.tokenizer
        return SglangJaxRolloutBackend(tokenizer=tokenizer, model_path=str(model_path))
    raise ValueError(f"Unknown rollout.backend={name!r}. Supported backends: {SUPPORTED_ROLLOUT_BACKENDS}")
