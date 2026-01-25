from __future__ import annotations

from typing import Any, Callable

from plugins.common.component_loader import instantiate_component, normalize_component_spec
from plugins.training.api import RolloutSampler
from plugins.training.rollout.backends.base import RolloutBackend
from plugins.training.rollout.backends.naive_sampler import NaiveSamplerRolloutBackend


SUPPORTED_ROLLOUT_BACKENDS = ("naive",)


def _builtin_registry() -> dict[str, Callable[..., RolloutBackend]]:
    return {
        "naive": lambda **kwargs: NaiveSamplerRolloutBackend(sampler=kwargs["sampler"]),
        "naive_sampler": lambda **kwargs: NaiveSamplerRolloutBackend(sampler=kwargs["sampler"]),
        "sampler": lambda **kwargs: NaiveSamplerRolloutBackend(sampler=kwargs["sampler"]),
    }


def create_rollout_backend(
    *,
    spec: Any,
    sampler: RolloutSampler | None = None,
    tokenizer: Any | None = None,
    model_path: str | None = None,
    **kwargs: Any,
) -> RolloutBackend:
    """Instantiate a rollout backend (builtin alias or import-path target).

    Supported forms:
    - Builtin: "naive"
    - Import path: "pkg.module:Symbol"
    - Dict: {"target": "...", "kwargs": {...}}
    """
    registry = _builtin_registry()
    normalized = normalize_component_spec(spec)

    # Always inject the sampler (all rollout backends should accept it).
    extra_kwargs: dict[str, Any] = dict(kwargs)
    if sampler is not None:
        extra_kwargs["sampler"] = sampler

    # Only inject optional convenience kwargs for builtin aliases. Import-path
    # targets must opt-in via their own `kwargs` to preserve add-only backends.
    if normalized.target in registry:
        if tokenizer is not None:
            extra_kwargs["tokenizer"] = tokenizer
        if model_path is not None:
            extra_kwargs["model_path"] = model_path

    backend = instantiate_component(spec, registry=registry, extra_kwargs=extra_kwargs)
    return backend
