from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from plugins.common.component_loader import instantiate_component
from plugins.sample.backends.base import GenerationBackend, GenerationResult
from plugins.sample.backends.jax_sampler import JaxSamplerGenerationBackend
from plugins.training.api import RolloutResult


def _builtin_engines() -> dict[str, Any]:
    return {
        "jax": lambda **kwargs: JaxSamplerGenerationBackend(sampler=kwargs["sampler"]),
        "jax_sampler": lambda **kwargs: JaxSamplerGenerationBackend(sampler=kwargs["sampler"]),
        "naive": lambda **kwargs: JaxSamplerGenerationBackend(sampler=kwargs["sampler"]),
    }


@dataclass
class EngineRolloutBackend:
    """Rollout backend that delegates generation to a `GenerationBackend`.

    This enables add-only sampling backends (e.g. `sglang`) without modifying
    runner logic. Select via `rollout.backend` as an import-path component spec:

      rollout:
        backend:
          target: plugins.training.rollout.backends.engine:EngineRolloutBackend
          kwargs:
            engine:
              target: plugins.sample.backends.sglang:SglangGenerationBackend
              kwargs: {endpoint: http://...}
    """

    engine: Any = "jax_sampler"
    sampler: Any | None = None

    _engine_impl: GenerationBackend | None = None
    _last_synced_params: Any | None = None

    def __post_init__(self) -> None:
        if self._engine_impl is not None:
            return

        spec = self.engine
        if spec is None:
            spec = "jax_sampler"

        registry = _builtin_engines()
        extra_kwargs: dict[str, Any] = {}
        if self.sampler is not None:
            extra_kwargs["sampler"] = self.sampler

        # Avoid passing `sampler` into arbitrary import-path backends (they may
        # not accept it). Only inject `extra_kwargs` for builtin aliases.
        if isinstance(spec, str):
            key = spec.strip()
            if ":" not in key and "." not in key:
                self._engine_impl = instantiate_component(key, registry=registry, extra_kwargs=extra_kwargs)
                return

        if isinstance(spec, dict):
            target = str(spec.get("target") or spec.get("name") or spec.get("type") or "").strip()
            if target in registry:
                self._engine_impl = instantiate_component(spec, registry=registry, extra_kwargs=extra_kwargs)
                return

        self._engine_impl = instantiate_component(spec, registry=registry)

    def sync_weights(self, params: Any) -> None:
        self._last_synced_params = params

    def flush_cache(self) -> None:
        impl = self._engine_impl
        if impl is None:
            return
        flush = getattr(impl, "flush_cache", None)
        if callable(flush):
            flush()

    def release_weights(self) -> None:
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
        if self._engine_impl is None:
            raise RuntimeError("EngineRolloutBackend has no generation engine initialized.")
        if self._last_synced_params is not None:
            params = self._last_synced_params
        result: GenerationResult = self._engine_impl.generate(
            prompts=list(prompts),
            system_prompt=str(system_prompt),
            global_length=int(global_length),
            max_length_sample=int(max_length_sample),
            params=params,
        )
        return RolloutResult(chat_prompts=result.chat_prompts, answers=result.answers, batch=result.batch)


__all__ = ["EngineRolloutBackend"]

