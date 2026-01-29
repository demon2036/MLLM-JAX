from __future__ import annotations

from typing import Any, Protocol

from plugins.api.sample.generation import GenerationRequest, GenerationResult


class GenerationBackend(Protocol):
    """Backend-agnostic generation interface (no RL/SFT semantics)."""

    def generate(self, request: GenerationRequest, *, params: Any | None = None) -> GenerationResult: ...


__all__ = ["GenerationBackend"]

