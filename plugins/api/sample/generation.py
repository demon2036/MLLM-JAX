from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class GenerationRequest:
    prompts: Sequence[str]
    max_new_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None
    extra: dict[str, Any] | None = None


@dataclass(frozen=True)
class GenerationResult:
    prompts: list[str]
    texts: list[str]
    token_ids: Any | None = None
    logprobs: Any | None = None
    extra: dict[str, Any] | None = None


__all__ = ["GenerationRequest", "GenerationResult"]

