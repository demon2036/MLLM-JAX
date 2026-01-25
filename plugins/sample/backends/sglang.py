from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from plugins.sample.backends.base import GenerationResult


@dataclass
class SglangGenerationBackend:
    """Placeholder for a future SGLang-based generation backend.

    Intentionally non-functional: this repo’s policy is add-only backends with
    no invasive edits. The concrete SGLang client can be implemented later
    without changing runner code (select via import-path component spec).
    """

    endpoint: str = ""
    model: str | None = None
    timeout_s: float = 30.0

    def generate(
        self,
        *,
        prompts: list[str],
        system_prompt: str,
        global_length: int,
        max_length_sample: int,
        params: Any | None,
    ) -> GenerationResult:
        raise NotImplementedError(
            "SglangGenerationBackend is a placeholder. "
            "Implement a concrete SGLang client and return GenerationResult."
        )


__all__ = ["SglangGenerationBackend"]

