from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class GenerationResult:
    chat_prompts: list[str]
    answers: list[str]
    batch: dict[str, np.ndarray]


class GenerationBackend(Protocol):
    """Prompt → answers (+ training batch) generation interface.

    This is intended to be the stable extension point for adding new sampling
    engines (e.g. remote `sglang`) without modifying existing runner code.
    """

    def generate(
        self,
        *,
        prompts: list[str],
        system_prompt: str,
        global_length: int,
        max_length_sample: int,
        params: Any | None,
    ) -> GenerationResult: ...


__all__ = ["GenerationBackend", "GenerationResult"]

