from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunRequest:
    system_prompt: str
    user_prompt: str
    label: str
    k: int | None = None

