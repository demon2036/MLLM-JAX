from __future__ import annotations

from typing import Any, Protocol


class CheckpointBackend(Protocol):
    def save(self, path: str, payload: Any) -> None: ...

    def load(self, path: str) -> Any: ...


__all__ = ["CheckpointBackend"]

