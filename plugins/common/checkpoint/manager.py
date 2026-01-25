from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from plugins.common.component_loader import instantiate_component
from plugins.common.checkpoint.backend import CheckpointBackend
from plugins.common.checkpoint.msgpack_backend import MsgpackCheckpointBackend


def _builtin_backends() -> dict[str, Callable[..., CheckpointBackend]]:
    return {
        "msgpack": lambda **_kwargs: MsgpackCheckpointBackend(),
        "msgpack_backend": lambda **_kwargs: MsgpackCheckpointBackend(),
    }


@dataclass(frozen=True)
class CheckpointManagerConfig:
    output_dir: str
    prefix: str = "checkpoint"
    backend: Any = "msgpack"
    save_every_steps: int = 0
    save_last: bool = True
    save_total_limit: int = 1
    resume_from: str | None = None


class CheckpointManager:
    def __init__(
        self,
        cfg: CheckpointManagerConfig,
        *,
        process_index: int | None = None,
    ) -> None:
        self._cfg = cfg
        self._process_index = int(process_index) if process_index is not None else None
        self._backend: CheckpointBackend = instantiate_component(cfg.backend, registry=_builtin_backends())
        self._saved_steps: list[int] = []

    @property
    def cfg(self) -> CheckpointManagerConfig:
        return self._cfg

    def is_writer(self) -> bool:
        return self._process_index is None or int(self._process_index) == 0

    def path_for(self, name: str) -> str:
        name = str(name).strip()
        if not name:
            raise ValueError("Checkpoint name must be non-empty.")
        filename = f"{self._cfg.prefix}_{name}.msgpack"
        return os.path.join(self._cfg.output_dir, filename)

    def maybe_resume(self) -> Any | None:
        if not self._cfg.resume_from:
            return None
        return self._backend.load(str(self._cfg.resume_from))

    def save(self, *, name: str, payload: Any) -> str | None:
        if not self.is_writer():
            return None
        path = self.path_for(name)
        self._backend.save(path, payload)
        return path

    def maybe_save_step(self, *, step: int, payload: Any) -> str | None:
        every = int(self._cfg.save_every_steps)
        if every <= 0:
            return None
        step_i = int(step)
        if step_i <= 0 or (step_i % every) != 0:
            return None

        path = self.save(name=f"step{step_i}", payload=payload)
        if path is None:
            return None

        limit = int(self._cfg.save_total_limit)
        if limit > 0:
            self._saved_steps.append(step_i)
            while len(self._saved_steps) > limit:
                old_step = self._saved_steps.pop(0)
                old_path = self.path_for(f"step{int(old_step)}")
                try:
                    os.remove(old_path)
                except FileNotFoundError:
                    pass
        return path

    def maybe_save_last(self, *, payload: Any) -> str | None:
        if not bool(self._cfg.save_last):
            return None
        return self.save(name="last", payload=payload)


__all__ = ["CheckpointManager", "CheckpointManagerConfig"]

