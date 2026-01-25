from __future__ import annotations

import os
from typing import Any

from plugins.common.checkpoint.msgpack_backend import MsgpackCheckpointBackend


def save_checkpoint(*, output_dir: str, state: Any, name: str = "last") -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"sft_state_{name}.msgpack")

    payload = {
        "step": int(getattr(state, "step", 0)),
        "params": getattr(state, "params"),
    }

    MsgpackCheckpointBackend().save(path, payload)
    return path


def load_checkpoint(path: str) -> dict[str, Any]:
    obj = MsgpackCheckpointBackend().load(path)
    if not isinstance(obj, dict):
        raise TypeError(f"Invalid checkpoint payload type: {type(obj).__name__}")
    return obj  # type: ignore[return-value]


__all__ = ["save_checkpoint", "load_checkpoint"]
