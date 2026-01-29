from __future__ import annotations

"""Checkpoint helpers for SFT (plugin-owned)."""

import os
from typing import Any

import flax
import jax
import numpy as np


def _to_numpy_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def save_checkpoint(*, output_dir: str, state: Any, name: str = "last") -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"sft_state_{name}.msgpack")

    payload = {
        "step": int(getattr(state, "step", 0)),
        "params": _to_numpy_tree(getattr(state, "params")),
    }

    data = flax.serialization.msgpack_serialize(payload)
    with open(path, "wb") as f:
        f.write(data)
    return path


def load_checkpoint(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        data = f.read()
    obj = flax.serialization.msgpack_restore(data)
    if not isinstance(obj, dict):
        raise TypeError(f"Invalid checkpoint payload type: {type(obj).__name__}")
    return obj  # type: ignore[return-value]


__all__ = ["save_checkpoint", "load_checkpoint"]
