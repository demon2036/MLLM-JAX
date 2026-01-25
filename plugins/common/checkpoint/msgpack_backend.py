from __future__ import annotations

import os
from typing import Any

import flax
import jax
import numpy as np


def _to_numpy_tree(tree: Any) -> Any:
    def _convert(x: Any) -> Any:
        if isinstance(x, (str, bytes, int, float, bool, type(None))):
            return x
        # JAX arrays / numpy arrays / arraylikes.
        try:
            return np.asarray(x)
        except Exception:
            return x

    return jax.tree_util.tree_map(_convert, tree)


class MsgpackCheckpointBackend:
    """Msgpack-based checkpoint backend using `flax.serialization`."""

    def save(self, path: str, payload: Any) -> None:
        payload = _to_numpy_tree(payload)
        data = flax.serialization.msgpack_serialize(payload)

        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    def load(self, path: str) -> Any:
        with open(path, "rb") as f:
            data = f.read()
        return flax.serialization.msgpack_restore(data)


__all__ = ["MsgpackCheckpointBackend"]

