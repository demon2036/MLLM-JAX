from __future__ import annotations

import numpy as np


def pad_2d_right(x: np.ndarray, target_len: int, pad_value: int) -> np.ndarray:
    """Right-pad a rank-2 NumPy array along axis=1 up to `target_len`."""
    if x.ndim != 2:
        raise ValueError(f"Expected rank-2 array, got shape={x.shape}")
    cur = int(x.shape[1])
    target = int(target_len)
    if cur == target:
        return x
    if cur > target:
        raise ValueError(f"Cannot pad to a smaller length: {cur} -> {target}")
    return np.pad(x, ((0, 0), (0, target - cur)), constant_values=pad_value)


__all__ = ["pad_2d_right"]

