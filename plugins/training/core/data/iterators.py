from __future__ import annotations

import random
from typing import Any, Iterable


def iter_indices(*, n: int, seed: int, shuffle: bool) -> list[int]:
    idx = list(range(int(n)))
    if not shuffle:
        return idx
    rng = random.Random(int(seed))
    rng.shuffle(idx)
    return idx


def batched(iterable: Iterable[Any], batch_size: int) -> Iterable[list[Any]]:
    batch: list[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= int(batch_size):
            yield batch
            batch = []
    if batch:
        yield batch


__all__ = ["iter_indices", "batched"]
