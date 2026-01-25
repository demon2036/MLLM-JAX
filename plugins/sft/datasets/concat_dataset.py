from __future__ import annotations

from bisect import bisect_right
from typing import Any, Sequence


class ConcatDataset:
    def __init__(self, datasets: Sequence[Any]):
        self._datasets = list(datasets)
        if not self._datasets:
            raise ValueError("ConcatDataset requires at least one dataset")

        cumulative_sizes: list[int] = []
        total = 0
        for ds in self._datasets:
            total += int(len(ds))
            cumulative_sizes.append(total)
        self._cumulative_sizes = cumulative_sizes

    def __len__(self) -> int:
        return self._cumulative_sizes[-1]

    def __getitem__(self, idx: int):
        if idx < 0:
            idx = len(self) + int(idx)
        if idx < 0 or idx >= len(self):
            raise IndexError("ConcatDataset index out of range")

        dataset_idx = bisect_right(self._cumulative_sizes, int(idx))
        prev = 0 if dataset_idx == 0 else self._cumulative_sizes[dataset_idx - 1]
        sample_idx = int(idx) - int(prev)
        return self._datasets[dataset_idx][sample_idx]


__all__ = ["ConcatDataset"]

