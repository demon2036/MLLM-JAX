from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np


@dataclass
class _Agg:
    sum: float = 0.0
    sumsq: float = 0.0
    count: int = 0
    min: float = float("inf")
    max: float = float("-inf")

    def add_values(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size == 0:
            return
        self.sum += float(values.sum())
        self.sumsq += float((values * values).sum())
        self.count += int(values.size)
        self.min = min(self.min, float(values.min()))
        self.max = max(self.max, float(values.max()))

    def add_scalar(self, value: float) -> None:
        v = float(value)
        self.sum += v
        self.sumsq += v * v
        self.count += 1
        self.min = min(self.min, v)
        self.max = max(self.max, v)


class StatsTracker:
    """Host-side metric recorder with optional multi-process reduction.

    Intended for JAX multi-host: record locally, then reduce small aggregates
    across processes (sum/sumsq/count/min/max) instead of allgathering full arrays.
    """

    def __init__(self, name: str = "") -> None:
        self._scope_stack: list[str] = [name.strip("/")] if name else []
        self._aggs: dict[str, _Agg] = {}
        self._kinds: dict[str, str] = {}  # key -> "scalar" | "summary"

    @contextmanager
    def scope(self, name: str) -> Iterator[None]:
        name = str(name).strip("/")
        if not name:
            yield
            return
        self._scope_stack.append(name)
        try:
            yield
        finally:
            self._scope_stack.pop()

    @contextmanager
    def disable_scope(self) -> Iterator[None]:
        """Temporarily record keys without any active scope prefix."""
        saved = self._scope_stack
        self._scope_stack = []
        try:
            yield
        finally:
            self._scope_stack = saved

    def _full_key(self, key: str) -> str:
        key = str(key).strip("/")
        if not self._scope_stack:
            return key
        return "/".join(self._scope_stack + [key])

    def scalar(self, **kvs: Any) -> None:
        for k, v in kvs.items():
            key = self._full_key(str(k))
            agg = self._aggs.get(key)
            if agg is None:
                agg = _Agg()
                self._aggs[key] = agg
            agg.add_scalar(float(v))
            self._kinds.setdefault(key, "scalar")

    def summary(self, key: str, values: Any) -> None:
        key_full = self._full_key(key)
        agg = self._aggs.get(key_full)
        if agg is None:
            agg = _Agg()
            self._aggs[key_full] = agg
        agg.add_values(np.asarray(values))
        self._kinds[key_full] = "summary"

    @contextmanager
    def record_timing(self, key: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - start
            # Match AReaL: timings live under a fixed `timeperf/` namespace.
            with self.disable_scope():
                self.scalar(**{f"timeperf/{str(key).strip('/')}": float(dt)})

    def export(self, *, reduce: bool = True, reset: bool = True) -> dict[str, float]:
        if reduce:
            out = self._export_reduced()
        else:
            out = self._export_from_aggs(self._aggs, self._kinds)
        if reset:
            self._aggs.clear()
            self._kinds.clear()
        return out

    def _export_from_aggs(self, aggs: dict[str, _Agg], kinds: dict[str, str]) -> dict[str, float]:
        result: dict[str, float] = {}
        for key, agg in aggs.items():
            kind = kinds.get(key, "scalar")
            if agg.count <= 0:
                continue
            mean = agg.sum / float(agg.count)
            var = max(0.0, agg.sumsq / float(agg.count) - mean * mean)
            std = math.sqrt(var)

            if kind == "summary":
                result[f"{key}/mean"] = float(mean)
                result[f"{key}/std"] = float(std)
                result[f"{key}/min"] = float(agg.min)
                result[f"{key}/max"] = float(agg.max)
                result[f"{key}__count"] = float(agg.count)
            else:
                result[key] = float(mean)
                result[key + "__count"] = float(agg.count)
        return result

    def _export_reduced(self) -> dict[str, float]:
        try:
            import jax
            import jax.numpy as jnp
            from jax.experimental.multihost_utils import process_allgather
        except Exception:
            return self._export_from_aggs(self._aggs, self._kinds)

        if int(jax.process_count()) <= 1:
            return self._export_from_aggs(self._aggs, self._kinds)

        keys = sorted(self._aggs.keys())
        if not keys:
            return {}

        # Ensure all processes share the same key set and order.
        import zlib

        key_blob = "\n".join(keys).encode()
        crc_local = np.asarray([zlib.crc32(key_blob)], dtype=np.uint32)
        crc_all = np.asarray(process_allgather(jnp.asarray(crc_local)))
        if int(jax.process_index()) == 0 and not np.all(crc_all == crc_all[0]):
            raise RuntimeError("StatsTracker keys differ across processes; cannot reduce safely.")

        # Pack aggregates into a dense matrix for a single allgather.
        # Columns: sum, sumsq, count, min, max
        mat = np.zeros((len(keys), 5), dtype=np.float64)
        for i, k in enumerate(keys):
            agg = self._aggs[k]
            mat[i, 0] = float(agg.sum)
            mat[i, 1] = float(agg.sumsq)
            mat[i, 2] = float(agg.count)
            mat[i, 3] = float(agg.min)
            mat[i, 4] = float(agg.max)

        gathered = np.asarray(process_allgather(jnp.asarray(mat)))
        # gathered: [process_count, num_keys, 5]
        sums = gathered[:, :, 0].sum(axis=0)
        sumsqs = gathered[:, :, 1].sum(axis=0)
        counts = gathered[:, :, 2].sum(axis=0)
        mins = gathered[:, :, 3].min(axis=0)
        maxs = gathered[:, :, 4].max(axis=0)

        if int(jax.process_index()) != 0:
            return {}

        reduced_aggs: dict[str, _Agg] = {}
        for i, k in enumerate(keys):
            reduced_aggs[k] = _Agg(
                sum=float(sums[i]),
                sumsq=float(sumsqs[i]),
                count=int(counts[i]),
                min=float(mins[i]),
                max=float(maxs[i]),
            )

        return self._export_from_aggs(reduced_aggs, self._kinds)


__all__ = ["StatsTracker"]
