from __future__ import annotations

from typing import Any

import numpy as np


def _device_process_index(device: Any) -> int:
    """Best-effort extraction of a JAX device's process index."""

    value = getattr(device, "process_index", None)
    if callable(value):
        return int(value())
    if value is None:
        # Fallback for older/alternate device objects.
        value = getattr(device, "process_id", 0)
        if callable(value):
            return int(value())
    return int(value or 0)


def _device_id(device: Any) -> int:
    value = getattr(device, "id", None)
    if callable(value):
        return int(value())
    if value is None:
        value = getattr(device, "device_id", 0)
        if callable(value):
            return int(value())
    try:
        return int(value)
    except Exception:
        return 0


def _device_sort_key(device: Any) -> tuple[int, int, str]:
    return (_device_process_index(device), _device_id(device), str(device))


def create_mesh(mesh_shape: str):
    """Create a JAX Mesh with safe multi-host defaults.

    Supported values:
      - "auto": dp across processes, fsdp within each host, tp=1.
        - On single-host, this is equivalent to "1,-1,1" using
          `mesh_utils.create_device_mesh` (keeps existing behavior).
        - On multi-host, this builds a host-local (dp,fsdp,tp) physical mesh so
          fsdp collectives do not cross hosts (critical for decode-heavy GRPO).
      - Any other string is forwarded to `MLLM_JAX.utils.get_jax_mesh2`.
    """

    import jax
    from jax.sharding import Mesh

    from MLLM_JAX.utils import get_jax_mesh2

    mesh_shape = str(mesh_shape or "").strip().lower()
    if mesh_shape == "":
        mesh_shape = "auto"

    if mesh_shape != "auto":
        return get_jax_mesh2(mesh_shape)

    process_count = int(jax.process_count())
    local_device_count = int(jax.local_device_count())

    # Single-host: preserve the historical behavior and let JAX choose a good
    # topology-aware ordering for the 1D mesh axis.
    if process_count <= 1:
        return get_jax_mesh2("1,-1,1")

    # Multi-host: enforce a host-local physical mesh layout so `fsdp` is local
    # to a worker and `dp` spans workers (avoids cross-host FSDP collectives).
    dp = process_count
    fsdp = local_device_count
    tp = 1

    devices = list(jax.devices())
    by_process: dict[int, list[Any]] = {i: [] for i in range(process_count)}
    for dev in devices:
        pi = _device_process_index(dev)
        if pi in by_process:
            by_process[pi].append(dev)

    for pi in range(process_count):
        by_process[pi].sort(key=_device_sort_key)
        if len(by_process[pi]) != local_device_count:
            raise RuntimeError(
                "Unable to build host-local mesh: expected each process to have "
                f"local_device_count={local_device_count} devices, but process {pi} has {len(by_process[pi])}. "
                "This usually indicates an unexpected JAX distributed topology."
            )

    ordered = [dev for pi in range(process_count) for dev in by_process[pi]]
    physical = np.asarray(ordered, dtype=object).reshape((dp, fsdp, tp))
    return Mesh(physical, ("dp", "fsdp", "tp"))


__all__ = ["create_mesh"]

