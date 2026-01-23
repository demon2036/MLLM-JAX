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
    """Create a JAX Mesh with portable defaults.

    Supported values:
      - "auto": full-device FSDP, topology-aware.
        - Equivalent to "1,-1,1" (dp=1, fsdp=device_count, tp=1) on both single-
          and multi-host. This maximizes parameter sharding and usually provides
          the best throughput for rollout-heavy (decode-dominated) GRPO.
      - "host_local": host-local sharding, portable across pod sizes.
        - On multi-host, this builds a host-local physical mesh:
          dp=process_count (across workers), fsdp=local_device_count (within a worker), tp=1.
        - This can be useful when you explicitly want dp across hosts, but it
          reduces the degree of parameter sharding vs full-device FSDP.
      - Any other string is forwarded to `MLLM_JAX.utils.get_jax_mesh2`.
    """

    import jax
    from jax.sharding import Mesh

    from MLLM_JAX.utils import get_jax_mesh2

    mesh_shape = str(mesh_shape or "").strip().lower()
    if mesh_shape == "":
        mesh_shape = "auto"

    if mesh_shape not in {"auto", "host_local", "host-local", "auto_host_local"}:
        return get_jax_mesh2(mesh_shape)

    # Default: use the upstream, topology-aware device mesh layout.
    if mesh_shape == "auto":
        return get_jax_mesh2("1,-1,1")

    process_count = int(jax.process_count())
    local_device_count = int(jax.local_device_count())

    if process_count <= 1:
        # Single-host: host-local is identical to full-device FSDP.
        return get_jax_mesh2("1,-1,1")

    # Multi-host: enforce a host-local physical mesh layout so `fsdp` is local
    # to a worker and `dp` spans workers.
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
