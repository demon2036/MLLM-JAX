from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.multihost_utils import process_allgather
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as PS

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from plugins.training.mesh import create_mesh  # noqa: E402


def _parse_dtype(value: str) -> Any:
    key = str(value or "").strip().lower()
    if key in {"bf16", "bfloat16"}:
        return jnp.bfloat16
    if key in {"f16", "fp16", "float16"}:
        return jnp.float16
    if key in {"f32", "fp32", "float32"}:
        return jnp.float32
    raise ValueError(f"Unsupported --dtype {value!r} (expected bf16/f16/f32).")


def _gib_to_bytes(gib: float) -> int:
    return int(float(gib) * (1024**3))


def _format_bytes(value: int) -> str:
    if value < 0:
        return str(value)
    gib = value / float(1024**3)
    if gib >= 0.1:
        return f"{value} ({gib:.3f} GiB)"
    mib = value / float(1024**2)
    return f"{value} ({mib:.1f} MiB)"


def _safe_memory_stats(device: Any) -> dict[str, Any] | None:
    try:
        stats = device.memory_stats()
    except Exception:
        return None
    if stats is None:
        return None
    if not isinstance(stats, dict):
        return {"raw": stats}
    return stats


def _extract_bytes(stats: dict[str, Any] | None) -> tuple[int | None, int | None]:
    if not stats:
        return None, None
    bytes_in_use = stats.get("bytes_in_use")
    peak_bytes_in_use = stats.get("peak_bytes_in_use")
    return (
        int(bytes_in_use) if bytes_in_use is not None else None,
        int(peak_bytes_in_use) if peak_bytes_in_use is not None else None,
    )


def _local_max_device_memory() -> tuple[int | None, int | None]:
    bytes_in_use_values: list[int] = []
    peak_bytes_values: list[int] = []
    for dev in jax.local_devices():
        stats = _safe_memory_stats(dev)
        bytes_in_use, peak_bytes_in_use = _extract_bytes(stats)
        if bytes_in_use is not None:
            bytes_in_use_values.append(bytes_in_use)
        if peak_bytes_in_use is not None:
            peak_bytes_values.append(peak_bytes_in_use)
    return (
        max(bytes_in_use_values) if bytes_in_use_values else None,
        max(peak_bytes_values) if peak_bytes_values else None,
    )


def _local_max_shard_bytes(arr: jax.Array) -> int:
    dtype_size = int(np.dtype(arr.dtype).itemsize)
    shard_bytes: list[int] = []
    for shard in arr.addressable_shards:
        n = int(np.prod(shard.data.shape))
        shard_bytes.append(n * dtype_size)
    return max(shard_bytes) if shard_bytes else 0


def _allgather_u64(value: int) -> np.ndarray:
    """All-gather a (potentially large) integer across processes.

    On TPU, `process_allgather` materializes `int64` inputs as `int32`, which can
    overflow for large byte counts. We instead send two `int32` words (hi/lo)
    and reconstruct an exact uint64 on host.
    """

    value_u64 = int(value) & 0xFFFFFFFFFFFFFFFF
    lo = np.asarray([value_u64 & 0xFFFFFFFF], dtype=np.uint32).view(np.int32)
    hi = np.asarray([(value_u64 >> 32) & 0xFFFFFFFF], dtype=np.uint32).view(np.int32)

    gathered_lo = process_allgather(lo)
    gathered_hi = process_allgather(hi)

    lo_u32 = np.asarray(gathered_lo, dtype=np.int32).reshape(-1).view(np.uint32)
    hi_u32 = np.asarray(gathered_hi, dtype=np.int32).reshape(-1).view(np.uint32)
    return ((hi_u32.astype(np.uint64) << np.uint64(32)) | lo_u32.astype(np.uint64)).astype(np.uint64)


def main() -> None:
    parser = argparse.ArgumentParser(description="TPU bench: sharded params vs all-gather-over-fsdp memory footprint.")
    parser.add_argument(
        "--mesh-shape",
        default="auto",
        help="Mesh shape string (default: auto). Supports auto/host_local or explicit like 1,-1,1.",
    )
    parser.add_argument(
        "--mode",
        choices=("sharded", "gather"),
        required=True,
        help="sharded: keep weights sharded on fsdp; gather: force replication (MaxText all_gather_over_fsdp style).",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        help="Array dtype (bf16/f16/f32). Default: bf16 (matches typical TPU inference/training).",
    )
    parser.add_argument(
        "--target-bytes-gib",
        type=float,
        default=5.6,
        help="Target total bytes to allocate, in GiB. Default ~Qwen2.5-3B bf16 param bytes (~5.6GiB).",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=4096,
        help="Second dimension for the allocated 2D buffer (keeps each dim <2^31).",
    )
    parser.add_argument(
        "--require-process-count",
        type=int,
        default=0,
        help="If >0, require jax.process_count() to equal this value (guards against launching only worker 0).",
    )
    args = parser.parse_args()

    require_multihost = os.environ.get("REQUIRE_MULTIHOST") == "1"

    try:
        jax.distributed.initialize()
    except Exception as e:
        if require_multihost or int(args.require_process_count) > 0:
            raise RuntimeError(
                "jax.distributed.initialize() failed but a multi-host runtime is required "
                "(REQUIRE_MULTIHOST=1 or --require-process-count is set). "
                "Launch this script on all TPU workers (`gcloud ... tpu-vm ssh --worker=all`)."
            ) from e
        if int(jax.process_index()) == 0:
            print(f"WARNING: jax.distributed.initialize() skipped: {e}")

    if require_multihost and int(jax.process_count()) <= 1:
        raise RuntimeError("Expected multi-host runtime (REQUIRE_MULTIHOST=1) but got jax.process_count()==1.")

    required = int(args.require_process_count)
    if required > 0 and int(jax.process_count()) != required:
        raise RuntimeError(f"Expected jax.process_count()=={required}, got {int(jax.process_count())}.")

    mesh = create_mesh(str(args.mesh_shape))
    fsdp = int(mesh.shape.get("fsdp", 1))
    dp = int(mesh.shape.get("dp", 1))
    tp = int(mesh.shape.get("tp", 1))

    dtype = _parse_dtype(args.dtype)
    dtype_size = int(np.dtype(dtype).itemsize)
    target_bytes = _gib_to_bytes(float(args.target_bytes_gib))
    target_elems = int(math.ceil(target_bytes / float(dtype_size)))

    cols = int(args.cols)
    if cols <= 0:
        raise ValueError("--cols must be > 0")
    rows = int(math.ceil(target_elems / float(cols)))
    # Ensure divisibility for `PS('fsdp', None)` partitioning.
    if fsdp > 1:
        rows = ((rows + fsdp - 1) // fsdp) * fsdp
    total_elems = int(rows) * int(cols)
    total_bytes = int(total_elems) * int(dtype_size)

    sharded = NamedSharding(mesh, PS("fsdp", None))
    replicated = NamedSharding(mesh, PS())

    def _alloc_zeros() -> jax.Array:
        return jnp.zeros((rows, cols), dtype=dtype)

    alloc = jax.jit(_alloc_zeros, out_shardings=sharded)

    if int(jax.process_index()) == 0:
        print(
            json.dumps(
                {
                    "jax_version": jax.__version__,
                    "backend": jax.default_backend(),
                    "process_count": int(jax.process_count()),
                    "local_device_count": int(jax.local_device_count()),
                    "device_count": int(jax.device_count()),
                    "mesh_shape": str(args.mesh_shape),
                    "mesh_axes": {"dp": dp, "fsdp": fsdp, "tp": tp},
                    "mode": str(args.mode),
                    "dtype": str(dtype),
                    "dtype_size": dtype_size,
                    "target_bytes": target_bytes,
                    "actual_total_bytes": total_bytes,
                    "actual_total_bytes_human": _format_bytes(total_bytes),
                    "buffer_shape": [rows, cols],
                },
                indent=2,
                sort_keys=True,
            )
        )

    with mesh:
        w_sharded = alloc()
        w_sharded.block_until_ready()

        if str(args.mode) == "sharded":
            w_final = w_sharded
        else:
            w_final = jax.lax.with_sharding_constraint(w_sharded, replicated)
            w_final.block_until_ready()

    local_shard_bytes = _local_max_shard_bytes(w_final)
    gathered_shard_bytes = _allgather_u64(local_shard_bytes)

    bytes_in_use, peak_bytes = _local_max_device_memory()
    gathered_bytes_in_use = _allgather_u64(bytes_in_use or 0) if bytes_in_use is not None else None
    gathered_peak_bytes = _allgather_u64(peak_bytes or 0) if peak_bytes is not None else None

    if int(jax.process_index()) == 0:
        print("per_device_array_bytes_local_max", _format_bytes(int(local_shard_bytes)))
        print("per_device_array_bytes_all_processes", [int(x) for x in gathered_shard_bytes.tolist()])
        print("per_device_array_bytes_all_processes_max", _format_bytes(int(gathered_shard_bytes.max(initial=0))))

        if bytes_in_use is None or peak_bytes is None:
            print("device_memory_stats", "unavailable (device.memory_stats() returned None)")
        else:
            print("device_bytes_in_use_local_max", _format_bytes(int(bytes_in_use)))
            print("device_peak_bytes_in_use_local_max", _format_bytes(int(peak_bytes)))
            if gathered_bytes_in_use is not None and gathered_peak_bytes is not None:
                print("device_bytes_in_use_all_processes", [int(x) for x in gathered_bytes_in_use.tolist()])
                print("device_peak_bytes_in_use_all_processes", [int(x) for x in gathered_peak_bytes.tolist()])
                print("device_bytes_in_use_all_processes_max", _format_bytes(int(gathered_bytes_in_use.max(initial=0))))
                print("device_peak_bytes_in_use_all_processes_max", _format_bytes(int(gathered_peak_bytes.max(initial=0))))


if __name__ == "__main__":
    main()
