# TPU VM: FSDP all-gather memory bench (Qwen2.5-3B proxy)

- **Title**: SOP: Measure per-chip memory for sharded params vs `all_gather_over_fsdp`-style replication (Qwen2.5-3B, bf16 proxy)
  **Prereqs**: `gcloud` authenticated; TPU API enabled; repo sync via Git; Python env on TPU (`jax[tpu]` + repo deps)
  **Scope**: `tests/tpu_memory_fsdp_allgather_bench.py`

## Why this exists

When weights are sharded on an `fsdp` axis, JAX/XLA can either:

- keep weights sharded and use collectives on outputs (often `all-reduce`), or
- replicate weights (remove `fsdp` sharding) which implies an **`all-gather`** of weights.

Replicating weights is usually **much more memory expensive**. This SOP records a minimal TPU repro to quantify the
memory delta (per chip) for a Qwen2.5-3B-sized bf16 buffer.

## Local sanity check (commands actually used)

This verifies the script works without TPU (8 virtual CPU devices).

```bash
cd /home/john/workdir/multi-host
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  python tests/tpu_memory_fsdp_allgather_bench.py --mesh-shape auto --mode sharded --target-bytes-gib 0.01 --cols 1024

XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  python tests/tpu_memory_fsdp_allgather_bench.py --mesh-shape auto --mode gather --target-bytes-gib 0.01 --cols 1024
```

Expected result:
- `per_device_array_bytes_*` differs by roughly `fsdp` (here `8×`).

## TPU run (TODO: fill after running)

This section will be updated with the exact commands and outputs once the TPU run is completed.

Planned runs:
- `mesh_shape=auto`, `mode=sharded`
- `mesh_shape=auto`, `mode=gather`
- `mesh_shape=host_local`, `mode=sharded`
- `mesh_shape=host_local`, `mode=gather`

## Expected Result (TPU)

- `mode=sharded` uses per-chip array bytes ≈ `total_bytes / fsdp`.
- `mode=gather` uses per-chip array bytes ≈ `total_bytes` (replicated across `fsdp`).
- Therefore, `gather` is ≈ `fsdp×` larger than `sharded` for weight buffers.

