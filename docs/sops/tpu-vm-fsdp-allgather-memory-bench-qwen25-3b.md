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

## TPU run (commands actually used)

TPU used in this run:
- `TPU_NAME=mllm-jax-v6e-16-spot-260124155247`
- `ZONE=europe-west4-a`

### 0) Create TPU + bootstrap Python env

```bash
cd /home/john/workdir/multi-host
scripts/create_tpu_vm_retry.sh --type v6e-16 --zones europe-west4-a,us-central2-b,us-east5-b --name mllm-jax-v6e-16-spot-260124155247
scripts/bootstrap_miniconda_on_tpu_vm.sh --name mllm-jax-v6e-16-spot-260124155247 --zone europe-west4-a --worker all --env-name mllm-jax --python 3.12
scripts/sync_env_to_tpu_vm.sh --name mllm-jax-v6e-16-spot-260124155247 --zone europe-west4-a --worker all
```

### 1) Install JAX TPU wheel + clone repo branch + deps

```bash
cd /home/john/workdir/multi-host
scripts/ssh_tpu_vm_root.sh --name mllm-jax-v6e-16-spot-260124155247 --zone europe-west4-a --worker all --command \
  'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; \
   pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; \
   cd /root; if [ ! -d MLLM-JAX/.git ]; then git clone --branch john/20260124-rollout-multihost-analysis https://github.com/demon2036/MLLM-JAX.git MLLM-JAX; fi; \
   cd /root/MLLM-JAX; git fetch origin; git checkout -B john/20260124-rollout-multihost-analysis origin/john/20260124-rollout-multihost-analysis; \
   pip install -U -r requirements-tpu.txt'
```

### 2) Run the bench (all workers)

Run each case as a separate process (so buffers don’t accumulate).

```bash
cd /home/john/workdir/multi-host
scripts/ssh_tpu_vm_root.sh --name mllm-jax-v6e-16-spot-260124155247 --zone europe-west4-a --worker all --command \
  'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; export REQUIRE_MULTIHOST=1; \
   python tests/tpu_memory_fsdp_allgather_bench.py --mesh-shape auto --mode sharded --require-process-count 4'

scripts/ssh_tpu_vm_root.sh --name mllm-jax-v6e-16-spot-260124155247 --zone europe-west4-a --worker all --command \
  'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; export REQUIRE_MULTIHOST=1; \
   python tests/tpu_memory_fsdp_allgather_bench.py --mesh-shape auto --mode gather --require-process-count 4'

scripts/ssh_tpu_vm_root.sh --name mllm-jax-v6e-16-spot-260124155247 --zone europe-west4-a --worker all --command \
  'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; export REQUIRE_MULTIHOST=1; \
   python tests/tpu_memory_fsdp_allgather_bench.py --mesh-shape host_local --mode sharded --require-process-count 4'

scripts/ssh_tpu_vm_root.sh --name mllm-jax-v6e-16-spot-260124155247 --zone europe-west4-a --worker all --command \
  'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; export REQUIRE_MULTIHOST=1; \
   python tests/tpu_memory_fsdp_allgather_bench.py --mesh-shape host_local --mode gather --require-process-count 4'
```

### 3) Delete TPU (avoid billing)

```bash
cd /home/john/workdir/multi-host
scripts/delete_tpu_vm.sh --name mllm-jax-v6e-16-spot-260124155247 --zone europe-west4-a
```

## Observed result (v6e-16; bf16; ~5.6GiB buffer)

- `auto (dp=1, fsdp=16) + sharded`: `per_device_array_bytes_local_max ≈ 0.350 GiB`
- `auto + gather`: `per_device_array_bytes_local_max ≈ 5.600 GiB`
- `host_local (dp=4, fsdp=4) + sharded`: `per_device_array_bytes_local_max ≈ 1.400 GiB`
- `host_local + gather`: `per_device_array_bytes_local_max ≈ 5.600 GiB`

This matches the rule of thumb:
- sharded weights ≈ `total/fsdp` per chip
- gathered weights ≈ `total` per chip

## Troubleshooting

- TPU `process_allgather` materializes `int64` inputs as `int32` (overflow for byte counts >2^31).
  - The script uses hi/lo `int32` words to reconstruct a `uint64` on host.

## Expected Result (TPU)

- `mode=sharded` uses per-chip array bytes ≈ `total_bytes / fsdp`.
- `mode=gather` uses per-chip array bytes ≈ `total_bytes` (replicated across `fsdp`).
- Therefore, `gather` is ≈ `fsdp×` larger than `sharded` for weight buffers.
