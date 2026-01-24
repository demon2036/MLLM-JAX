# Memory: v6e-8 vs v6e-16 GRPO multi-host speedup

## Goal

- Explain why TPU `v6e-16` can be much slower than `v6e-8` for GRPO training in this repo.
- Deliver concrete optimizations (mesh + launch guardrails + optional code changes) that **improve v6e-16 throughput**.
- Verify by running **both** `v6e-8` and `v6e-16` with `wandb_mode=online` so results are observable.

## Key findings (updated after validation)

These findings supersede the older hypothesis text in the SOP; the SOP will be updated accordingly.

1) **`v6e-16` is multi-host and must be launched on all workers**
   - `v6e-16` TPU VM is **4 workers** (JAX `process_count=4`, `local_device_count=4`, `device_count=16`).
   - Launching only worker 0 effectively uses **4 chips**, which can make it look slower than `v6e-8` (8 chips).
   - Repo guardrails now fail-fast when `REQUIRE_MULTIHOST=1` but `jax.process_count()==1`.

2) **Mesh choice matters; for this GRPO workload full-device FSDP is faster**
   - For this decode-dominated GRPO/GSM8K benchmark (Qwen2.5-3B, K=8, len up to 1024), **full-device FSDP** (`dp=1, fsdp=device_count, tp=1`) is faster than host-local sharding (`dp=process_count, fsdp=local_device_count, tp=1`) on `v6e-16`.
   - Concretely, the previous “auto=host-local” behavior was much slower; switching `mesh_shape: auto` to resolve to `1,-1,1` (full-device FSDP) fixed the slowdown.

3) **Topology reminder: `v6e-8` in this project is single-host**
   - `v6e-8` TPU VM is **1 worker** (JAX `process_count=1`, `local_device_count=8`, `device_count=8`).
   - Host-local vs cross-host mesh discussions mainly apply to multi-host pods like `v6e-16`.

## Completion criteria

- Repo changes:
  - A safe multihost launcher exists that fails fast if a subset of workers is launched.
  - A v6e-16-friendly mesh is easy to select (config and/or `auto` mesh resolution).
- Validation:
  - `python -m pytest -q` passes locally (exit 0).
  - One `v6e-8` run completes with `wandb_mode=online`.
  - One `v6e-16` run completes with `wandb_mode=online` and shows improved step time vs the misconfigured baseline.

## Evidence log (filled as we run)

### v6e-16 (multi-host) W&B evidence (already completed)

- **Baseline (slow)**: `on2okepg`
  - URL: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/on2okepg
  - Repo git SHA: `92fe39b`
  - `time/train/step_avg_last10_s`: `18.6275`
  - `throughput/train/valid_tokens_per_s`: `1879.873`
  - Notes: this corresponds to the earlier (wrong-for-this-workload) host-local mesh behavior.

- **Fixed (fast)**: `aqhfh8oo`
  - URL: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/aqhfh8oo
  - Repo git SHA: `92fe39b`
  - `time/train/step_avg_last10_s`: `10.0358`
  - `throughput/train/valid_tokens_per_s`: `4144.308`
  - Notes: `mesh_shape: auto` after the fix (full-device FSDP).

- **Cross-check (also fast)**: `lp716wne`
  - URL: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/lp716wne
  - Repo git SHA: `92fe39b`
  - `time/train/step_avg_last10_s`: `10.0366`
  - `throughput/train/valid_tokens_per_s`: `4143.038`
  - Notes: explicit `mesh_shape: 1,-1,1` (full-device FSDP).

### v6e-8 (single-host) W&B evidence (completed in this iteration)

- **Run**: `potc8br6`
  - URL: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/potc8br6
  - Repo git SHA: `92fe39b`
  - `time/train/step_avg_last10_s`: `15.1143`
  - `throughput/train/valid_tokens_per_s`: `2293.045`
  - TPU: `mllm-jax-v6e-8-spot-260124030148` (zone `europe-west4-a`)
  - Exit code: `0` (`/root/MLLM-JAX/logs/nohup_grpo_gsm8k_qwen25_3b_bs128_steps12_v6e8_bench_20260123_192641.exit`)
  - TPU deletion: confirmed deleted (no `tpu-vm list` entries remaining in `europe-west4-a`)

#### Commands used (v6e-8 run)

Local machine (provision + bootstrap + secrets + delete):

```bash
cd /home/john/workdir/multi-host
scripts/create_tpu_vm.sh --type v6e-8 --zone europe-west4-a --name mllm-jax-v6e-8-spot-260124030148
scripts/bootstrap_miniconda_on_tpu_vm.sh --name mllm-jax-v6e-8-spot-260124030148 --zone europe-west4-a --worker all --env-name mllm-jax --python 3.12
scripts/sync_env_to_tpu_vm.sh --name mllm-jax-v6e-8-spot-260124030148 --zone europe-west4-a --worker all
scripts/delete_tpu_vm.sh --name mllm-jax-v6e-8-spot-260124030148 --zone europe-west4-a
```

TPU VM (worker 0) (Git sync + deps + launch):

```bash
cd /root
git clone --branch multi-host https://github.com/demon2036/MLLM-JAX.git MLLM-JAX

source /root/miniconda3/etc/profile.d/conda.sh
conda activate mllm-jax
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

cd /root/MLLM-JAX
pip install -U -r requirements-tpu.txt

export PRINT_TRAIN_TIME_BREAKDOWN=1
export ROLLOUT_FAST_GENERATE=1
export ROLLOUT_FAST_QWEN2_DECODE_ATTENTION=1
bash scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh --config plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps12_v6e8_bench.yaml
```

### Quick delta summary

- **v6e-16 vs v6e-8 (fixed configs)**:
  - Step time: `15.1143s` → `10.0358s` (≈ `1.51×` faster on v6e-16)
  - Throughput: `2293` → `4144` valid tokens/s (≈ `1.81×` higher on v6e-16)

## 2026-01-24: Code-only check (rollout shard_map + dp/fsdp mesh)

User question: "multi-host rollout seems slow; is rollout not using shard_map, and is it using full FSDP instead of 4dp×4fsdp?"

### Findings (from code, no TPU run)

1) **Rollout does not require `shard_map` to be sharded**
   - Rollout path in the runner is: `plugins/training/runner/grpo_gsm8k.py` → `plugins/training/rollout/backends/naive_sampler.py` → `plugins/training/rollout/sampling.py` → `sampler.generate(...)`.
   - The sampler/model compute is driven by **JAX sharding via Mesh + NamedSharding** (params loaded sharded; batch arrays constructed as global sharded arrays), not by wrapping the whole rollout in `shard_map`.
   - `shard_map` is used for the *sampling function* (top-k / greedy argmax over logits) inside the sampler, e.g. `MLLM_JAX/sample/sample_state_right_padding2.py` defines `sample_fn = shard_map(...)` and then JITs it.

2) **Yes, `mesh_shape: auto` = full-device FSDP (dp=1, fsdp=device_count, tp=1)**
   - `plugins/training/mesh.py` resolves:
     - `auto` → `get_jax_mesh2("1,-1,1")` ⇒ **dp=1, fsdp=device_count, tp=1**
     - `host_local` → **dp=process_count, fsdp=local_device_count, tp=1** (on v6e-16 this is 4×4×1)
   - Default configs in this repo generally set `mesh_shape: auto` (full-device FSDP).

3) **For this repo’s GRPO rollout-heavy workload, full-device FSDP was measured faster than 4dp×4fsdp**
   - This repo keeps `mesh_shape: host_local` as an explicit option, but documents it as the “slow reference” for the v6e-16 speed gap reproduction config.
   - If a multi-host rollout is slow, the codebase points to two common causes:
     - You are effectively running **host-local sharding** (`mesh_shape: host_local`) which reduces parameter sharding (`fsdp=4` instead of `16`) and can slow decode-heavy rollout.
     - You launched only a subset of workers, so `jax.process_count()` is lower than expected (runner prints `process=i/N`).

4) **Why host-local (4dp×4fsdp) can be slower than full-device FSDP in this repo**
   - Param partition rules for Qwen/Llama in this repo shard weights on **`fsdp` (and `tp`)**, not `dp`:
     - `MLLM_JAX/utils.py:get_partition_rules_llama()` returns `PS('fsdp','tp')` / `PS('tp','fsdp')` (no `dp`).
     - `training2.py:get_state()` applies these rules when placing params onto the mesh.
   - Therefore `host_local` (fsdp=4) makes each device’s weight shard **~4× larger** than full-device FSDP (fsdp=16), increasing per-token decode compute.
   - Batch/caches are already sharded across **dp×fsdp = device_count**, so the usual “DP scales throughput” intuition does not apply unless you also scale global batch; here it mostly trades away model sharding for locality.

### Evidence (files inspected)

- Mesh resolution: `plugins/training/mesh.py`
- Runner mesh + multi-host guardrails: `plugins/training/runner/grpo_gsm8k.py`
- Rollout backend path: `plugins/training/rollout/backends/factory.py`, `plugins/training/rollout/backends/naive_sampler.py`, `plugins/training/rollout/sampling.py`
- `shard_map` usage for sampling: `MLLM_JAX/sample/sample_state_right_padding2.py`
- Param partition rules (no `dp` sharding): `MLLM_JAX/utils.py` (`get_partition_rules_llama`), `training2.py` (`get_state`)
- Sampler/cache batch sharding (uses `mesh.axis_names`): `MLLM_JAX/utils.py` (`_form_global_array`), `MLLM_JAX/language/qwen2/configuration_qwen2.py` (`init_cache`)
- Config comments documenting `auto` vs `host_local` speed on v6e-16:
  - `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps20_v6e16_bench.yaml`
  - `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps20_v6e16_bench_badmesh.yaml`

### Local verification

- `pytest -q` → `15 passed` (exit `0`)

## 2026-01-24: MaxText sharding system (git clone + code read)

User follow-up: “dp 应该永远快过 fsdp；这个 repo 的 `fsdp` 是不是其实是 tp？请 git 下来 MaxText 研究它的切片系统。”

### Evidence (commands actually used)

```bash
cd /home/john/workdir/multi-host
mkdir -p workdir
git clone --depth 1 https://github.com/google/maxtext.git workdir/maxtext
git -C workdir/maxtext rev-parse --short HEAD
```

- MaxText revision inspected: `b646a53`

### Findings (MaxText sharding model)

1) **MaxText uses the same JAX “mesh + PartitionSpec” paradigm**
   - Mesh axis names live in config (`mesh_axes`), and sharding is expressed via `PartitionSpec` / `NamedSharding`.
   - `shard_mode: auto` uses `jax.lax.with_sharding_constraint(...)` (hint); `shard_mode: explicit` uses `jax.sharding.reshard(...)` (enforce): `workdir/maxtext/src/MaxText/sharding.py`.

2) **MaxText separates `data` (DP) vs `fsdp` (param sharding) vs `tensor` (TP)**
   - Base config defines a rich axis set: `mesh_axes: ['data', ... 'fsdp', ... 'tensor', ...]` and a large `logical_axis_rules` table:
     - `workdir/maxtext/src/MaxText/configs/base.yml`
   - Physical mesh shape comes from `ici_*_parallelism` / `dcn_*_parallelism` and is built by `create_device_mesh`:
     - `workdir/maxtext/src/MaxText/maxtext_utils.py` (`create_device_mesh`)
   - Default base.yml effectively recommends (single-slice) “put the `-1` on `fsdp`” so `fsdp` becomes `device_count`:
     - `ici_fsdp_parallelism: -1 # recommended ICI axis to be auto-sharded` (`base.yml`)

3) **Weights are typically sharded on `fsdp`/`tensor`, not on `data`**
   - Example: `Embed` param uses `sharding=("vocab", "embed")`, which is mapped to physical mesh axes via rules:
     - `workdir/maxtext/src/MaxText/layers/embeddings.py`
   - Example: simple matmul weights are sharded by logical axes (`("embed","mlp")`), then dot uses `out_sharding`:
     - `workdir/maxtext/src/MaxText/layers/simple_layer.py`

4) **MaxText only “all-gathers over fsdp” when it explicitly asks for it**
   - There is an explicit helper `all_gather_over_fsdp(...)` that removes fsdp sharding to force replication and let XLA insert all-gather:
     - `workdir/maxtext/src/MaxText/sharding.py` (`all_gather_over_fsdp`)
   - One concrete usage is vocab tiling loss (with a TODO to gather only embedding table, not all params):
     - `workdir/maxtext/src/MaxText/vocabulary_tiling.py`

### Mapping back to this repo (`plugins/training/mesh.py`)

- MaxText `data` ↔ this repo `dp`
- MaxText `fsdp` ↔ this repo `fsdp` (parameter/model sharding axis)
- MaxText `tensor` ↔ this repo `tp`

So this repo’s `fsdp` axis is **not “Torch FSDP semantics”**; it’s MaxText-style **JAX mesh axis naming** where parameters are sharded via `PartitionSpec` and compute is lowered SPMD. In decode-heavy rollout with fixed global batch, shrinking `fsdp` (e.g. host-local `4×4`) can make each device’s parameter shard larger and slower even though it “adds dp”.

## 2026-01-24: Why MaxText sometimes all-gathers (and this repo usually doesn’t)

The key difference is **not** “MaxText needs all-gather but MLLM-JAX doesn’t”. Both can run with sharded weights.
MaxText only triggers `all_gather_over_fsdp` in specific codepaths where it wants weights replicated (e.g. vocab tiling).

### Evidence: JAX SPMD matmul can be `all-reduce` (no weight all-gather)

We compiled a tiny sharded matmul on CPU with 8 virtual devices and inspected the compiled HLO:

```bash
cd /home/john/workdir/multi-host
XLA_FLAGS=--xla_force_host_platform_device_count=8 python - <<'PY'
import re
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as PS

mesh = Mesh(np.asarray(jax.devices(), dtype=object).reshape((1, 8, 1)), ("dp", "fsdp", "tp"))
x_arr = jax.device_put(jnp.ones((2, 64), jnp.float32), NamedSharding(mesh, PS()))
w_arr = jax.device_put(jnp.ones((64, 64), jnp.float32), NamedSharding(mesh, PS("fsdp", "tp")))

@jax.jit
def f(a, b):
  return a @ b

with mesh:
  compiled = f.lower(x_arr, w_arr).compile()

text = compiled.as_text()
print("all-reduce", "all-reduce" in text)
print("all-gather", "all-gather" in text)
PY
```

Observed:
- `all-reduce True`
- `all-gather False`

Interpretation:
- Keeping weights sharded can implement matmul as **partial dot + all-reduce** (summing partial results),
  which avoids replicating the full weight on every device (more memory-friendly).

### Evidence: forcing replication triggers `all-gather` (MaxText `all_gather_over_fsdp` style)

```bash
cd /home/john/workdir/multi-host
XLA_FLAGS=--xla_force_host_platform_device_count=8 python - <<'PY'
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as PS

mesh = Mesh(np.asarray(jax.devices(), dtype=object).reshape((1, 8, 1)), ("dp", "fsdp", "tp"))
replicated_w = NamedSharding(mesh, PS())
x_arr = jax.device_put(jnp.ones((2, 64), jnp.float32), NamedSharding(mesh, PS()))
w_arr = jax.device_put(jnp.ones((64, 64), jnp.float32), NamedSharding(mesh, PS("fsdp", "tp")))

@jax.jit
def f(a, b):
  b_full = jax.lax.with_sharding_constraint(b, replicated_w)
  return a @ b_full

with mesh:
  compiled = f.lower(x_arr, w_arr).compile()

text = compiled.as_text()
print("all-gather", "all-gather" in text)
print("all-reduce", "all-reduce" in text)
PY
```

Observed:
- `all-gather True`
- `all-reduce False`

Interpretation:
- This is the same mechanism MaxText documents for `all_gather_over_fsdp(...)`:
  remove `fsdp` sharding → weights become replicated → XLA inserts **all-gather**.

### Which is more memory-efficient?

- **No explicit all-gather (all-reduce strategy)** is typically more memory-efficient for weights:
  weights stay sharded (≈ `1/fsdp` per device).
- **All-gather over fsdp** increases peak memory for the gathered weights (replicated on each device).

This repo’s GRPO runner does not call a MaxText-style `all_gather_over_fsdp` on params; its weight sharding stays in effect via `PartitionSpec` rules (e.g. `MLLM_JAX/utils.py:get_partition_rules_llama`), so it tends to follow the “keep weights sharded + collectives as needed” pattern.

SOP recorded: `docs/sops/jax-spmd-allgather-vs-allreduce.md`

## 2026-01-24: TPU memory bench (Qwen2.5-3B proxy; sharded vs all-gather)

User request: “开一个 branch，在 TPU 上写简单代码验证：MaxText-style `all_gather_over_fsdp` 和 MLLM-JAX-style sharded weights 哪个更占显存？”

### Setup

- Branch: `john/20260124-rollout-multihost-analysis`
- Repo SHA on TPU: `6bcf856`
- TPU: `v6e-16` (4 workers; `process_count=4`, `local_device_count=4`, `device_count=16`)
- JAX on TPU: `0.9.0`

### Method

- Use `tests/tpu_memory_fsdp_allgather_bench.py` to allocate a bf16 buffer sized to ~Qwen2.5-3B params:
  - total bytes ≈ `5.6 GiB` (bf16 proxy)
- Compare:
  - `mesh_shape: auto` (full-device FSDP, `dp=1, fsdp=16, tp=1`)
  - `mesh_shape: host_local` (`dp=4, fsdp=4, tp=1`)
  - `mode=sharded` (keep `PS('fsdp', None)`)
  - `mode=gather` (force replication via `with_sharding_constraint(..., PS())`, MaxText `all_gather_over_fsdp` style)

### Results (per-chip)

All runs use `dtype=bf16` and allocate ~`5.600 GiB` total buffer.

- `auto + sharded` (`dp=1, fsdp=16`):
  - `per_device_array_bytes_local_max`: `0.350 GiB` (≈ `5.6/16`)
  - `device_bytes_in_use_local_max`: `0.350 GiB`
- `auto + gather`:
  - `per_device_array_bytes_local_max`: `5.600 GiB` (replicated)
  - `device_bytes_in_use_local_max`: `6.006 GiB`
- `host_local + sharded` (`dp=4, fsdp=4`):
  - `per_device_array_bytes_local_max`: `1.400 GiB` (≈ `5.6/4`)
  - `device_bytes_in_use_local_max`: `1.400 GiB`
- `host_local + gather`:
  - `per_device_array_bytes_local_max`: `5.600 GiB` (replicated)
  - `device_bytes_in_use_local_max`: `7.259 GiB`

Interpretation:

- **All-gather replication** over the `fsdp` axis increases per-chip weight memory from ≈ `total/fsdp` → `total`.
  - On `auto` (`fsdp=16`): ≈ `16×` larger (0.35 GiB → 5.6 GiB for the weight buffer).
  - On `host_local` (`fsdp=4`): ≈ `4×` larger (1.4 GiB → 5.6 GiB).
- Switching `auto` → `host_local` reduces `fsdp` from 16 → 4, so **sharded weight memory becomes ≈4× larger** per chip.

### Commands used (TPU run)

Local machine:

```bash
cd /home/john/workdir/multi-host
TPU_NAME="mllm-jax-v6e-16-spot-260124155247"
ZONE="europe-west4-a"

scripts/create_tpu_vm_retry.sh --type v6e-16 --zones europe-west4-a,us-central2-b,us-east5-b --name "$TPU_NAME"
scripts/bootstrap_miniconda_on_tpu_vm.sh --name "$TPU_NAME" --zone "$ZONE" --worker all --env-name mllm-jax --python 3.12
scripts/sync_env_to_tpu_vm.sh --name "$TPU_NAME" --zone "$ZONE" --worker all

scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --worker all --command \
  'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; \
   pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; \
   cd /root; if [ ! -d MLLM-JAX/.git ]; then git clone --branch john/20260124-rollout-multihost-analysis https://github.com/demon2036/MLLM-JAX.git MLLM-JAX; fi; \
   cd /root/MLLM-JAX; git fetch origin; git checkout -B john/20260124-rollout-multihost-analysis origin/john/20260124-rollout-multihost-analysis; \
   pip install -U -r requirements-tpu.txt'

scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --worker all --command \
  'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; export REQUIRE_MULTIHOST=1; \
   python tests/tpu_memory_fsdp_allgather_bench.py --mesh-shape auto --mode sharded --require-process-count 4'

scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --worker all --command \
  'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; export REQUIRE_MULTIHOST=1; \
   python tests/tpu_memory_fsdp_allgather_bench.py --mesh-shape auto --mode gather --require-process-count 4'

scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --worker all --command \
  'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; export REQUIRE_MULTIHOST=1; \
   python tests/tpu_memory_fsdp_allgather_bench.py --mesh-shape host_local --mode sharded --require-process-count 4'

scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --worker all --command \
  'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; export REQUIRE_MULTIHOST=1; \
   python tests/tpu_memory_fsdp_allgather_bench.py --mesh-shape host_local --mode gather --require-process-count 4'

scripts/delete_tpu_vm.sh --name "$TPU_NAME" --zone "$ZONE"
```

SOP recorded: `docs/sops/tpu-vm-fsdp-allgather-memory-bench-qwen25-3b.md`
