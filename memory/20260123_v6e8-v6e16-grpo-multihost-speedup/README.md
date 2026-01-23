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
  - `time/train/step_avg_last10_s`: `18.6275`
  - `throughput/train/valid_tokens_per_s`: `1879.873`
  - Notes: this corresponds to the earlier (wrong-for-this-workload) host-local mesh behavior.

- **Fixed (fast)**: `aqhfh8oo`
  - URL: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/aqhfh8oo
  - `time/train/step_avg_last10_s`: `10.0358`
  - `throughput/train/valid_tokens_per_s`: `4144.308`
  - Notes: `mesh_shape: auto` after the fix (full-device FSDP).

- **Cross-check (also fast)**: `lp716wne`
  - URL: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/lp716wne
  - `time/train/step_avg_last10_s`: `10.0366`
  - `throughput/train/valid_tokens_per_s`: `4143.038`
  - Notes: explicit `mesh_shape: 1,-1,1` (full-device FSDP).

### v6e-8 (single-host) W&B evidence (completed in this iteration)

- **Run**: `potc8br6`
  - URL: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/potc8br6
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
