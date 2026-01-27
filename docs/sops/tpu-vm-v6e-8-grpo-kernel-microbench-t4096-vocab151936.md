# SOP: TPU v6e-8 GRPO logits-level microbench (B=1,T=4096,V=151936) + W&B memory_stats

- **Title**: SOP: Run `scripts/grpo_kernel_microbench.py` on a TPU `v6e-8` and log step time + `memory_stats()` to W&B (online)
  **Prereqs**: TPU VM reachable; repo synced via Git (no SCP); `/root/.env` contains `WANDB_API_KEY`; W&B project access
  **Environment (verified)**:
  - TPU VM: `minionerec-sft-subsetdiff-v6e8-euwest4a-260126160555` (`v6e-8`), zone `europe-west4-a`, external IP `35.204.214.6`
  - Python env: `/root/venvs/mllm-jax/bin/python`
  - JAX/jaxlib: `0.9.0` / `0.9.0`
  - Repo: `https://github.com/demon2036/MLLM-JAX.git`, branch `feat-kernel@0ccacd5`

## Goal

- Compare pure-JAX baseline vs GRPO kernel variants on TPU for per-device logits shape:
  - `batch_per_device=1`, `seq_len=4096`, `vocab=151936`, `logits_dtype=bfloat16`
- Log to W&B (online):
  - `bench/step_s_mean|p50|p90`
  - `mem/peak_bytes_in_use_max`, `mem/bytes_in_use_max` from `jax.local_devices()[i].memory_stats()`

## Steps (commands actually used)

### 0) Locate the TPU VM (local)

- `gcloud compute tpus tpu-vm list --zone=europe-west4-a`
- `gcloud compute tpus tpu-vm describe minionerec-sft-subsetdiff-v6e8-euwest4a-260126160555 --zone=europe-west4-a --format="json(networkEndpoints)"`

### 1) Git sync on TPU (no SCP)

- `ssh -o StrictHostKeyChecking=no -i "$env:USERPROFILE\\.ssh\\google_compute_engine" root@35.204.214.6 'set -euo pipefail; cd /root/MLLM-JAX; git fetch --all --prune; git checkout feat-kernel; git pull --ff-only origin feat-kernel; git rev-parse --short HEAD'`

### 2) Verify JAX versions on TPU (optional)

- `ssh -o StrictHostKeyChecking=no -i "$env:USERPROFILE\\.ssh\\google_compute_engine" root@35.204.214.6 'set -euo pipefail; /root/venvs/mllm-jax/bin/python -m pip show jax | sed -n "1,4p"; /root/venvs/mllm-jax/bin/python -m pip show jaxlib | sed -n "1,4p"'`

### 3) Run baseline (kernel off) microbench (W&B online)

- `ssh -o StrictHostKeyChecking=no -i "$env:USERPROFILE\\.ssh\\google_compute_engine" root@35.204.214.6 'set -euo pipefail; cd /root/MLLM-JAX; export PYTHONUNBUFFERED=1; /root/venvs/mllm-jax/bin/python -u scripts/grpo_kernel_microbench.py --config plugins/training/configs/grpo_kernel_microbench_bpd1_t4096_vocab151936_baseline.yaml'`

### 4) Run kernel variants (W&B online)

- Pallas bwd:
  - `ssh -o StrictHostKeyChecking=no -i "$env:USERPROFILE\\.ssh\\google_compute_engine" root@35.204.214.6 'set -euo pipefail; cd /root/MLLM-JAX; export PYTHONUNBUFFERED=1; /root/venvs/mllm-jax/bin/python -u scripts/grpo_kernel_microbench.py --config plugins/training/configs/grpo_kernel_microbench_bpd1_t4096_vocab151936_pallas.yaml'`
- JAX bwd:
  - `ssh -o StrictHostKeyChecking=no -i "$env:USERPROFILE\\.ssh\\google_compute_engine" root@35.204.214.6 'set -euo pipefail; cd /root/MLLM-JAX; export PYTHONUNBUFFERED=1; /root/venvs/mllm-jax/bin/python -u scripts/grpo_kernel_microbench.py --config plugins/training/configs/grpo_kernel_microbench_bpd1_t4096_vocab151936_jax_bwd.yaml'`

## Expected Result

- Each run exits `0`, prints a `metrics: {...}` dict, and W&B shows:
  - `bench/step_s_mean` and `mem/peak_bytes_in_use_max`.

## Observed Result (this verified run)

- Baseline: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-kernel-microbench/runs/wc4gfkdy
  - `bench/step_s_mean=0.00901273879717337`
  - `mem/peak_bytes_in_use_max=6226795008`
- Kernel (pallas bwd): https://wandb.ai/johntitordemon2036/mllm-jax-grpo-kernel-microbench/runs/2jesd1au
  - `bench/step_s_mean=0.05521106299856911`
  - `mem/peak_bytes_in_use_max=6226977280`
- Kernel (jax bwd): https://wandb.ai/johntitordemon2036/mllm-jax-grpo-kernel-microbench/runs/1l866e2l
  - `bench/step_s_mean=0.05293255299620796`
  - `mem/peak_bytes_in_use_max=6227687424`

## References

- Microbench script: `scripts/grpo_kernel_microbench.py`
- Kernel impls: `plugins/training/kernels/grpo_loss_pallas.py`, `plugins/training/kernels/entropy_pallas.py`
- Task log: `memory/20260127_grpo-kernel-perf-fix/README.md`

