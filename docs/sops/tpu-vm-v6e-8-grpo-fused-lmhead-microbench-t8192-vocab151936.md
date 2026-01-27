# SOP: TPU v6e-8 GRPO fused LM-head microbench (B=1,T=8192,H=4096,V=151936) + W&B memory_stats

- **Title**: SOP: Run `scripts/grpo_fused_lm_head_microbench.py` on a TPU `v6e-8` and log step time + `memory_stats()` to W&B (online)
  **Prereqs**: TPU VM reachable; repo synced via Git (no SCP); `/root/.env` contains `WANDB_API_KEY`; W&B project access
  **Environment (verified)**:
  - TPU VM: `minionerec-sft-subsetdiff-v6e8-euwest4a-260126160555` (`v6e-8`), zone `europe-west4-a`, external IP `35.204.214.6`
  - Python env: `/root/venvs/mllm-jax/bin/python`
  - JAX/jaxlib: `0.9.0` / `0.9.0`
  - Repo: `https://github.com/demon2036/MLLM-JAX.git`, branch `feat-kernel@1cd2090`

## Goal

- Compare pure-JAX baseline vs fused LM-head GRPO kernel on TPU for:
  - `batch_per_device=1`, `seq_len=8192`, `hidden=4096`, `vocab=151936`, `dtype=bfloat16`
- Log to W&B (online):
  - `bench/step_s_mean|p50|p90`
  - `mem/peak_bytes_in_use_max`, `mem/bytes_in_use_max` from `jax.local_devices()[i].memory_stats()`

## Steps (commands actually used)

### 1) Git sync on TPU (no SCP)

- `ssh -o StrictHostKeyChecking=no -i "$env:USERPROFILE\\.ssh\\google_compute_engine" root@35.204.214.6 'set -euo pipefail; cd /root/MLLM-JAX; git fetch --all --prune; git checkout feat-kernel; git pull --ff-only origin feat-kernel; git rev-parse --short HEAD'`

### 2) Run baseline (kernel off) microbench (W&B online)

- `ssh -o StrictHostKeyChecking=no -i "$env:USERPROFILE\\.ssh\\google_compute_engine" root@35.204.214.6 'set -euo pipefail; cd /root/MLLM-JAX; export PYTHONUNBUFFERED=1; /root/venvs/mllm-jax/bin/python -u scripts/grpo_fused_lm_head_microbench.py --config plugins/training/configs/grpo_fused_lm_head_microbench_bpd1_t8192_h4096_vocab151936_baseline.yaml'`

### 3) Run fused kernel microbench (W&B online)

- `ssh -o StrictHostKeyChecking=no -i "$env:USERPROFILE\\.ssh\\google_compute_engine" root@35.204.214.6 'set -euo pipefail; cd /root/MLLM-JAX; export PYTHONUNBUFFERED=1; /root/venvs/mllm-jax/bin/python -u scripts/grpo_fused_lm_head_microbench.py --config plugins/training/configs/grpo_fused_lm_head_microbench_bpd1_t8192_h4096_vocab151936_fused.yaml'`

## Expected Result

- Each run exits `0`, prints a `metrics: {...}` dict, and W&B shows:
  - `bench/step_s_mean` and `mem/peak_bytes_in_use_max`.

## Observed Result (this verified run)

- Baseline: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-fused-lmhead-microbench/runs/br66y8np
  - `bench/step_s_mean=0.0635085138026625`
  - `mem/peak_bytes_in_use_max=6564776448`
- Fused: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-fused-lmhead-microbench/runs/m9ctmroq
  - `bench/step_s_mean=0.19173476110154297`
  - `mem/peak_bytes_in_use_max=6575472640`

## References

- Microbench script: `scripts/grpo_fused_lm_head_microbench.py`
- Fused kernel: `plugins/training/kernels/grpo_fused_lm_head.py`
- Task log: `memory/20260127_grpo-fused-lmhead-kernel/README.md`

