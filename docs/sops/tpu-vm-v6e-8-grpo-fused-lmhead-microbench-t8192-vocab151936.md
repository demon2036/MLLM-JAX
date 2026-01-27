# SOP: TPU v6e-8 GRPO fused LM-head microbench (B=1,T=8192,H=4096,V=151936) + W&B memory_stats

- **Title**: SOP: Run `scripts/grpo_fused_lm_head_microbench.py` on a TPU `v6e-8` and log step time + `memory_stats()` to W&B (online)
  **Prereqs**: TPU VM reachable; repo synced via Git (no SCP); `/root/.env` contains `WANDB_API_KEY`; W&B project access
  **Environment (verified)**:
  - TPU VM: `minionerec-sft-subsetdiff-v6e8-euwest4a-260126160555` (`v6e-8`), zone `europe-west4-a`, external IP `35.204.214.6`
  - Python env: `/root/venvs/mllm-jax/bin/python`
  - JAX/jaxlib: `0.9.0` / `0.9.0`
  - Repo: `https://github.com/demon2036/MLLM-JAX.git`, branch `feat-kernel@a9fe5c5`

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

### 4) Run fp32-logits baseline (kernel off) microbench (W&B online)

- `ssh -o StrictHostKeyChecking=no -i "$env:USERPROFILE\\.ssh\\google_compute_engine" root@35.204.214.6 'set -euo pipefail; cd /root/MLLM-JAX; export PYTHONUNBUFFERED=1; /root/venvs/mllm-jax/bin/python -u scripts/grpo_fused_lm_head_microbench.py --config plugins/training/configs/grpo_fused_lm_head_microbench_bpd1_t8192_h4096_vocab151936_baseline_f32logits.yaml'`

### 5) Run fp32-logits fused kernel microbench (W&B online)

- `ssh -o StrictHostKeyChecking=no -i "$env:USERPROFILE\\.ssh\\google_compute_engine" root@35.204.214.6 'set -euo pipefail; cd /root/MLLM-JAX; export PYTHONUNBUFFERED=1; /root/venvs/mllm-jax/bin/python -u scripts/grpo_fused_lm_head_microbench.py --config plugins/training/configs/grpo_fused_lm_head_microbench_bpd1_t8192_h4096_vocab151936_fused_f32logits.yaml'`

## Expected Result

- Each run exits `0`, prints a `metrics: {...}` dict, and W&B shows:
  - `bench/step_s_mean` and `mem/peak_bytes_in_use_max`.

## Observed Result (this verified run)

- Baseline (bf16 logits): https://wandb.ai/johntitordemon2036/mllm-jax-grpo-fused-lmhead-microbench/runs/qzr02wl0
  - `bench/step_s_mean=0.06374881879746681`
  - `mem/peak_bytes_in_use_max=6564776448`
- Fused (bf16 logits, bs=2048): https://wandb.ai/johntitordemon2036/mllm-jax-grpo-fused-lmhead-microbench/runs/e846csu5
  - `bench/step_s_mean=0.1557564590999391`
  - `mem/peak_bytes_in_use_max=6568435200`
- Fused (bf16 logits, bs=1024): https://wandb.ai/johntitordemon2036/mllm-jax-grpo-fused-lmhead-microbench/runs/qbobk14f
  - `bench/step_s_mean=0.12882797849888447`
  - `mem/peak_bytes_in_use_max=6568061952`

- Baseline (fp32 logits): https://wandb.ai/johntitordemon2036/mllm-jax-grpo-fused-lmhead-microbench/runs/vpuzizea
  - `bench/step_s_mean=0.08782019619975472`
  - `mem/peak_bytes_in_use_max=6564682752`
- Fused (fp32 logits, `cast_logits_to_hidden_dtype=false`): https://wandb.ai/johntitordemon2036/mllm-jax-grpo-fused-lmhead-microbench/runs/kv01ry1h
  - `bench/step_s_mean=0.1286411664012121`
  - `mem/peak_bytes_in_use_max=6567677440`

## References

- Microbench script: `scripts/grpo_fused_lm_head_microbench.py`
- Fused kernel: `plugins/training/kernels/grpo_fused_lm_head.py`
- Task log: `memory/20260127_grpo-fused-lmhead-kernel/README.md`
