# GRPO fused LM-head kernel (no logits/dlogits materialization)

## Goal

- Implement a fused GRPO loss that consumes LM hidden states + `lm_head` weights and:
  - computes per-token logp + GRPO clipped objective,
  - avoids materializing `[B,T,V]` logits (and ideally avoids `[B,T,V]` dlogits),
  - matches the repo reference numerics (within tolerances),
  - improves TPU peak HBM vs pure-JAX baseline at shape `(B=1,T=8192,V=*)`.

## Completion criteria

- Unit tests pass (local syntax + TPU `pytest`).
- TPU microbench A/B run exists (W&B `online`):
  - baseline (pure JAX logits + reference loss)
  - fused kernel enabled
- Reported `mem/peak_bytes_in_use_max` is lower for fused kernel.
- SOP updated with exact commands actually run.

## Evidence (to be filled as executed)

### Code changes

- Files:
  - `plugins/training/kernels/grpo_fused_lm_head.py`
  - `plugins/training/grpo/module.py`
  - `training2.py`
  - `scripts/grpo_fused_lm_head_microbench.py`
  - `plugins/training/configs/grpo_fused_lm_head_microbench_bpd1_t8192_h4096_vocab151936_baseline.yaml`
  - `plugins/training/configs/grpo_fused_lm_head_microbench_bpd1_t8192_h4096_vocab151936_fused.yaml`
  - `tests/test_grpo_fused_lm_head.py`
  - `plugins/sft/jax/train.py` (pytest fix for `device="cpu"` on TPU VMs)
  - `plugins/sft/runner/sid_sft.py` (default `rope_theta` for older HF `LlamaConfig`)

### TPU runs

- Fused microbench baseline (kernel off): https://wandb.ai/johntitordemon2036/mllm-jax-grpo-fused-lmhead-microbench/runs/br66y8np
- Fused microbench kernel on: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-fused-lmhead-microbench/runs/m9ctmroq

### Key metrics

- `mem/peak_bytes_in_use_max`:
  - baseline: `6564776448`
  - fused: `6575472640`

- `bench/step_s_mean` (optional):
  - baseline: `0.0635085138026625`
  - fused: `0.19173476110154297`

### Notes

- Correctness: TPU `pytest -q` passes (`42 passed`).
- Memory: fused LM-head kernel does **not** beat the baseline peak HBM in this microbench yet (slightly worse by ~10MB).
- Perf: fused kernel is ~3x slower in this microbench (expected due to per-vocab tiling + multiple matmuls).

### Commands actually run (TPU)

- Git sync:
  - `cd /root/MLLM-JAX && git fetch --all --prune && git checkout feat-kernel && git pull --ff-only origin feat-kernel && git rev-parse --short HEAD`
- Tests:
  - `cd /root/MLLM-JAX && /root/venvs/mllm-jax/bin/python -m pytest -q`
- Microbench baseline:
  - `cd /root/MLLM-JAX && /root/venvs/mllm-jax/bin/python -u scripts/grpo_fused_lm_head_microbench.py --config plugins/training/configs/grpo_fused_lm_head_microbench_bpd1_t8192_h4096_vocab151936_baseline.yaml`
- Microbench fused:
  - `cd /root/MLLM-JAX && /root/venvs/mllm-jax/bin/python -u scripts/grpo_fused_lm_head_microbench.py --config plugins/training/configs/grpo_fused_lm_head_microbench_bpd1_t8192_h4096_vocab151936_fused.yaml`
