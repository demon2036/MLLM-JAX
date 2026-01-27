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
  - `plugins/training/configs/grpo_fused_lm_head_microbench_bpd1_t8192_h4096_vocab151936_baseline_f32logits.yaml`
  - `plugins/training/configs/grpo_fused_lm_head_microbench_bpd1_t8192_h4096_vocab151936_fused_f32logits.yaml`
  - `tests/test_grpo_fused_lm_head.py`
  - `plugins/sft/jax/train.py` (pytest fix for `device="cpu"` on TPU VMs)
  - `plugins/sft/runner/sid_sft.py` (default `rope_theta` for older HF `LlamaConfig`)

### TPU runs

- Baseline (bf16 logits, kernel off): https://wandb.ai/johntitordemon2036/mllm-jax-grpo-fused-lmhead-microbench/runs/qzr02wl0
- Fused (bf16 logits, kernel on, bs=2048): https://wandb.ai/johntitordemon2036/mllm-jax-grpo-fused-lmhead-microbench/runs/e846csu5
- Fused (bf16 logits, kernel on, bs=1024): https://wandb.ai/johntitordemon2036/mllm-jax-grpo-fused-lmhead-microbench/runs/qbobk14f

- Baseline (fp32 logits, kernel off): https://wandb.ai/johntitordemon2036/mllm-jax-grpo-fused-lmhead-microbench/runs/vpuzizea
- Fused (fp32 logits, kernel on, `cast_logits_to_hidden_dtype=false`): https://wandb.ai/johntitordemon2036/mllm-jax-grpo-fused-lmhead-microbench/runs/kv01ry1h

### Key metrics

- `mem/peak_bytes_in_use_max`:
  - baseline (bf16 logits): `6564776448`
  - fused (bf16 logits, bs=2048): `6568435200`
  - fused (bf16 logits, bs=1024): `6568061952`

  - baseline (fp32 logits): `6564682752`
  - fused (fp32 logits): `6567677440`

- `bench/step_s_mean` (optional):
  - baseline (bf16 logits): `0.06374881879746681`
  - fused (bf16 logits, bs=2048): `0.1557564590999391`
  - fused (bf16 logits, bs=1024): `0.12882797849888447`

  - baseline (fp32 logits): `0.08782019619975472`
  - fused (fp32 logits): `0.1286411664012121`

### Notes

- Correctness: TPU `pytest -q` passes (`42 passed`).
- Memory: fused LM-head kernel still does **not** beat the baseline peak HBM in this microbench yet (best attempt is still higher by a few MB).
- Perf: fused kernel remains slower in this microbench due to per-vocab tiling (many matmuls).

### Commands actually run (TPU)

- Git sync:
  - `cd /root/MLLM-JAX && git fetch --all --prune && git checkout feat-kernel && git pull --ff-only origin feat-kernel && git rev-parse --short HEAD`
- Tests:
  - `cd /root/MLLM-JAX && /root/venvs/mllm-jax/bin/python -m pytest -q`
- Microbench baseline:
  - `cd /root/MLLM-JAX && /root/venvs/mllm-jax/bin/python -u scripts/grpo_fused_lm_head_microbench.py --config plugins/training/configs/grpo_fused_lm_head_microbench_bpd1_t8192_h4096_vocab151936_baseline.yaml`
- Microbench fused:
  - `cd /root/MLLM-JAX && /root/venvs/mllm-jax/bin/python -u scripts/grpo_fused_lm_head_microbench.py --config plugins/training/configs/grpo_fused_lm_head_microbench_bpd1_t8192_h4096_vocab151936_fused.yaml`

- Microbench baseline (fp32 logits):
  - `cd /root/MLLM-JAX && /root/venvs/mllm-jax/bin/python -u scripts/grpo_fused_lm_head_microbench.py --config plugins/training/configs/grpo_fused_lm_head_microbench_bpd1_t8192_h4096_vocab151936_baseline_f32logits.yaml`
- Microbench fused (fp32 logits):
  - `cd /root/MLLM-JAX && /root/venvs/mllm-jax/bin/python -u scripts/grpo_fused_lm_head_microbench.py --config plugins/training/configs/grpo_fused_lm_head_microbench_bpd1_t8192_h4096_vocab151936_fused_f32logits.yaml`
