# TPU VM v6e-8 GRPO/GSM8K rollout speed debug (Qwen2.5-3B)

- **Title**: SOP: Reproduce + speed up GRPO rollout time on `v6e-8` (Qwen2.5-3B) without changing `max_length_sample`
  **Prereqs**: `v6e-8` TPU VM is `READY`; conda env `mllm-jax` exists; repo synced via Git; outbound internet for HF model download
  **Environment (verified)**:
  - TPU VM: `mllm-jax-v6e-8-260120021350` (`v6e-8`, 1 host), zone `us-east1-d`, project `civil-rarity-482610-s5`
  - Python: `3.12.12`
  - JAX/JAXLIB: `0.8.2` / `0.8.2` (backend `tpu`, `device_count=8`)
  - Branch: `improve-rollout`
  - Commits used:
    - `b043fc0` (adds step time breakdown + sampler timing hooks)
    - `6a24e97` (adds Qwen2 decode attention patch)

## What we learned (root cause)

- Rollout time is dominated by **decode** (token-by-token) compute.
- Baseline decode path is expensive because:
  - `Sampler.generate` uses a Python loop for decode (host orchestration overhead).
  - Qwen2 attention fallback path (used when `q_len=1`) casts Q/K/V to `float32`, making per-token attention matmuls expensive on TPU.

## Steps (commands actually used)

### 0) Git-sync the repo on TPU (no SCP for code)

- `cd /root/MLLM-JAX`
- `git fetch --all --prune`
- `git checkout improve-rollout`
- `git reset --hard origin/improve-rollout`
- `git rev-parse --short HEAD`

### 1) Baseline rollout timing (no patches)

This keeps `rollout.max_length_sample=1024` (from `plugins/training/configs/grpo_gsm8k_bs128_steps100.yaml`).

- `rm -f /tmp/libtpu_lockfile || true`
- `source /root/miniconda3/etc/profile.d/conda.sh && conda activate mllm-jax`
- `cd /root/MLLM-JAX`
- `export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PRINT_TRAIN_TIME_BREAKDOWN=1`
- `unset ROLLOUT_FAST_GENERATE ROLLOUT_FAST_QWEN2_DECODE_ATTENTION PRINT_SAMPLER_GENERATE_TIMING`
- `python -u scripts/run_grpo_gsm8k_training.py --config plugins/training/configs/grpo_gsm8k_bs128_steps100.yaml --set steps=4 --set model_path=Qwen/Qwen2.5-3B-Instruct 2>&1 | tee /root/rollout_logs/baseline_3b_steps4.log`

### 2) Speed up rollout by removing Python decode loop (while_loop)

- `export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PRINT_TRAIN_TIME_BREAKDOWN=1`
- `export ROLLOUT_FAST_GENERATE=1`
- `unset ROLLOUT_FAST_QWEN2_DECODE_ATTENTION PRINT_SAMPLER_GENERATE_TIMING`
- `python -u scripts/run_grpo_gsm8k_training.py --config plugins/training/configs/grpo_gsm8k_bs128_steps100.yaml --set steps=4 --set model_path=Qwen/Qwen2.5-3B-Instruct 2>&1 | tee /root/rollout_logs/fast_generate_3b_steps4.log`

### 3) Speed up rollout further by patching Qwen2 decode attention dtype

- `export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PRINT_TRAIN_TIME_BREAKDOWN=1`
- `export ROLLOUT_FAST_GENERATE=1`
- `export ROLLOUT_FAST_QWEN2_DECODE_ATTENTION=1`
- `unset PRINT_SAMPLER_GENERATE_TIMING`
- `python -u scripts/run_grpo_gsm8k_training.py --config plugins/training/configs/grpo_gsm8k_bs128_steps100.yaml --set steps=4 --set model_path=Qwen/Qwen2.5-3B-Instruct 2>&1 | tee /root/rollout_logs/fast_generate_decodeattn_3b_steps4.log`

Notes:
- After attention modularization (`MLLM_JAX/language/attention.py`), Opt 2 patches the decode fallback (`_naive_sdpa` for `q_len=1`).
- Before attention modularization (older commits), Opt 2 patched `Qwen2Attention.__call__` directly.

### 4) W&B-backed timing comparison runs (20 steps; same `max_length_sample=1024`)

These are the W&B-online runs used to produce the aligned timing comparison in:
- `answers/v6e-8-grpo-rollout-speedup-qwen25-3b.md`

Notes:
- Config file used (unchanged): `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml`
- For faster iteration, `STEPS=20` was exported via env (no YAML edits).
- W&B API key is loaded from `/root/.env` by `scripts/run_grpo_gsm8k_training.py` (never commit secrets).

**Baseline (no patches)**

- W&B: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/xbeufmfo`
- Command (on TPU VM):
  - `cd /root/MLLM-JAX; export WANDB_MODE=online WANDB_PROJECT=mllm-jax-grpo-gsm8k WANDB_NAME=grpo_gsm8k_v6e8_qwen25_3b_baseline_b341e45_20260120_145313 STEPS=20 PRINT_TRAIN_TIME_BREAKDOWN=1 TOKENIZERS_PARALLELISM=false; unset ROLLOUT_FAST_GENERATE ROLLOUT_FAST_QWEN2_DECODE_ATTENTION PRINT_SAMPLER_GENERATE_TIMING; bash scripts/tpu_vm_start_grpo_gsm8k_qwen25_3b_bs128_steps100_nohup.sh`

**Opt 1 only (`ROLLOUT_FAST_GENERATE=1`)**

- W&B: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/wnlkdhdr`
- Command (on TPU VM):
  - `cd /root/MLLM-JAX; export WANDB_MODE=online WANDB_PROJECT=mllm-jax-grpo-gsm8k WANDB_NAME=grpo_gsm8k_v6e8_qwen25_3b_fastgen_b341e45_20260120_150735 STEPS=20 PRINT_TRAIN_TIME_BREAKDOWN=1 TOKENIZERS_PARALLELISM=false ROLLOUT_FAST_GENERATE=1; unset ROLLOUT_FAST_QWEN2_DECODE_ATTENTION PRINT_SAMPLER_GENERATE_TIMING; bash scripts/tpu_vm_start_grpo_gsm8k_qwen25_3b_bs128_steps100_nohup.sh`

**Opt 1 + Opt 2 (`ROLLOUT_FAST_GENERATE=1`, `ROLLOUT_FAST_QWEN2_DECODE_ATTENTION=1`)**

- W&B: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/5toqx4ph`
- Command (on TPU VM):
  - `cd /root/MLLM-JAX; export WANDB_MODE=online WANDB_PROJECT=mllm-jax-grpo-gsm8k WANDB_NAME=grpo_gsm8k_v6e8_qwen25_3b_rolloutfast_b341e45_20260120_143527 STEPS=20 PRINT_TRAIN_TIME_BREAKDOWN=1 TOKENIZERS_PARALLELISM=false ROLLOUT_FAST_GENERATE=1 ROLLOUT_FAST_QWEN2_DECODE_ATTENTION=1; unset PRINT_SAMPLER_GENERATE_TIMING; bash scripts/tpu_vm_start_grpo_gsm8k_qwen25_3b_bs128_steps100_nohup.sh`

### 5) Full 100-step W&B run (Opt 1 + Opt 2; same `max_length_sample=1024`)

- W&B: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/zy9aibuc`
- Command (on TPU VM):
  - `cd /root/MLLM-JAX; ts=$(date -u +%Y%m%d_%H%M%S); commit=$(git rev-parse --short HEAD); export WANDB_MODE=online WANDB_PROJECT=mllm-jax-grpo-gsm8k WANDB_NAME="grpo_gsm8k_v6e8_qwen25_3b_opt2_steps100_${commit}_${ts}" TOKENIZERS_PARALLELISM=false PRINT_TRAIN_TIME_BREAKDOWN=1 ROLLOUT_FAST_GENERATE=1 ROLLOUT_FAST_QWEN2_DECODE_ATTENTION=1; unset STEPS; bash scripts/tpu_vm_start_grpo_gsm8k_qwen25_3b_bs128_steps100_nohup.sh`
- Verified:
  - `logs/nohup_grpo_gsm8k_qwen25_3b_bs128_steps100_latest.exit` is `0`
  - `time/train/step_avg_last10_s = 12.890168357500807` (avg of steps 90-99)

### 6) Full 100-step W&B run (Opt 1 + Opt 2, after attention modularization)

- W&B: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/b4yhlcce`
- Command (on TPU VM):
  - `cd /root/MLLM-JAX; export WANDB_MODE=online WANDB_PROJECT=mllm-jax-grpo-gsm8k TOKENIZERS_PARALLELISM=false PRINT_TRAIN_TIME_BREAKDOWN=1 ROLLOUT_FAST_GENERATE=1 ROLLOUT_FAST_QWEN2_DECODE_ATTENTION=1; unset PRINT_SAMPLER_GENERATE_TIMING; bash scripts/tpu_vm_start_grpo_gsm8k_qwen25_3b_bs128_steps100_nohup.sh`
  - Secrets: ensure `WANDB_API_KEY` is set in the environment (never commit it).
- Verified:
  - `logs/nohup_grpo_gsm8k_qwen25_3b_bs128_steps100_latest.exit` is `0`
  - `time/train/step_avg_last10_s = 12.13907113699679` (avg of steps 90-99)

## Expected Result

- The runner prints per-step breakdown when `PRINT_TRAIN_TIME_BREAKDOWN=1`, including:
  - `rollout_generate=...s`
  - `update=...s`
  - `completion_len_max=...`
- With `ROLLOUT_FAST_GENERATE=1` + `ROLLOUT_FAST_QWEN2_DECODE_ATTENTION=1`, `rollout_generate` drops materially vs baseline at the same `max_length_sample=1024`.

## Troubleshooting

- `Unable to initialize backend 'tpu' ... already in use`:
  - `rm -f /tmp/libtpu_lockfile` and ensure no other training process is running (`pgrep -af run_grpo_gsm8k_training.py`).
- `wandb disabled due to init error: API key cannot start or end with whitespace`:
  - Ensure `WANDB_API_KEY` has no leading/trailing whitespace (common culprit: CRLF `\r` at end of the value).
- `wandb disabled due to init error: ... 401 ... user is not logged in`:
  - `WANDB_API_KEY` is invalid/expired; re-login with a valid key.
- Logs disappearing after repo clean:
  - Keep logs outside the repo dir (example used here: `/root/rollout_logs/`).

## References

- `plugins/training/runner/grpo_gsm8k.py`
- `plugins/training/rollout_optimizations/fast_sampler_generate.py`
- `plugins/training/rollout_optimizations/qwen2_decode_attention.py`
- `MLLM_JAX/sample/sample_state_right_padding2.py`
- `MLLM_JAX/language/qwen2/modular_qwen2.py`
