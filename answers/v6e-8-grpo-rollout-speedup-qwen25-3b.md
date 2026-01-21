# v6e-8 GRPO rollout speed optimizations (Qwen2.5-3B)

## Goal

Observed symptom: a single GRPO training step is dominated by rollout (token-by-token decode), often **15s+**.

Constraint: **do not change config** (especially `rollout.max_length_sample`); improve via **implementation-only** changes.

This note documents:
- Root causes for the rollout bottleneck
- Two implementation optimizations (plugins-only, non-invasive)
- W&B-backed timing comparisons on `v6e-8` for `Qwen/Qwen2.5-3B-Instruct`

## Environment (verified)

- TPU VM: `v6e-8` (1 host, 8 chips), zone `us-east1-d`, project `civil-rarity-482610-s5`
- Repo branch: `improve-rollout`
- Commits:
  - `b341e45` (Runs A-D)
  - `883c018` (Run E; after attention modularization)
  - `3cf1aa5` (Run F; fp32 attention-score output on TPU)
- Python/JAX on TPU: Python `3.12.12`, `jax==0.8.2`, `jaxlib==0.8.2`

## Config (kept constant)

Using config file (unchanged):
- `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml`

Key values (as printed at runtime):
- `model_path=Qwen/Qwen2.5-3B-Instruct`
- `rollout.batch_size=128` (global sequences/step)
- `rollout.num_pre_q=8` (K)
- `rollout.global_length=512`
- `rollout.max_length_sample=1024` (**NOT modified**)
- `train.micro_batch_size=32` (runner infers `grad_accum_steps=4`)

For faster iteration in these timing experiments only:
- exported `STEPS=20` via env (no YAML edits; `max_length_sample` unchanged)

## Root cause analysis (why rollout is slow)

Rollout is dominated by **decode** (one-token-at-a-time generation):

1) **Python-side decode loop & per-token orchestration**
   - Baseline `Sampler.generate` uses a Python loop and emits a very chatty progress bar.
   - This adds host overhead (and log spam) on top of TPU compute.

2) **Qwen2 attention decode fallback runs matmuls in float32**
   - In `Qwen2Attention.__call__`, the decode path (`q_len=1`) casts Q/K/V to `float32`.
   - On TPU this makes per-token attention substantially more expensive.

## Implemented optimizations (plugins-only)

### Opt 1: `ROLLOUT_FAST_GENERATE=1` (remove Python decode loop)

- Patch target: `MLLM_JAX/sample/sample_state_right_padding2.py:Sampler.generate`
- Implementation: `plugins/training/rollout_optimizations/fast_sampler_generate.py`
- Approach: replace Python per-token loop with a `jax.lax.while_loop`-based decode (keeping the same max lengths and termination behavior).
- Expected effect: reduce rollout time by removing host orchestration overhead.

### Opt 2: `ROLLOUT_FAST_QWEN2_DECODE_ATTENTION=1` (bf16 matmuls in decode attention)

- Patch target (current): `MLLM_JAX/language/attention.py:_naive_sdpa` (decode fallback when `q_len=1`)
- Patch target (legacy, before attention modularization): `MLLM_JAX/language/qwen2/modular_qwen2.py:Qwen2Attention.__call__`
- Implementation: `plugins/training/rollout_optimizations/qwen2_decode_attention.py`
- Approach: in the `q_len=1` fallback path, keep BF16 operands but compute attention-score output in `fp32` (via `lax.dot_general(preferred_element_type=jnp.float32)`); softmax stays `fp32`.
- Expected effect: reduce per-token attention cost during decode.

## Experiments (all on W&B, v6e-8, Qwen2.5-3B)

All runs:
- `WANDB_PROJECT=mllm-jax-grpo-gsm8k`
- `WANDB_MODE=online`
- `PRINT_TRAIN_TIME_BREAKDOWN=1` (also logs timing to W&B under `time/train/*`)

### Run A: Baseline (no patches)

- W&B: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/xbeufmfo
- Env flags: none (both `ROLLOUT_FAST_GENERATE` and `ROLLOUT_FAST_QWEN2_DECODE_ATTENTION` unset)
- Summary metrics (`wandb-summary.json`):
  - `time/train/step_avg_last10_s = 20.1819`
  - `eval/reward/func/reward_correct/mean = 0.765625`

### Run B: Opt 1 only (fast generate)

- W&B: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/wnlkdhdr
- Env flags:
  - `ROLLOUT_FAST_GENERATE=1`
  - `ROLLOUT_FAST_QWEN2_DECODE_ATTENTION` unset
- Summary metrics:
  - `time/train/step_avg_last10_s = 17.0686`
  - `eval/reward/func/reward_correct/mean = 0.7421875`

### Run C: Opt 1 + Opt 2 (fast generate + decode attention bf16)

- W&B: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/5toqx4ph
- Env flags:
  - `ROLLOUT_FAST_GENERATE=1`
  - `ROLLOUT_FAST_QWEN2_DECODE_ATTENTION=1`
- Summary metrics:
  - `time/train/step_avg_last10_s = 12.1213`
  - `eval/reward/func/reward_correct/mean = 0.765625`

### Run D: Opt 1 + Opt 2 (100 steps)

This is the same Opt 1 + Opt 2 condition as Run C, but run for the full `steps=100`.

- W&B: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/zy9aibuc
- `wandb_name`: `grpo_gsm8k_v6e8_qwen25_3b_opt2_steps100_b341e45_20260121_013736`
- Env flags:
  - `ROLLOUT_FAST_GENERATE=1`
  - `ROLLOUT_FAST_QWEN2_DECODE_ATTENTION=1`
- Summary metrics (`wandb-summary.json`):
  - `time/train/step_avg_last10_s = 12.890168357500807`
  - `eval/reward/func/reward_correct/mean = 0.7421875`

### Run E: Opt 1 + Opt 2 (100 steps, after attention modularization)

This reruns the same Opt 1 + Opt 2 condition after absorbing `main`'s
"attention as a standalone module" refactor (adds `MLLM_JAX/language/attention.py`).

Implementation difference:
- Before: Opt 2 patched `Qwen2Attention.__call__` (in `modular_qwen2.py`).
- After: Opt 2 patches `MLLM_JAX/language/attention.py:_naive_sdpa` for `q_len=1`
  (decode fallback), keeping matmuls in model dtype.

- W&B: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/b4yhlcce
- `wandb_name`: `grpo_gsm8k_qwen25_3b_bs128_steps100_len1024_883c018_20260121_030815`
- Env flags:
  - `ROLLOUT_FAST_GENERATE=1`
  - `ROLLOUT_FAST_QWEN2_DECODE_ATTENTION=1`
- Summary metrics (`wandb-summary.json`):
  - `time/train/step_avg_last10_s = 12.13907113699679`
  - `eval/reward/func/reward_correct/mean = 0.78125`

### Run F: Opt 1 + Opt 2 (100 steps, fp32 attention-score output on TPU)

This is the same Opt 1 + Opt 2 condition, but with an updated Opt 2 that keeps
BF16 operands while producing FP32 attention-score output (to reduce score
rounding error) on TPU.

- W&B: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/d7cl9smf
- `wandb_name`: `grpo_gsm8k_v6e8_qwen25_3b_fp32scores_3cf1aa5_20260121_045609`
- Env flags:
  - `ROLLOUT_FAST_GENERATE=1`
  - `ROLLOUT_FAST_QWEN2_DECODE_ATTENTION=1`
- Summary metrics (`wandb-summary.json`):
  - `time/train/step_avg_last10_s = 12.67789875749877`
  - `eval/reward/func/reward_correct/mean = 0.8046875`

## Results (aligned)

### Overall step time (steady-state)

Using `time/train/step_avg_last10_s` (avg of steps 10-19):

| Condition | Step avg last10 (s) | Speedup vs baseline |
| --- | ---: | ---: |
| Baseline | 20.1819 | 1.00x |
| + Opt 1 (fast generate) | 17.0686 | 1.18x |
| + Opt 1 + Opt 2 | 12.1213 | 1.66x |

### 100-step confirmation run

- Run D (steps=100) `time/train/step_avg_last10_s` (avg of steps 90-99): `12.890168357500807` (W&B: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/zy9aibuc)
- Run E (steps=100, after attention modularization) `time/train/step_avg_last10_s` (avg of steps 90-99): `12.13907113699679` (W&B: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/b4yhlcce)
- Run F (steps=100, fp32 attention-score output on TPU) `time/train/step_avg_last10_s` (avg of steps 90-99): `12.67789875749877` (W&B: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/d7cl9smf)

### Per-step breakdown (representative `step=10`)

All 3 runs have the same update cost (~`update=3.89s`); speedup comes from rollout decode.

```
BASELINE
step=10 ... dt=23.72s ... rollout_generate=19.76s ... update=3.89s ...

FASTGEN
step=10 ... dt=19.25s ... rollout_generate=15.28s ... update=3.89s ...

FASTGEN_DECODE
step=10 ... dt=13.08s ... rollout_generate=9.12s ... update=3.89s ...
```

## Accuracy sanity notes (why it may look < 0.85)

The metric shown above (`eval/reward/func/reward_correct/mean`) is:
- computed on `eval_batches=1` and inferred `prompt_batch_size=16` (so **16 GSM8K questions**),
- with `num_pre_q=8` sampled completions per question,
- and it reports **mean correctness over all sampled completions** (not "best-of-8 per question").

Observed range in these runs is ~`0.74-0.77`.
If you expect `0.85-0.95`, the next check to add is a **per-question pass@K** metric (any-correct within each K-sample group) and/or a larger eval sample size.

## Repro (TPU VM)

On the TPU VM (repo already git-synced; `/root/.env` contains `WANDB_API_KEY`):

```bash
cd /root/MLLM-JAX
export WANDB_MODE=online
export WANDB_PROJECT=mllm-jax-grpo-gsm8k
export WANDB_NAME=<unique_name>
export TOKENIZERS_PARALLELISM=false
export PRINT_TRAIN_TIME_BREAKDOWN=1
export STEPS=20

# Baseline:
unset ROLLOUT_FAST_GENERATE ROLLOUT_FAST_QWEN2_DECODE_ATTENTION
bash scripts/tpu_vm_start_grpo_gsm8k_qwen25_3b_bs128_steps100_nohup.sh

# Opt 1:
export ROLLOUT_FAST_GENERATE=1
unset ROLLOUT_FAST_QWEN2_DECODE_ATTENTION
bash scripts/tpu_vm_start_grpo_gsm8k_qwen25_3b_bs128_steps100_nohup.sh

# Opt 1 + Opt 2:
export ROLLOUT_FAST_GENERATE=1
export ROLLOUT_FAST_QWEN2_DECODE_ATTENTION=1
bash scripts/tpu_vm_start_grpo_gsm8k_qwen25_3b_bs128_steps100_nohup.sh
```

Each launch prints `PID=...` and writes:
- `logs/nohup_<...>_latest.log`
- `logs/nohup_<...>_latest.exit` (expect `0`)
