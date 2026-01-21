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
  - `143ce86` (Run G; full test-set eval sweep)
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

### Run G: Opt 1 + Opt 2 (100 steps, full test-set eval sweep)

This run enables a **full eval split sweep** (GSM8K `test`, 1319 questions) and logs
**per-question** correctness metrics to W&B (instead of the lightweight `eval_batches=1` estimate).

- W&B: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/hfor399o
- Commit: `143ce86`
- Env flags:
  - `ROLLOUT_FAST_GENERATE=1`
  - `ROLLOUT_FAST_QWEN2_DECODE_ATTENTION=1`
  - `EVAL_FULL_SWEEP=1`
- Summary metrics (`wandb-summary.json`):
  - `time/train/step_avg_last10_s = 12.712246937098097`
  - `eval_full/accuracy/pass_at_1 = 0.800606520090978` (**treat as accuracy**: full test split, one completion per question)
  - `time/eval_full/step_s = 761.7025823409931` (~12.7 minutes)
  - `time/eval_full/rollout_generate_s = 756.6426108320011`

### Run H: Opt 1 + Opt 2 (micro_batch_size=64, 100 steps) -> OOM

This is a memory stress test (update micro-batch doubled), launched to see if
we can reduce grad accumulation steps on `v6e-8` after the rollout bf16 changes.

- W&B: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/b83xx09r
- Commit: `143ce86`
- Env flags:
  - `ROLLOUT_FAST_GENERATE=1`
  - `ROLLOUT_FAST_QWEN2_DECODE_ATTENTION=1`
  - `EVAL_FULL_SWEEP=1`
  - `TRAIN_MICRO_BATCH_SIZE=64` (runner infers `grad_accum_steps=2`)
- Result:
  - XLA OOM during `jit(training_step)`; exit code `1` written to `logs/nohup_grpo_gsm8k_qwen25_3b_bs128_steps100_latest.exit`

### Run I: Full test split accuracy (each question once)

This validates the updated full-sweep eval metric/logging (commit `b8e232c`):
- full GSM8K `test` split (1319 questions)
- `samples_per_question=1`
- logs `eval_full/accuracy` to W&B

- W&B: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/75542270
- Commit: `b8e232c`
- `steps=4` (validation run)
- Summary metrics (`wandb-summary.json`):
  - `eval_full/accuracy = 0.6990144048521607`
  - `eval_full/samples_per_question = 1`
  - `time/eval_full/step_s = 140.63900410399947`
  - `time/eval_full/rollout_generate_s = 139.90924680598255`

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
- Run G (steps=100, full test-set eval sweep) `time/train/step_avg_last10_s` (avg of steps 90-99): `12.712246937098097` (W&B: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/hfor399o)

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

## Accuracy / eval notes (full test split, each question once)

The legacy eval metric shown in Runs A-F (`eval/reward/func/reward_correct/mean`) is:
- computed on `eval_batches=1` and inferred `prompt_batch_size=16` (so **16 GSM8K questions**),
- with `num_pre_q=8` sampled completions per question,
- and it reports **mean correctness over all sampled completions** (a *mean@K* style metric).

If you want the common "GSM8K accuracy" style number (full test split, **one completion per question**):
- enable `EVAL_FULL_SWEEP=1` to run the **entire eval split once** at the end of training,
- read the logged metric under:
  - `eval_full/accuracy` (post `b8e232c`; `eval_full/accuracy/pass_at_1` is kept as an alias)

Example (Run G, full `test` set, 1319 questions):
- `accuracy(pass@1) = 0.8006`

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
export EVAL_FULL_SWEEP=1  # full `eval_split` sweep once at end (adds ~10–15 min)

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
