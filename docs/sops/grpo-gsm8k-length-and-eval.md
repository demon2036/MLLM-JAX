# GRPO/GSM8K: sequence length knobs + eval knobs

- **Title**: SOP: Understand `max_length_sample`, `global_length`, `max_length_total`, and `eval_batches` in the GRPO/GSM8K runner
  **Prereqs**: None
  **Scope**: `projects/gsm8k_grpo/jax/train.py`, `plugins/training/rl/rollout/sampling.py`, `projects/gsm8k_grpo/scripts/run_train.py`

## Sequence length knobs

### `rollout.max_length_sample`

- Meaning: **max generation length for the completion** (upper bound).
- Where it’s used: passed to `sampler.generate(..., max_length=max_length_sample, ...)`.
- Intuition: bigger => longer completions allowed => more compute, more memory pressure in rollout + update.

### `rollout.global_length`

- Meaning: **minimum padded prompt length** used for sampler “prefill bucket” selection.
- Where it’s used:
  - In `plugins/training/rl/rollout/sampling.py`, the rollout code chooses `prefill_length = sampler.find_ceil(max(global_length, prompt_len))`,
  - then right-pads the prompt up to `prefill_length` before generation.
- Intuition: it’s a *prompt-side* shape knob, not a completion length cap.

### `train.max_length_total`

- What it intends to mean (conceptually): the **training/update total sequence length cap** (prompt + completion) for model shapes.
- What it does *today* in this repo:
  - It’s still plumbed through `projects/gsm8k_grpo/scripts/run_train.py` into `training2.get_state(..., max_lengths=...)`,
  - but the current `TrainGRPOModule` implementation does **not** use `max_lengths` to truncate or mask tokens in the loss.
- Practical guidance:
  - Don’t set it in YAML configs (keep it derived/default); treat it as a legacy/compat field unless/until the training module starts using it for truncation/bucketing.

## Eval knobs

### `eval_every_steps`

- Meaning: run an eval rollout+reward pass every **N** steps (`0` disables eval).

### `eval_batches_per_process` (alias: `eval_batches`)

- Meaning: how many **prompt-batches** to evaluate each time eval triggers.
- In `projects/gsm8k_grpo/jax/train.py`, each eval “batch” does:
  - `prompts_per_pass` prompts per process (derived from global `rollout.batch_size`),
  - each prompt repeated `rollout.n` times (GRPO group size),
  - then gathers rewards across all processes.
- Total eval sequences per eval event (global):
  - `eval_batches_per_process * prompts_per_pass * rollout.n * process_count`
- Why it exists: increase it if you want a more stable eval estimate (averaged over more prompts) at the cost of more eval time.

## References

- `projects/gsm8k_grpo/jax/train.py`
- `plugins/training/rl/rollout/sampling.py`
- `projects/gsm8k_grpo/scripts/run_train.py`
- `MLLM_JAX/train_modules/__init__.py`
