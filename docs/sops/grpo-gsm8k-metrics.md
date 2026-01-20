# GRPO/GSM8K metrics (what they mean)

- **Title**: SOP: Understand key `GRPO + GSM8K` training metrics (esp. `train/other/total_valid_token_count`)
  **Prereqs**: None
  **Scope**: `plugins/training/runner/grpo_gsm8k.py` + `MLLM_JAX/train_modules/__init__.py`

## Definitions

### `train/other/total_valid_token_count`

- Meaning: the **global count of completion tokens that actually contribute to the loss** in the current training step.
- How it’s computed (current GRPO runner):
  - The runner builds `labels` as a **0/1 mask** over the sequence length (`1` = completion token to train on; `0` = prompt/padding).
  - It then sums `labels[:, 1:]` (skip the first token due to next-token shift): `valid_tokens_local = int(labels[:, 1:].sum())`.
  - Finally it sums across all JAX processes to log a **global** count.
- Why it exists: loss is computed as a **sum over tokens**, then normalized by `total_valid_token_count` so the reported `train/other/loss` is roughly an **average per valid token**, even when completion lengths vary a lot between samples.

## How to use it

- Compare runs with different sequence lengths: `train/other/batch_global` might be the same, but `train/other/total_valid_token_count` can differ a lot (longer generations => more valid tokens => more compute).
- Interpret throughput metrics:
  - `throughput/train/valid_tokens_per_s` is essentially `train/other/total_valid_token_count / time/train/step_s`.

## Related metrics (GRPO runner)

- `train/other/batch_global`: total sequences per step across all processes.
- `train/other/batch_local`: sequences per step on the current process.
- `train/seq_len/*`: statistics of prompt/completion/total lengths per sequence (global).

## References

- `plugins/training/runner/grpo_gsm8k.py`
- `plugins/training/grpo/update.py`
- `MLLM_JAX/train_modules/__init__.py`
