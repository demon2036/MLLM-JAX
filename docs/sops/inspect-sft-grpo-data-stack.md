# Inspect SFT + GRPO GSM8K data stack

- **Title**: SOP: Inspect data reading, batching, collate, and sharding for SFT datasets and GRPO GSM8K
  **Prereqs**: Repo checkout; no JAX runtime required
  **Environment (verified)**:
  - OS: Linux 6.14.0-37-generic (Ubuntu)
  - Python: 3.12.2
  - JAX: not used
  - Hardware: local CPU shell (no accelerator required)

## Steps (commands actually used)

- Identify SFT dataset files:
  - `rg --files plugins/sft/datasets`
- Inspect SFT dataset utilities + tasks:
  - `sed -n '1,260p' plugins/sft/datasets/csv_utils.py`
  - `sed -n '1,260p' plugins/sft/datasets/tokenizer_utils.py`
  - `sed -n '1,260p' plugins/sft/datasets/sid_next_item.py`
  - `sed -n '1,260p' plugins/sft/datasets/sid_item_alignment.py`
  - `sed -n '1,260p' plugins/sft/datasets/eval_sid_next_item.py`
  - `sed -n '1,260p' plugins/sft/datasets/fusion_seq_rec.py`
- Inspect SFT batching/collate:
  - `sed -n '1,260p' plugins/sft/jax/data.py`
  - `sed -n '1,220p' plugins/sft/jax/train.py`
- Locate GSM8K dataset load + data flow in GRPO runner:
  - `rg -n 'load_dataset|gsm8k|dataset' plugins/training/runner/grpo_gsm8k.py`
  - `sed -n '230,330p' plugins/training/runner/grpo_gsm8k.py`
  - `sed -n '420,540p' plugins/training/runner/grpo_gsm8k.py`
  - `sed -n '600,780p' plugins/training/runner/grpo_gsm8k.py`
- Confirm entrypoint wiring:
  - `sed -n '1,260p' scripts/run_grpo_gsm8k_training.py`

## Expected Result

- SFT datasets show CSV/JSON parsing + prompt formatting + optional pretokenize; batching uses right-padding and dp/fsdp sharding.
- GRPO GSM8K uses HF `load_dataset`, process-level sharding via slicing, rollout-produced token batches padded across passes, then sharded into global arrays.

## Troubleshooting

- If `rg` is unavailable, use `grep -RIn --include='*.py' ...` instead.

## References

- `plugins/sft/datasets/csv_utils.py`
- `plugins/sft/datasets/tokenizer_utils.py`
- `plugins/sft/datasets/sid_next_item.py`
- `plugins/sft/datasets/sid_item_alignment.py`
- `plugins/sft/datasets/eval_sid_next_item.py`
- `plugins/sft/datasets/fusion_seq_rec.py`
- `plugins/sft/jax/data.py`
- `plugins/sft/jax/train.py`
- `plugins/training/runner/grpo_gsm8k.py`
- `scripts/run_grpo_gsm8k_training.py`
