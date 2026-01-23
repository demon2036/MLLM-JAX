# Task
- Fix rollout.batch_size semantics comment and align SOP with runner behavior.

# Plan
1) Update YAML comments and SOP semantics text.
2) Run local validation commands for config printing and tests.
3) Record evidence and summarize changes.

## Step 1 - Update YAML comments + SOP semantics
Completion criteria: semantics aligned in config comments/docs, print-config key clarified, missing configs added, tests updated.
Evidence:
- Updated `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml` comment semantics.
- Added `plugins/training/configs/grpo_gsm8k_default.yaml`.
- Added `plugins/training/configs/grpo_gsm8k_bs128_steps100.yaml`.
- Updated `scripts/run_grpo_gsm8k_training.py` (global prompt wording, output key).
- Updated `tests/test_grpo_training_print_config_cli.py` (expects `sequences_global_per_step`).
- Updated `docs/sops/grpo-gsm8k-runner-batch-size.md` semantics section.

## Step 2 - Run local validation
Completion criteria: print-config and tests exit 0.
Evidence:
- `python scripts/run_grpo_gsm8k_training.py --print-config` (exit 0).
- `python scripts/run_grpo_gsm8k_training.py --print-config --config plugins/training/configs/grpo_gsm8k_bs128_steps100.yaml` (exit 0).
- `python scripts/run_grpo_gsm8k_training.py --print-config --config plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml` (exit 0).
- `pytest -q` (exit 0): 15 passed.

## Step 3 - Finalize notes
Completion criteria: SOP updated and evidence captured.
Evidence:
- Updated SOP: `docs/sops/grpo-gsm8k-runner-batch-size.md` reflects global prompt semantics + legacy key deprecation.
