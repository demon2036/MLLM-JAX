# GSM8K configs minimalization + YAML-only launcher policy

- **Title**: SOP: Keep only user-specified GSM8K YAMLs and enforce config-only launch behavior
  **Prereqs**: Repo branch `max-rl`; Python available for `--print-config` and pytest
  **Environment (verified)**:
  - Date: 2026-02-09
  - Branch: `max-rl`
  - Commit during validation: local working tree before commit

## Goal

1. Keep only the two user-requested configs in `projects/gsm8k_grpo/configs/`.
2. Ensure launcher runs only with `--config <yaml>` and does not inject extra runtime overrides.
3. Put OOM-oriented knobs in YAML (`micro_batch_size_per_device` + `gradient_checkpointing`).

## Kept configs

- `projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_batch128_roll8_literal_v6e8.yaml`
- `projects/gsm8k_grpo/configs/rl_gsm8k_qwen25_3b_batch128_roll8_literal_maxrl_v6e8.yaml`

All other `projects/gsm8k_grpo/configs/*.yaml` were removed.

## Implementation notes

- `run_train.py` default `--config` now points to:
  - `projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_batch128_roll8_literal_v6e8.yaml`
- Added YAML-level switch `train.gradient_checkpointing` (default true), wired through:
  - `projects/gsm8k_grpo/scripts/run_train.py`
  - `projects/gsm8k_grpo/jax/train.py`
  - `training2.py`
  - `plugins/training/rl/ppo/state.py`
  - `plugins/training/rl/ppo/module.py`
- Launcher `scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh` no longer injects rollout fast env defaults; it now only forwards `--config` to runner.

## Validation commands

```bash
python projects/gsm8k_grpo/scripts/run_train.py --print-config
python projects/gsm8k_grpo/scripts/run_train.py --print-config --config projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_batch128_roll8_literal_v6e8.yaml
python projects/gsm8k_grpo/scripts/run_train.py --print-config --config projects/gsm8k_grpo/configs/rl_gsm8k_qwen25_3b_batch128_roll8_literal_maxrl_v6e8.yaml
pytest -q tests/test_grpo_training_print_config_cli.py tests/test_rl_algorithm_factory_maxrl.py tests/test_advantage_estimators.py
```

## Expected result

- `--print-config` shows:
  - `rollout.batch_size: 128`
  - `rollout.n: 8`
  - `sequences_global_per_step: 1024`
  - `train.gradient_checkpointing: true`
- pytest subset passes.

## References

- `projects/gsm8k_grpo/configs/README.md`
- `scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh`
- `docs/sops/grpo-gsm8k-runner-batch-size.md`
