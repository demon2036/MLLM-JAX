# SID SFT (Project)

This folder holds **project-scoped** configs and run notes for SID SFT.

- Project-scoped implementation lives in `projects/sid_sft/` (runner, datasets, JAX backend, metrics).
- Project wiring happens via YAML configs in `projects/sid_sft/configs/`:
  - `projects/sid_sft/configs/train/<tpu-type>/...`
  - `projects/sid_sft/configs/eval/<tpu-type>/...`
  - (legacy flat configs may still exist for backwards compatibility)

## Run (local / TPU VM)

From repo root:

- `./scripts/run_sid_sft.sh --config projects/sid_sft/configs/sid_sft_smoke_tiny_wandb_online.yaml --run-mode train_eval`
- Example (TPU v6e-8, 3-epoch train):
  - `./scripts/run_sid_sft.sh --config projects/sid_sft/configs/train/v6e-8/sid_sft_jax_qwen25_1p5b_base_industrial_v6e8_e3_adamw_train.yaml --run-mode train`
