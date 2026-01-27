# SID SFT (Project)

This folder holds **project-scoped** configs and run notes for SID SFT.

- Reusable implementation lives in `plugins/sft/` (runner, datasets, JAX backend, metrics).
- Project wiring happens via YAML configs in `projects/sid_sft/configs/`.

## Run (local / TPU VM)

From repo root:

- `./scripts/run_sid_sft.sh --config projects/sid_sft/configs/sid_sft_smoke_tiny_wandb_online.yaml --run-mode train_eval`

