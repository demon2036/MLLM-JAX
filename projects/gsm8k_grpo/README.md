# GSM8K GRPO (JAX)

This project contains the GSM8K-specific GRPO runner and its YAML configs.

## What lives here

- Entry point: `projects/gsm8k_grpo/scripts/run_train.py`
- Training loop: `projects/gsm8k_grpo/jax/train.py`
- Configs: `projects/gsm8k_grpo/configs/*.yaml`

## What does NOT live here

Reusable training components belong in `plugins/`, e.g.:

- `plugins/training/core/*` (mesh/sharding/ckpt/logging/etc.)
- `plugins/training/rl/*` (algorithms + rollout/reward/advantage/update building blocks)

