#!/usr/bin/env bash
set -euo pipefail

# Start a 100-step GRPO/GSM8K run (Qwen2.5-3B) using an explicit YAML config
# (no training hyperparam env overrides).
#
# Config note:
# - The runner treats `rollout.batch_size` as global prompts/step. This launcher
#   targets 128 sequences/step by using `rollout.batch_size=16` and `rollout.n=8`.
#
# This script is intended to be executed ON the TPU VM inside a Git-synced repo
# checkout (e.g. `/root/MLLM-JAX`).
#
# Optional env vars:
# - ENV_NAME: conda env name (default: mllm-jax)
# - CONFIG_PATH: override YAML config path
# - PY_ARGS: extra args for `scripts/run_grpo_gsm8k_training.py` (appended)

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

ENV_NAME="${ENV_NAME:-mllm-jax}"
CONFIG_PATH="${CONFIG_PATH:-plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml}"

RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
WANDB_NAME_DEFAULT="grpo_gsm8k_qwen25_3b_bs128_n8_steps100_len1024_${COMMIT}_${RUN_TS}"

PY_ARGS="${PY_ARGS:-}"
PY_ARGS="$PY_ARGS --set wandb_name=$WANDB_NAME_DEFAULT"

export ENV_NAME
export CONFIG_PATH
export PY_ARGS

bash scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh
