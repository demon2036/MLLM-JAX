#!/usr/bin/env bash
set -euo pipefail

# Start a 1-step GRPO/GSM8K smoke run with `rollout.backend=sglang_jax` on a TPU VM.
#
# This script is intended to be executed ON the TPU VM inside a Git-synced repo
# checkout (e.g. `/root/MLLM-JAX`). It starts the training job in the background
# via `nohup` and prints the PID + log paths.
#
# Environment variables (optional overrides):
# - ENV_NAME (default: mllm-jax)
# - WANDB_MODE (default: disabled)
# - WANDB_PROJECT / WANDB_API_KEY are read via `.env` loading in the Python entrypoint.
#
# - MODEL_PATH, STEPS, ROLLOUT_BACKEND, BATCH_SIZE, NUM_PRE_Q, GLOBAL_LENGTH,
#   MAX_LENGTH_SAMPLE, PPO_EPOCHS, GRAD_ACCUM_STEPS, BETA
# - SGLANG_JAX_* knobs (see `plugins/training/rollout_backends/sglang_jax.py`)

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

mkdir -p logs

RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
LOG_FILE="logs/nohup_grpo_gsm8k_sglang_jax_smoke_${RUN_ID}.log"
EXIT_FILE="logs/nohup_grpo_gsm8k_sglang_jax_smoke_${RUN_ID}.exit"
PID_FILE="logs/nohup_grpo_gsm8k_sglang_jax_smoke_${RUN_ID}.pid"

LATEST_LOG="logs/nohup_grpo_gsm8k_sglang_jax_smoke_latest.log"
LATEST_EXIT="logs/nohup_grpo_gsm8k_sglang_jax_smoke_latest.exit"
LATEST_PID="logs/nohup_grpo_gsm8k_sglang_jax_smoke_latest.pid"

# NOTE: The `LATEST_*` files live under `logs/`, so the symlink target must be
# relative to that directory. If we point them at `logs/<file>`, they'll resolve
# as `logs/logs/<file>` when opened.
ln -sf "$(basename "$LOG_FILE")" "$LATEST_LOG"
ln -sf "$(basename "$EXIT_FILE")" "$LATEST_EXIT"
ln -sf "$(basename "$PID_FILE")" "$LATEST_PID"

rm -f "$EXIT_FILE" "$PID_FILE"
rm -f /tmp/libtpu_lockfile || true

ENV_NAME="${ENV_NAME:-mllm-jax}"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
export STEPS="${STEPS:-1}"
export ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-sglang_jax}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export NUM_PRE_Q="${NUM_PRE_Q:-4}"
export GLOBAL_LENGTH="${GLOBAL_LENGTH:-512}"
export MAX_LENGTH_SAMPLE="${MAX_LENGTH_SAMPLE:-64}"
export PPO_EPOCHS="${PPO_EPOCHS:-1}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export BETA="${BETA:-0.0}"

export SGLANG_JAX_LOG_LEVEL="${SGLANG_JAX_LOG_LEVEL:-info}"

RUNNER="set -euo pipefail; \
  rm -f /tmp/libtpu_lockfile || true; \
  if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then \
    source /root/miniconda3/etc/profile.d/conda.sh; conda activate '$ENV_NAME'; \
  fi; \
  cd '$REPO_DIR'; \
  python -u scripts/run_grpo_gsm8k_training.py; \
  echo \\$? > '$EXIT_FILE'"

nohup bash -lc "$RUNNER" >"$LOG_FILE" 2>&1 &
pid="$!"
echo "$pid" >"$PID_FILE"

echo "PID=$pid"
echo "LOG_FILE=$LOG_FILE"
echo "EXIT_FILE=$EXIT_FILE"
echo "LATEST_LOG=$LATEST_LOG"
echo "LATEST_EXIT=$LATEST_EXIT"
echo "LATEST_PID=$LATEST_PID"
