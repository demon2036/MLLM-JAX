#!/usr/bin/env bash
set -euo pipefail

# Start GRPO/GSM8K training from a YAML config via `nohup` on a TPU VM.
#
# This script is intended to be executed ON the TPU VM inside a Git-synced repo
# checkout (e.g. `/root/MLLM-JAX`). It starts the training job in the background
# via `nohup` and prints the PID + log paths.
#
# Environment variables (optional overrides):
# - ENV_NAME (default: mllm-jax)
# - CONFIG_PATH (default: plugins/training/configs/grpo_gsm8k_default.yaml)
# - WANDB_MODE / WANDB_PROJECT / WANDB_NAME (only exported when non-empty)
# - PY_ARGS: extra CLI args for `scripts/run_grpo_gsm8k_training.py` (string)

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

mkdir -p logs

ENV_NAME="${ENV_NAME:-mllm-jax}"
CONFIG_PATH="${CONFIG_PATH:-plugins/training/configs/grpo_gsm8k_default.yaml}"
PY_ARGS="${PY_ARGS:-}"

RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
CONFIG_TAG="$(basename "$CONFIG_PATH")"
CONFIG_TAG="${CONFIG_TAG%.yaml}"

LOG_FILE="logs/nohup_${CONFIG_TAG}_${RUN_ID}.log"
EXIT_FILE="logs/nohup_${CONFIG_TAG}_${RUN_ID}.exit"
PID_FILE="logs/nohup_${CONFIG_TAG}_${RUN_ID}.pid"

LATEST_LOG="logs/nohup_${CONFIG_TAG}_latest.log"
LATEST_EXIT="logs/nohup_${CONFIG_TAG}_latest.exit"
LATEST_PID="logs/nohup_${CONFIG_TAG}_latest.pid"

ln -sf "$(basename "$LOG_FILE")" "$LATEST_LOG"
ln -sf "$(basename "$EXIT_FILE")" "$LATEST_EXIT"
ln -sf "$(basename "$PID_FILE")" "$LATEST_PID"

rm -f "$EXIT_FILE" "$PID_FILE"
rm -f /tmp/libtpu_lockfile || true

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [ -n "${WANDB_MODE:-}" ]; then
  export WANDB_MODE
fi
if [ -n "${WANDB_PROJECT:-}" ]; then
  export WANDB_PROJECT
fi
if [ -n "${WANDB_NAME:-}" ]; then
  export WANDB_NAME
fi

RUNNER="set -euo pipefail; \
  rm -f /tmp/libtpu_lockfile || true; \
  if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then \
    source /root/miniconda3/etc/profile.d/conda.sh; conda activate '$ENV_NAME'; \
  fi; \
  cd '$REPO_DIR'; \
  set +e; \
  python -u scripts/run_grpo_gsm8k_training.py --config '$CONFIG_PATH' $PY_ARGS; \
  status=\\$?; \
  set -e; \
  echo \\$status > '$EXIT_FILE'; \
  exit \\$status"

nohup bash -lc "$RUNNER" >"$LOG_FILE" 2>&1 &
pid="$!"
echo "$pid" >"$PID_FILE"

echo "PID=$pid"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "LOG_FILE=$LOG_FILE"
echo "EXIT_FILE=$EXIT_FILE"
echo "LATEST_LOG=$LATEST_LOG"
echo "LATEST_EXIT=$LATEST_EXIT"
echo "LATEST_PID=$LATEST_PID"
