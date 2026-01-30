#!/usr/bin/env bash
set -euo pipefail

# Start MiniOneRec SID SFT from a YAML config via `nohup` on a TPU VM.
#
# This script is intended to be executed ON the TPU VM inside a Git-synced repo
# checkout (e.g. `/root/MLLM-JAX`). It starts the job in the background via
# `nohup` and prints the PID + log paths.
#
# IMPORTANT (repo policy):
# - Do NOT use env vars to override training hyperparams.
# - If you want to change config values, create a new YAML config file and pass
#   it explicitly via `--config ...`.

usage() {
  cat <<'USAGE'
Start MiniOneRec SID SFT via nohup from an explicit YAML config.

Usage:
  bash scripts/tpu_vm_start_sid_sft_from_config_nohup.sh --config projects/minionerec/sft/configs/<group>/<file>.yaml

Optional:
  --env-name NAME   Conda env name (default: mllm-jax)
  --run-mode MODE   train|eval|train_eval (default: train)

Notes:
  - This script intentionally does NOT accept hyperparam overrides. Make a new
    YAML file if you need different training settings.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

mkdir -p logs

ENV_NAME="mllm-jax"
CONFIG_PATH=""
RUN_MODE="train"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="${2:-}"; shift 2 ;;
    --env-name)
      ENV_NAME="${2:-}"; shift 2 ;;
    --run-mode)
      RUN_MODE="${2:-}"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if [[ -z "$CONFIG_PATH" ]]; then
  echo "Missing required arg: --config" >&2
  usage >&2
  exit 2
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 2
fi

case "$RUN_MODE" in
  train|eval|train_eval) ;;
  *)
    echo "Invalid --run-mode: $RUN_MODE (expected train|eval|train_eval)" >&2
    exit 2 ;;
esac

RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
CONFIG_TAG="$(basename "$CONFIG_PATH")"
CONFIG_TAG="${CONFIG_TAG%.yaml}"
HOST_TAG="$(hostname -s 2>/dev/null || hostname)"

LOG_FILE="logs/nohup_sid_sft_${CONFIG_TAG}_${RUN_MODE}_${HOST_TAG}_${RUN_ID}.log"
EXIT_FILE="logs/nohup_sid_sft_${CONFIG_TAG}_${RUN_MODE}_${HOST_TAG}_${RUN_ID}.exit"
PID_FILE="logs/nohup_sid_sft_${CONFIG_TAG}_${RUN_MODE}_${HOST_TAG}_${RUN_ID}.pid"

LATEST_LOG="logs/nohup_sid_sft_${CONFIG_TAG}_${RUN_MODE}_${HOST_TAG}_latest.log"
LATEST_EXIT="logs/nohup_sid_sft_${CONFIG_TAG}_${RUN_MODE}_${HOST_TAG}_latest.exit"
LATEST_PID="logs/nohup_sid_sft_${CONFIG_TAG}_${RUN_MODE}_${HOST_TAG}_latest.pid"

ln -sf "$(basename "$LOG_FILE")" "$LATEST_LOG"
ln -sf "$(basename "$EXIT_FILE")" "$LATEST_EXIT"
ln -sf "$(basename "$PID_FILE")" "$LATEST_PID"

rm -f "$EXIT_FILE" "$PID_FILE"
rm -f /tmp/libtpu_lockfile || true

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

RUNNER="set -euo pipefail; \
  rm -f /tmp/libtpu_lockfile || true; \
  if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then \
    source /root/miniconda3/etc/profile.d/conda.sh; conda activate '$ENV_NAME'; \
  fi; \
  cd '$REPO_DIR'; \
  set +e; \
  python -u scripts/run_sid_sft.py --config '$CONFIG_PATH' --run-mode '$RUN_MODE'; \
  status=\$?; \
  set -e; \
  echo \$status > '$EXIT_FILE'; \
  exit \$status"

nohup bash -lc "$RUNNER" >"$LOG_FILE" 2>&1 &
pid="$!"
echo "$pid" >"$PID_FILE"

echo "PID=$pid"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "RUN_MODE=$RUN_MODE"
echo "LOG_FILE=$LOG_FILE"
echo "EXIT_FILE=$EXIT_FILE"
echo "LATEST_LOG=$LATEST_LOG"
echo "LATEST_EXIT=$LATEST_EXIT"
echo "LATEST_PID=$LATEST_PID"
