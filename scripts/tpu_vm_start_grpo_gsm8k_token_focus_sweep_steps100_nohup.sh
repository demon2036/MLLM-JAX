#!/usr/bin/env bash
set -euo pipefail

# Start GRPO/GSM8K token-focus sweep (100 steps, eval_full_every_steps=50) via `nohup` on a TPU VM.
#
# Intended usage: execute ON the TPU VM inside a Git-synced repo checkout.
# It runs `scripts/run_grpo_gsm8k_token_focus_sweep_steps100_v6e8.sh` in the background and
# records PID/exit code files under `logs/`.

usage() {
  cat <<'USAGE'
Start GRPO/GSM8K token-focus sweep (steps=100, full eval every 50) via nohup.

Usage:
  bash scripts/tpu_vm_start_grpo_gsm8k_token_focus_sweep_steps100_nohup.sh

Optional:
  --env-name NAME   Conda env name (default: mllm-jax)

Notes:
  - This script does NOT override hyperparams. Edit YAMLs under:
      projects/gsm8k_grpo/configs/token_focus_sweep_steps100/
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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="${2:-}"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
CONFIG_TAG="grpo_gsm8k_token_focus_sweep_steps100_v6e8"

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

RUNNER="set -euo pipefail; \
  rm -f /tmp/libtpu_lockfile || true; \
  if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then \
    source /root/miniconda3/etc/profile.d/conda.sh; conda activate '$ENV_NAME'; \
  fi; \
  cd '$REPO_DIR'; \
  set +e; \
  bash scripts/run_grpo_gsm8k_token_focus_sweep_steps100_v6e8.sh; \
  status=\$?; \
  set -e; \
  echo \$status > '$EXIT_FILE'; \
  exit \$status"

nohup bash -lc "$RUNNER" >"$LOG_FILE" 2>&1 &
pid="$!"
echo "$pid" >"$PID_FILE"

echo "PID=$pid"
echo "LOG_FILE=$LOG_FILE"
echo "EXIT_FILE=$EXIT_FILE"
echo "LATEST_LOG=$LATEST_LOG"
echo "LATEST_EXIT=$LATEST_EXIT"
echo "LATEST_PID=$LATEST_PID"

