#!/usr/bin/env bash
set -euo pipefail

# Start OpenOneRec RecIF-Bench eval via `nohup` on a TPU VM from an explicit YAML config.
#
# Repo policy reminder:
# - Do NOT use env vars to override evaluation hyperparams; make a new YAML config.

usage() {
  cat <<'USAGE'
Start OpenOneRec RecIF-Bench eval via nohup from a YAML config.

Usage:
  bash scripts/tpu_vm_start_openonerec_recif_bench_eval_from_config_nohup.sh \
    --config projects/openonerec_recif_bench_eval/configs/recif_bench_onerec_1p7b.yaml

Optional:
  --env-name NAME   Conda env name (default: mllm-jax)
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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="${2:-}"; shift 2 ;;
    --env-name)
      ENV_NAME="${2:-}"; shift 2 ;;
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

RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
CONFIG_TAG="$(basename "$CONFIG_PATH")"
CONFIG_TAG="${CONFIG_TAG%.yaml}"

LOG_FILE="logs/nohup_openonerec_recif_${CONFIG_TAG}_${RUN_ID}.log"
EXIT_FILE="logs/nohup_openonerec_recif_${CONFIG_TAG}_${RUN_ID}.exit"
PID_FILE="logs/nohup_openonerec_recif_${CONFIG_TAG}_${RUN_ID}.pid"

LATEST_LOG="logs/nohup_openonerec_recif_${CONFIG_TAG}_latest.log"
LATEST_EXIT="logs/nohup_openonerec_recif_${CONFIG_TAG}_latest.exit"
LATEST_PID="logs/nohup_openonerec_recif_${CONFIG_TAG}_latest.pid"

ln -sf "$(basename "$LOG_FILE")" "$LATEST_LOG"
ln -sf "$(basename "$EXIT_FILE")" "$LATEST_EXIT"
ln -sf "$(basename "$PID_FILE")" "$LATEST_PID"

rm -f "$EXIT_FILE" "$PID_FILE"
rm -f /tmp/libtpu_lockfile || true

RUNNER="set -euo pipefail; \
  rm -f /tmp/libtpu_lockfile || true; \
  if [ -f /root/.env ]; then set -a; source /root/.env; set +a; fi; \
  if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then \
    source /root/miniconda3/etc/profile.d/conda.sh; conda activate '$ENV_NAME'; \
    export LD_LIBRARY_PATH=\"/root/miniconda3/envs/$ENV_NAME/lib:\${LD_LIBRARY_PATH:-}\"; \
    export PJRT_DEVICE=\"\${PJRT_DEVICE:-TPU}\"; \
  fi; \
  cd '$REPO_DIR'; \
  set +e; \
  python -u projects/openonerec_recif_bench_eval/run.py --config '$CONFIG_PATH'; \
  status=\$?; \
  set -e; \
  echo \$status > '$EXIT_FILE'; \
  exit \$status"

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
