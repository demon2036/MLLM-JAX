#!/usr/bin/env bash
set -euo pipefail

# Start GRPO/GSM8K training from a YAML config via `nohup` on a multi-host TPU VM.
#
# Run this on *all* TPU workers (e.g. via `gcloud ... --worker=all`) so
# `jax.distributed.initialize()` can form the multi-host runtime.
#
# This wrapper sets:
# - REQUIRE_MULTIHOST=1 (fail fast if only worker 0 launched)
# - optionally REQUIRE_JAX_PROCESS_COUNT=N (exact process count guard)
#
# IMPORTANT (repo policy):
# - Do NOT use env vars to override training hyperparams.
# - If you want to change config values, create a new YAML config file and pass
#   it explicitly via `--config ...`.

usage() {
  cat <<'USAGE'
Start GRPO/GSM8K training via nohup on a multi-host TPU VM.

Usage:
  bash scripts/tpu_vm_start_grpo_gsm8k_from_config_multihost_nohup.sh --config projects/gsm8k_grpo/configs/<file>.yaml

Optional:
  --env-name NAME               Conda env name (default: mllm-jax)
  --require-jax-process-count N Require jax.process_count()==N

Notes:
  - Launch this on all workers: `gcloud ... tpu-vm ssh --worker=all --command ...`
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ENV_NAME="mllm-jax"
CONFIG_PATH=""
REQUIRE_JAX_PROCESS_COUNT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="${2:-}"; shift 2 ;;
    --env-name)
      ENV_NAME="${2:-}"; shift 2 ;;
    --require-jax-process-count)
      REQUIRE_JAX_PROCESS_COUNT="${2:-}"; shift 2 ;;
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

export REQUIRE_MULTIHOST="${REQUIRE_MULTIHOST:-1}"
if [[ -n "$REQUIRE_JAX_PROCESS_COUNT" ]]; then
  export REQUIRE_JAX_PROCESS_COUNT="$REQUIRE_JAX_PROCESS_COUNT"
fi

exec bash scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh \
  --env-name "$ENV_NAME" \
  --config "$CONFIG_PATH"
