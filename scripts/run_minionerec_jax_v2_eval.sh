#!/usr/bin/env bash
set -euo pipefail

CONFIG=""
PRINT_CONFIG=0
RUN_TAG=""

usage() {
  echo "Usage: $0 --config <path>.yaml [--print-config] [--run-tag <tag>]" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="${2:-}"
      shift 2
      ;;
    --print-config)
      PRINT_CONFIG=1
      shift 1
      ;;
    --run-tag)
      RUN_TAG="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "Missing --config" >&2
  usage
fi

ARGS=(--config "$CONFIG")
if [[ "$PRINT_CONFIG" -eq 1 ]]; then
  ARGS+=(--print-config)
fi
if [[ -n "$RUN_TAG" ]]; then
  ARGS+=(--run-tag "$RUN_TAG")
fi

python scripts/run_minionerec_jax_v2_eval.py "${ARGS[@]}"
