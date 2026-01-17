#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Delete a Cloud TPU VM (to stop billing).

Usage:
  scripts/delete_tpu_vm.sh --name TPU_NAME --zone ZONE [--project PROJECT]

Examples:
  scripts/delete_tpu_vm.sh --name my-tpu --zone us-central2-b
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TPU_NAME=""
ZONE=""
PROJECT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      TPU_NAME="${2:-}"; shift 2 ;;
    --zone)
      ZONE="${2:-}"; shift 2 ;;
    --project)
      PROJECT="${2:-}"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if [[ -z "$TPU_NAME" || -z "$ZONE" ]]; then
  echo "Missing required args: --name and/or --zone" >&2
  usage >&2
  exit 2
fi

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud not found on PATH." >&2
  exit 1
fi

export CLOUDSDK_CORE_DISABLE_PROMPTS=1

if [[ -z "$PROJECT" ]]; then
  PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
fi
if [[ -z "$PROJECT" ]]; then
  echo "Could not determine gcloud project. Set one via: gcloud config set project <PROJECT>" >&2
  exit 1
fi

gcloud alpha compute tpus tpu-vm delete "$TPU_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --quiet
