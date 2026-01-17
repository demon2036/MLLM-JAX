#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
SSH to a Cloud TPU VM as root (via gcloud).

Usage:
  scripts/ssh_tpu_vm_root.sh --name TPU_NAME --zone ZONE [--project PROJECT] [--command CMD]

Examples:
  scripts/ssh_tpu_vm_root.sh --name my-tpu --zone us-central2-b
  scripts/ssh_tpu_vm_root.sh --name my-tpu --zone us-central2-b --command 'whoami'
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TPU_NAME=""
ZONE=""
PROJECT=""
COMMAND=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      TPU_NAME="${2:-}"; shift 2 ;;
    --zone)
      ZONE="${2:-}"; shift 2 ;;
    --project)
      PROJECT="${2:-}"; shift 2 ;;
    --command)
      COMMAND="${2:-}"; shift 2 ;;
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

ssh_args=(
  "alpha" "compute" "tpus" "tpu-vm" "ssh"
  "root@${TPU_NAME}"
  "--project=$PROJECT"
  "--zone=$ZONE"
)

if [[ -n "$COMMAND" ]]; then
  ssh_args+=("--quiet" "--command=$COMMAND")
fi

gcloud "${ssh_args[@]}"
