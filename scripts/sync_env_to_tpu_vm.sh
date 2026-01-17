#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Sync a local .env file to a Cloud TPU VM (optionally all workers).

This is intended for secrets (e.g. WANDB_API_KEY) that must NOT be committed to Git.

Usage:
  scripts/sync_env_to_tpu_vm.sh --name TPU_NAME --zone ZONE
                               [--project PROJECT]
                               [--src PATH_TO_ENV]
                               [--dest REMOTE_PATH]
                               [--worker WORKER]

Defaults:
  --src    .env
  --dest   /root/.env
  --worker all

Example:
  scripts/sync_env_to_tpu_vm.sh --name my-tpu --zone us-central2-b
  scripts/sync_env_to_tpu_vm.sh --name my-tpu --zone us-central2-b --worker all
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TPU_NAME=""
ZONE=""
PROJECT=""
SRC=".env"
DEST="/root/.env"
WORKER="all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      TPU_NAME="${2:-}"; shift 2 ;;
    --zone)
      ZONE="${2:-}"; shift 2 ;;
    --project)
      PROJECT="${2:-}"; shift 2 ;;
    --src)
      SRC="${2:-}"; shift 2 ;;
    --dest)
      DEST="${2:-}"; shift 2 ;;
    --worker)
      WORKER="${2:-}"; shift 2 ;;
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

if [[ ! -f "$SRC" ]]; then
  echo "Env file not found: $SRC" >&2
  exit 1
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

echo "Project: $PROJECT"
echo "TPU:     $TPU_NAME ($ZONE)"
echo "Worker:  $WORKER"
echo "SRC:     $SRC"
echo "DEST:    $DEST"
echo

gcloud alpha compute tpus tpu-vm scp "$SRC" "root@${TPU_NAME}:${DEST}" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --worker="$WORKER" \
  --quiet

gcloud alpha compute tpus tpu-vm ssh "root@${TPU_NAME}" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --worker="$WORKER" \
  --quiet \
  --command "set -euo pipefail; chmod 600 '$DEST'; test -f '$DEST'; echo 'env_synced=1 dest=$DEST'; ls -la '$DEST'"

