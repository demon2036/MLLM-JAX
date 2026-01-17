#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Create a Cloud TPU VM (spot by default).

Usage:
  scripts/create_tpu_vm.sh [--type v4-8|v6e-8] [--zone ZONE] [--name TPU_NAME]
                           [--version RUNTIME_VERSION] [--project PROJECT]
                           [--spot|--on-demand] [--async] [--dry-run]

Defaults:
  --type    v6e-8
  --zone    us-central2-b
  --spot    enabled (use --on-demand to disable)
  --version auto:
            v6e-* -> v6e-ubuntu-2404
            v4-*  -> tpu-ubuntu2204-base

Examples:
  scripts/create_tpu_vm.sh
  scripts/create_tpu_vm.sh --type v4-8 --zone us-central2-b
  scripts/create_tpu_vm.sh --type v6e-8 --name mllm-jax-v6e-8-$(date +%y%m%d%H%M%S)

Notes:
  - This will allocate billable resources (especially once READY).
  - Use scripts/delete_tpu_vm.sh to delete when finished.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TPU_TYPE="v6e-8"
ZONE="us-central2-b"
TPU_NAME=""
RUNTIME_VERSION=""
PROJECT=""
USE_SPOT="1"
USE_ASYNC="0"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --type)
      TPU_TYPE="${2:-}"; shift 2 ;;
    --zone)
      ZONE="${2:-}"; shift 2 ;;
    --name)
      TPU_NAME="${2:-}"; shift 2 ;;
    --version)
      RUNTIME_VERSION="${2:-}"; shift 2 ;;
    --project)
      PROJECT="${2:-}"; shift 2 ;;
    --spot)
      USE_SPOT="1"; shift ;;
    --on-demand|--no-spot)
      USE_SPOT="0"; shift ;;
    --async)
      USE_ASYNC="1"; shift ;;
    --dry-run)
      DRY_RUN="1"; shift ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

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

if [[ -z "$TPU_NAME" ]]; then
  ts="$(date +%y%m%d%H%M%S)"
  TPU_NAME="mllm-jax-${TPU_TYPE}-${ts}"
fi

if [[ -z "$RUNTIME_VERSION" ]]; then
  case "$TPU_TYPE" in
    v6e-*)
      RUNTIME_VERSION="v6e-ubuntu-2404" ;;
    v4-*)
      RUNTIME_VERSION="tpu-ubuntu2204-base" ;;
    *)
      echo "Unsupported --type '$TPU_TYPE' (expected v4-* or v6e-*). Provide --version explicitly." >&2
      exit 2 ;;
  esac
fi

create_args=(
  "alpha" "compute" "tpus" "tpu-vm" "create" "$TPU_NAME"
  "--project=$PROJECT"
  "--zone=$ZONE"
  "--accelerator-type=$TPU_TYPE"
  "--version=$RUNTIME_VERSION"
  "--quiet"
)
if [[ "$USE_SPOT" == "1" ]]; then
  create_args+=("--spot")
fi
if [[ "$USE_ASYNC" == "1" ]]; then
  create_args+=("--async")
fi

echo "Project:         $PROJECT"
echo "Zone:            $ZONE"
echo "TPU name:        $TPU_NAME"
echo "Accelerator:     $TPU_TYPE"
echo "Runtime version: $RUNTIME_VERSION"
echo "Provisioning:    $([[ "$USE_SPOT" == "1" ]] && echo spot || echo on-demand)"
echo

echo "gcloud ${create_args[*]}"
if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

gcloud "${create_args[@]}"

echo
echo "Next:"
echo "  - SSH as root:  scripts/ssh_tpu_vm_root.sh --name \"$TPU_NAME\" --zone \"$ZONE\""
echo "  - Delete TPU:   scripts/delete_tpu_vm.sh --name \"$TPU_NAME\" --zone \"$ZONE\""
