#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Create a Cloud TPU VM, retrying across multiple zones.

Usage:
  scripts/create_tpu_vm_retry.sh --type v6e-16 --zone us-east5-b --zone us-east1-d [--name TPU_NAME]
  scripts/create_tpu_vm_retry.sh --type v6e-16 --zones us-east5-b,us-east1-d [--name TPU_NAME]

Options:
  --type TYPE              TPU accelerator type (e.g. v6e-8, v6e-16, v4-8)
  --zone ZONE              Add a zone candidate (repeatable)
  --zones Z1,Z2,...        Comma-separated zone candidates (alternative to repeated --zone)
  --name TPU_NAME          TPU VM name (defaults to scripts/create_tpu_vm.sh auto-name)
  --project PROJECT        GCP project (defaults to gcloud config)
  --version RUNTIME        Runtime version (defaults to scripts/create_tpu_vm.sh auto-version)
  --spot                   Use spot (default)
  --on-demand              Use on-demand (disable spot)
  --async                  Create asynchronously
  --dry-run                Print attempted commands only

Notes:
  - This script calls scripts/create_tpu_vm.sh internally.
  - If a zone fails with insufficient capacity, it continues to the next one.
  - On success it prints the chosen zone and exits 0.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TPU_TYPE="v6e-8"
TPU_NAME=""
PROJECT=""
RUNTIME_VERSION=""
USE_SPOT="1"
USE_ASYNC="0"
DRY_RUN="0"

zones=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --type)
      TPU_TYPE="${2:-}"; shift 2 ;;
    --zone)
      zones+=("${2:-}"); shift 2 ;;
    --zones)
      IFS=',' read -r -a zones_from_csv <<<"${2:-}"
      for z in "${zones_from_csv[@]}"; do
        if [[ -n "${z// /}" ]]; then
          zones+=("${z}")
        fi
      done
      shift 2 ;;
    --name)
      TPU_NAME="${2:-}"; shift 2 ;;
    --project)
      PROJECT="${2:-}"; shift 2 ;;
    --version)
      RUNTIME_VERSION="${2:-}"; shift 2 ;;
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

if [[ ${#zones[@]} -eq 0 ]]; then
  echo "Missing required zone candidates. Provide --zone (repeatable) or --zones." >&2
  usage >&2
  exit 2
fi

create_base=(
  "scripts/create_tpu_vm.sh"
  "--type" "$TPU_TYPE"
)
if [[ -n "$TPU_NAME" ]]; then
  create_base+=("--name" "$TPU_NAME")
fi
if [[ -n "$PROJECT" ]]; then
  create_base+=("--project" "$PROJECT")
fi
if [[ -n "$RUNTIME_VERSION" ]]; then
  create_base+=("--version" "$RUNTIME_VERSION")
fi
if [[ "$USE_SPOT" == "1" ]]; then
  create_base+=("--spot")
else
  create_base+=("--on-demand")
fi
if [[ "$USE_ASYNC" == "1" ]]; then
  create_base+=("--async")
fi
if [[ "$DRY_RUN" == "1" ]]; then
  create_base+=("--dry-run")
fi

last_rc=1
for zone in "${zones[@]}"; do
  zone="$(echo "$zone" | xargs)"
  if [[ -z "$zone" ]]; then
    continue
  fi
  echo
  echo "=== Attempt zone: $zone (type=$TPU_TYPE, provisioning=$([[ "$USE_SPOT" == "1" ]] && echo spot || echo on-demand)) ==="
  if "${create_base[@]}" --zone "$zone"; then
    echo
    echo "SUCCESS: created TPU in zone=$zone"
    exit 0
  else
    last_rc=$?
    echo
    echo "FAILED: zone=$zone (exit=$last_rc). Trying next zone..."
  fi
done

echo
echo "ERROR: all zones failed for type=$TPU_TYPE." >&2
exit "$last_rc"

