#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Auto-restart a GRPO/GSM8K training run from a YAML config on a TPU VM.

This is designed for spot/preemptible TPUs: if the TPU is PREEMPTED, the script
will delete it, create a new one, and re-run the same config (expected to use a
GCS checkpoint dir with resume=true).

Usage:
  scripts/tpu_vm_autorestart_grpo_gsm8k_from_config.sh --config projects/gsm8k_grpo/configs/<file>.yaml

Required:
  --config PATH             YAML config path (inside the repo on the TPU VM)

Optional:
  --type v4-8|v6e-8         TPU type (default: v4-8)
  --zone ZONE               GCP zone (default: us-central2-b)
  --zones Z1,Z2,...         Zone fallback list (overrides --zone)
  --project PROJECT         GCP project (default: gcloud config project)
  --spot                    Use spot TPU (default)
  --on-demand               Use on-demand TPU (no spot)
  --branch BRANCH_OR_SHA    Git branch or commit to run (default: test-rl)
  --repo-url URL            Git repo URL (default: https://github.com/demon2036/MLLM-JAX.git)
  --name-prefix PREFIX      TPU name prefix (default: mllm-jax-<type>-grpo-autorestart)
  --env-name ENV            Conda env name (default: mllm-jax)
  --python VERSION          Conda Python version (default: 3.12)
  --poll-secs N             Poll interval seconds (default: 60)
  --max-restarts N          Max restarts on failures (0=infinite; default: 0)

Notes:
  - Secrets are synced via scripts/sync_env_to_tpu_vm.sh (expects local .env).
  - Code is synced via Git on the TPU VM (no scp for code).
  - The workload is launched via: scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

CONFIG_PATH=""

TPU_TYPE="v4-8"
ZONE="us-central2-b"
ZONES_CSV=""
PROJECT=""
USE_SPOT="1"

BRANCH_OR_SHA="test-rl"
REPO_URL="https://github.com/demon2036/MLLM-JAX.git"
NAME_PREFIX=""

ENV_NAME="mllm-jax"
PYTHON_VERSION="3.12"

POLL_SECS="60"
MAX_RESTARTS="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="${2:-}"; shift 2 ;;
    --type)
      TPU_TYPE="${2:-}"; shift 2 ;;
    --zone)
      ZONE="${2:-}"; shift 2 ;;
    --zones)
      ZONES_CSV="${2:-}"; shift 2 ;;
    --project)
      PROJECT="${2:-}"; shift 2 ;;
    --spot)
      USE_SPOT="1"; shift ;;
    --on-demand|--no-spot)
      USE_SPOT="0"; shift ;;
    --branch)
      BRANCH_OR_SHA="${2:-}"; shift 2 ;;
    --repo-url)
      REPO_URL="${2:-}"; shift 2 ;;
    --name-prefix)
      NAME_PREFIX="${2:-}"; shift 2 ;;
    --env-name)
      ENV_NAME="${2:-}"; shift 2 ;;
    --python)
      PYTHON_VERSION="${2:-}"; shift 2 ;;
    --poll-secs)
      POLL_SECS="${2:-}"; shift 2 ;;
    --max-restarts)
      MAX_RESTARTS="${2:-}"; shift 2 ;;
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

if [[ -z "$NAME_PREFIX" ]]; then
  NAME_PREFIX="mllm-jax-${TPU_TYPE}-grpo-autorestart"
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

CONFIG_TAG="$(basename "$CONFIG_PATH")"
CONFIG_TAG="${CONFIG_TAG%.yaml}"
LATEST_LOG="logs/nohup_${CONFIG_TAG}_latest.log"
LATEST_EXIT="logs/nohup_${CONFIG_TAG}_latest.exit"

# Zones to try (in order).
ZONE_LIST=()
if [[ -n "$ZONES_CSV" ]]; then
  while IFS= read -r z; do
    [[ -n "$z" ]] && ZONE_LIST+=("$z")
  done < <(printf '%s' "$ZONES_CSV" | tr ',' '\n')
else
  ZONE_LIST+=("$ZONE")
fi

_tpu_state() {
  local name="$1"
  local zone="$2"
  gcloud alpha compute tpus tpu-vm describe "$name" \
    --project="$PROJECT" \
    --zone="$zone" \
    --format='value(state)' 2>/dev/null || true
}

_wait_ready() {
  local name="$1"
  local zone="$2"
  local deadline=$((SECONDS + 1800))
  while true; do
    local state
    state="$(_tpu_state "$name" "$zone")"
    if [[ "$state" == "READY" ]]; then
      return 0
    fi
    if [[ "$state" == "PREEMPTED" || "$state" == "TERMINATED" || "$state" == "STOPPED" ]]; then
      echo "TPU entered terminal state during provisioning: state=$state" >&2
      return 1
    fi
    if (( SECONDS > deadline )); then
      echo "Timed out waiting for TPU READY (state=$state)" >&2
      return 1
    fi
    echo "Waiting for TPU READY... zone=$zone state=${state:-<unknown>} (sleep 30s)"
    sleep 30
  done
}

_remote() {
  local name="$1"
  local zone="$2"
  local cmd="$3"
  scripts/ssh_tpu_vm_root.sh \
    --name "$name" \
    --zone "$zone" \
    --project "$PROJECT" \
    --command "$cmd"
}

_delete_tpu() {
  local name="$1"
  local zone="$2"
  set +e
  scripts/delete_tpu_vm.sh --name "$name" --zone "$zone" --project "$PROJECT" >/dev/null 2>&1
  set -e
}

restarts=0
attempt=0
while true; do
  if [[ "$MAX_RESTARTS" != "0" && "$restarts" -ge "$MAX_RESTARTS" ]]; then
    echo "Reached --max-restarts=$MAX_RESTARTS, stopping." >&2
    exit 1
  fi

  zone_idx=$((attempt % ${#ZONE_LIST[@]}))
  ZONE="${ZONE_LIST[$zone_idx]}"
  attempt=$((attempt + 1))

  ts="$(date -u +%y%m%d%H%M%S)"
  TPU_NAME="${NAME_PREFIX}-${ts}"
  echo
  echo "=== Attempt ${attempt} (restarts=${restarts}) ==="
  echo "TPU_NAME=$TPU_NAME"
  echo "PROJECT=$PROJECT"
  echo "ZONE=$ZONE"
  echo "TPU_TYPE=$TPU_TYPE"
  echo "PROVISIONING=$([[ "$USE_SPOT" == "1" ]] && echo spot || echo on-demand)"
  echo "BRANCH_OR_SHA=$BRANCH_OR_SHA"
  echo "CONFIG_PATH=$CONFIG_PATH"

  set +e
  if [[ "$USE_SPOT" == "1" ]]; then
    scripts/create_tpu_vm.sh --type "$TPU_TYPE" --zone "$ZONE" --project "$PROJECT" --name "$TPU_NAME" --spot
  else
    scripts/create_tpu_vm.sh --type "$TPU_TYPE" --zone "$ZONE" --project "$PROJECT" --name "$TPU_NAME" --on-demand
  fi
  create_rc=$?
  set -e

  if [[ $create_rc -ne 0 ]]; then
    echo "TPU create failed (rc=$create_rc). Will retry with next zone after 30s..." >&2
    _delete_tpu "$TPU_NAME" "$ZONE"
    restarts=$((restarts + 1))
    sleep 30
    continue
  fi

  if ! _wait_ready "$TPU_NAME" "$ZONE"; then
    _delete_tpu "$TPU_NAME" "$ZONE"
    restarts=$((restarts + 1))
    continue
  fi

  scripts/sync_env_to_tpu_vm.sh --name "$TPU_NAME" --zone "$ZONE" --project "$PROJECT" --worker all
  scripts/bootstrap_miniconda_on_tpu_vm.sh --name "$TPU_NAME" --zone "$ZONE" --project "$PROJECT" --env-name "$ENV_NAME" --python "$PYTHON_VERSION"

  _remote "$TPU_NAME" "$ZONE" "set -euo pipefail; REPO_URL='$REPO_URL'; REPO_DIR=/root/MLLM-JAX; if [ ! -d \"\$REPO_DIR/.git\" ]; then git clone \"\$REPO_URL\" \"\$REPO_DIR\"; fi; cd \"\$REPO_DIR\"; git fetch --all --prune; git checkout '$BRANCH_OR_SHA'; if git rev-parse --verify \"origin/$BRANCH_OR_SHA\" >/dev/null 2>&1; then git reset --hard \"origin/$BRANCH_OR_SHA\"; fi; echo git_head=\"\$(git rev-parse --short HEAD)\"; git status -sb"

  _remote "$TPU_NAME" "$ZONE" "set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate '$ENV_NAME'; pip install -U pip; pip install -U \"jax[tpu]\" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; cd /root/MLLM-JAX; pip install -U -r requirements-tpu.txt; python -c 'import jax; print(jax.__version__); print(jax.default_backend())'"

  _remote "$TPU_NAME" "$ZONE" "set -euo pipefail; cd /root/MLLM-JAX; bash scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh --config '$CONFIG_PATH'"

  echo
  echo "Monitoring (poll ${POLL_SECS}s):"
  echo "  - ${LATEST_LOG}"
  echo "  - ${LATEST_EXIT}"

  while true; do
    state="$(_tpu_state "$TPU_NAME" "$ZONE")"
    if [[ "$state" == "PREEMPTED" ]]; then
      echo "TPU PREEMPTED. Deleting and restarting..."
      _delete_tpu "$TPU_NAME" "$ZONE"
      restarts=$((restarts + 1))
      break
    fi
    if [[ -z "$state" ]]; then
      echo "TPU not found (maybe deleted). Restarting..."
      restarts=$((restarts + 1))
      break
    fi

    set +e
    exit_code="$(_remote "$TPU_NAME" "$ZONE" "set -euo pipefail; cd /root/MLLM-JAX; if [ -f '$LATEST_EXIT' ]; then cat '$LATEST_EXIT'; else echo ''; fi" 2>/dev/null)"
    rc=$?
    set -e

    if [[ $rc -ne 0 ]]; then
      echo "SSH failed (state=$state). Checking again in ${POLL_SECS}s..."
      sleep "$POLL_SECS"
      continue
    fi

    exit_code="$(echo "$exit_code" | tr -d '[:space:]')"
    if [[ -n "$exit_code" ]]; then
      echo "Job finished: exit_code=$exit_code"
      if [[ "$exit_code" == "0" ]]; then
        echo "Success. Cleaning up TPU."
        _delete_tpu "$TPU_NAME" "$ZONE"
        exit 0
      fi

      if [[ "$exit_code" == "42" ]]; then
        echo "Preemption-signaled exit (42). Deleting and restarting..."
        _delete_tpu "$TPU_NAME" "$ZONE"
        restarts=$((restarts + 1))
        break
      fi

      echo "Non-zero exit. Tail of log:"
      set +e
      _remote "$TPU_NAME" "$ZONE" "set -euo pipefail; cd /root/MLLM-JAX; tail -n 80 '$LATEST_LOG' || true" || true
      set -e
      echo "Keeping TPU for debugging (delete it when done): $TPU_NAME (zone=$ZONE)" >&2
      exit 1
    fi

    sleep "$POLL_SECS"
  done
done
