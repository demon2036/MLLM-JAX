#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Bootstrap Miniconda + a Python env on a TPU VM (runs via gcloud SSH as root).

Usage:
  scripts/bootstrap_miniconda_on_tpu_vm.sh --name TPU_NAME --zone ZONE
                                          [--project PROJECT]
                                          [--env-name ENV_NAME]
                                          [--python PYTHON_VERSION]

Defaults:
  --env-name mllm-jax
  --python   3.12
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TPU_NAME=""
ZONE=""
PROJECT=""
ENV_NAME="mllm-jax"
PYTHON_VERSION="3.12"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      TPU_NAME="${2:-}"; shift 2 ;;
    --zone)
      ZONE="${2:-}"; shift 2 ;;
    --project)
      PROJECT="${2:-}"; shift 2 ;;
    --env-name)
      ENV_NAME="${2:-}"; shift 2 ;;
    --python)
      PYTHON_VERSION="${2:-}"; shift 2 ;;
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

install_miniconda_cmd='set -euo pipefail; if [ ! -d /root/miniconda3 ]; then curl -fsSL -o /root/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; bash /root/miniconda.sh -b -p /root/miniconda3; rm -f /root/miniconda.sh; fi; /root/miniconda3/bin/conda --version'
create_env_cmd="set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true; conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true; if ! conda env list | grep -Eq \"^${ENV_NAME}[[:space:]]\"; then conda create -y -n \"$ENV_NAME\" python=$PYTHON_VERSION; fi; conda activate \"$ENV_NAME\"; python --version; pip install -U pip"

gcloud alpha compute tpus tpu-vm ssh "root@${TPU_NAME}" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --quiet \
  --command "$install_miniconda_cmd"

gcloud alpha compute tpus tpu-vm ssh "root@${TPU_NAME}" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --quiet \
  --command "$create_env_cmd"
