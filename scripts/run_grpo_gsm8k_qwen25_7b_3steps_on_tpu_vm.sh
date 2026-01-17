#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run a 3-step GSM8K+GRPO smoke-train with Qwen2.5-7B on a Cloud TPU VM.

This script (from your local machine) will:
  - Bootstrap conda env on the TPU VM
  - Sync this repo to the TPU VM via Git (no SCP)
  - Install TPU JAX + Python deps
  - Run `scripts/run_smoke_grpo_gsm8k_qwen25_7b.py` for 3 steps

Usage:
  scripts/run_grpo_gsm8k_qwen25_7b_3steps_on_tpu_vm.sh --name TPU_NAME --zone ZONE
    [--project PROJECT] [--env-name ENV_NAME] [--ref REF]

Defaults:
  --env-name mllm-jax
  --ref      main

Examples:
  scripts/run_grpo_gsm8k_qwen25_7b_3steps_on_tpu_vm.sh --name my-tpu --zone us-central2-b
  scripts/run_grpo_gsm8k_qwen25_7b_3steps_on_tpu_vm.sh --name my-tpu --zone us-central2-b --ref <commit-sha>
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
REF="main"

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
    --ref|--commit|--branch)
      REF="${2:-}"; shift 2 ;;
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

if [[ -z "$PROJECT" ]]; then
  PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
fi
if [[ -z "$PROJECT" ]]; then
  echo "Could not determine gcloud project. Set one via: gcloud config set project <PROJECT>" >&2
  exit 1
fi

REPO_URL="https://github.com/demon2036/MLLM-JAX.git"
REPO_DIR="/root/MLLM-JAX"

echo "Project:   $PROJECT"
echo "TPU:       $TPU_NAME ($ZONE)"
echo "Conda env: $ENV_NAME"
echo "Git ref:   $REF"
echo

scripts/bootstrap_miniconda_on_tpu_vm.sh \
  --name "$TPU_NAME" \
  --zone "$ZONE" \
  --project "$PROJECT" \
  --env-name "$ENV_NAME" \
  --python "3.12"

scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --project "$PROJECT" --command \
  "set -euo pipefail; \
   REPO_URL='$REPO_URL'; REPO_DIR='$REPO_DIR'; \
   if [ ! -d \"\$REPO_DIR/.git\" ]; then git clone \"\$REPO_URL\" \"\$REPO_DIR\"; fi; \
   cd \"\$REPO_DIR\"; git fetch --all --prune; git checkout '$REF'; git status -sb"

scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --project "$PROJECT" --command \
  "set -euo pipefail; \
   source /root/miniconda3/etc/profile.d/conda.sh; conda activate '$ENV_NAME'; \
   python --version; pip --version; \
   pip install -U pip; \
   pip install -U 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; \
   pip install -U torch --index-url https://download.pytorch.org/whl/cpu; \
   cd '$REPO_DIR'; pip install -U -r requirements-tpu.txt; \
   python -c \"import jax, jaxlib; print('jax', jax.__version__, 'jaxlib', jaxlib.__version__); print(jax.default_backend()); print(jax.devices())\""

scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --project "$PROJECT" --command \
  "set -euo pipefail; \
   rm -f /tmp/libtpu_lockfile || true; \
   source /root/miniconda3/etc/profile.d/conda.sh; conda activate '$ENV_NAME'; \
   cd '$REPO_DIR'; \
   export HF_HUB_ENABLE_HF_TRANSFER=1; \
   export WANDB_MODE=disabled; \
   export TOKENIZERS_PARALLELISM=false; \
   export MODEL_PATH='Qwen/Qwen2.5-7B-Instruct'; \
   export STEPS=3; \
   export BATCH_SIZE=1; \
   export NUM_PRE_Q=8; \
   export MAX_LENGTH_SAMPLE=64; \
   export PPO_EPOCHS=1; \
   export BETA=0.0; \
   python -u scripts/run_smoke_grpo_gsm8k_qwen25_7b.py"

