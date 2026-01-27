#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="/var/log/tpu_startup_sid_sft_official_eval.log"
if command -v logger >/dev/null 2>&1; then
  exec > >(tee -a "$LOG_FILE" | logger -t tpu-startup-sid-sft) 2>&1
else
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

trap 'log "startup failed at line $LINENO";' ERR

log "startup begin"

meta() {
  curl -fsSL -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" || true
}

WANDB_API_KEY="$(meta WANDB_API_KEY | tr -d '\r' | xargs)"
WANDB_MODE="$(meta WANDB_MODE | tr -d '\r' | xargs)"
WANDB_PROJECT="$(meta WANDB_PROJECT | tr -d '\r' | xargs)"
REPO_URL="$(meta REPO_URL | tr -d '\r' | xargs)"
REPO_REF="$(meta REPO_REF | tr -d '\r' | xargs)"

if [[ -z "$WANDB_API_KEY" ]]; then
  echo "WANDB_API_KEY is required via instance metadata."
  exit 1
fi

if [[ -z "$WANDB_MODE" ]]; then
  WANDB_MODE="online"
fi
if [[ -z "$WANDB_PROJECT" ]]; then
  WANDB_PROJECT="minionerec-sid-sft"
fi
if [[ -z "$REPO_URL" ]]; then
  REPO_URL="https://github.com/demon2036/MLLM-JAX.git"
fi
if [[ -z "$REPO_REF" ]]; then
  REPO_REF="john/sid3-fixed-prefill-single-compile"
fi

export WANDB_API_KEY WANDB_MODE WANDB_PROJECT
export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false

apt-get update
apt-get install -y git git-lfs curl
git lfs install

if [[ ! -d /root/miniconda3 ]]; then
  curl -fsSL -o /root/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /root/miniconda.sh -b -p /root/miniconda3
  rm -f /root/miniconda.sh
fi

source /root/miniconda3/etc/profile.d/conda.sh
yes | conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
yes | conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
if ! conda env list | awk '{print $1}' | grep -qx mllm-jax; then
  conda create -y -n mllm-jax python=3.12
fi
conda activate mllm-jax

python -m pip install -U pip
python -m pip install -U wandb
python - <<'PY'
import os
import time
import wandb

project = os.environ.get("WANDB_PROJECT", "minionerec-sid-sft")
run = wandb.init(
    project=project,
    name=f"sid-sft-startup-{int(time.time())}",
    mode=os.environ.get("WANDB_MODE", "online"),
)
run.log({"startup/heartbeat": 1})
run.finish()
PY
python -m pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python -m pip install -U torch --index-url https://download.pytorch.org/whl/cpu

if [[ ! -d /root/MLLM-JAX/.git ]]; then
  git clone "$REPO_URL" /root/MLLM-JAX
fi

cd /root/MLLM-JAX
git fetch --all
git checkout "$REPO_REF"
git pull

python -m pip install -U -r requirements-tpu.txt
python -m pip install -U fire pandas

mkdir -p workdir
if [[ ! -d workdir/MiniOneRec/.git ]]; then
  git clone https://github.com/AkaliKong/MiniOneRec workdir/MiniOneRec
fi

mkdir -p workdir/hf_ckpts
if [[ ! -d workdir/hf_ckpts/kkknight_MiniOneRec/.git ]]; then
  git clone https://huggingface.co/kkknight/MiniOneRec workdir/hf_ckpts/kkknight_MiniOneRec
fi

cd workdir/hf_ckpts/kkknight_MiniOneRec
git lfs pull --include "Industrial_ckpt/*,Office_ckpt/*"
cd /root/MLLM-JAX

python - <<'PY'
import json
import os

paths = [
    "workdir/hf_ckpts/kkknight_MiniOneRec/Industrial_ckpt/config.json",
    "workdir/hf_ckpts/kkknight_MiniOneRec/Office_ckpt/config.json",
]

for path in paths:
    if not os.path.exists(path):
        continue
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "rope_theta" not in cfg:
        cfg["rope_theta"] = 10000.0
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
            f.write("\n")
PY

run_eval() {
  local tag="$1"
  local config_path="$2"
  local output_dir="$3"
  local item_path="$4"
  rm -f /tmp/libtpu_lockfile || true
  ./scripts/run_sid_sft.sh --config "$config_path" --run-mode eval
  local calc_log="${output_dir}/calc_${tag}.log"
  python workdir/MiniOneRec/calc.py --path "${output_dir}/eval_predictions.json" --item_path "${item_path}" | tee "$calc_log"
  ARTIFACT_TAG="$tag" OUTPUT_DIR="$output_dir" CALC_LOG="$calc_log" python - <<'PY'
import json
import os
import time

import wandb

tag = os.environ["ARTIFACT_TAG"]
output_dir = os.environ["OUTPUT_DIR"]
calc_log = os.environ["CALC_LOG"]
project = os.environ.get("WANDB_PROJECT", "minionerec-sid-sft")

run = wandb.init(
    project=project,
    name=f"sid-sft-{tag}-artifacts-{int(time.time())}",
    mode=os.environ.get("WANDB_MODE", "online"),
)
artifact = wandb.Artifact(f"sid-sft-{tag}-eval", type="eval_predictions")

for name in ["eval_predictions.json", "eval_predictions.metrics.json"]:
    path = os.path.join(output_dir, name)
    if os.path.exists(path):
        artifact.add_file(path)
if os.path.exists(calc_log):
    artifact.add_file(calc_log)

metrics_path = os.path.join(output_dir, "eval_predictions.metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    for k, v in (metrics.get("hr") or {}).items():
        run.summary[f"eval_metrics/hr@{k}"] = v
    for k, v in (metrics.get("ndcg") or {}).items():
        run.summary[f"eval_metrics/ndcg@{k}"] = v
    if "invalid_prediction_count" in metrics:
        run.summary["eval_metrics/invalid_prediction_count"] = metrics["invalid_prediction_count"]

run.log_artifact(artifact)
run.finish()
PY
}

run_eval \
  "industrial" \
  "projects/sid_sft/configs/sid_sft_jax_eval_official_minionerec_industrial_ckpt.yaml" \
  "runs/sid_sft_jax_eval_official_minionerec_industrial_ckpt" \
  "workdir/MiniOneRec/data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt"

run_eval \
  "office" \
  "projects/sid_sft/configs/sid_sft_jax_eval_official_minionerec_office_ckpt.yaml" \
  "runs/sid_sft_jax_eval_official_minionerec_office_ckpt" \
  "workdir/MiniOneRec/data/Amazon/info/Office_Products_5_2016-10-2018-11.txt"
