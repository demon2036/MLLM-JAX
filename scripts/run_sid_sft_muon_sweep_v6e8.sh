#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE' >&2
Run a small Muon LR sweep for MiniOneRec SID SFT (TPU).

Usage (from repo root, on TPU VM):
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate mllm-jax
  if [ -f /root/.env ]; then set -a; source /root/.env; set +a; fi
  bash scripts/run_sid_sft_muon_sweep_v6e8.sh

Notes:
  - This script does NOT override hyperparams. Add/edit YAMLs under
    `projects/sid_sft/configs/train/v6e-8/muon_sweep/` instead.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

configs=(
  # Baseline
  projects/sid_sft/configs/train/v6e-8/sid_sft_jax_qwen25_1p5b_base_industrial_v6e8_e3_adamw_train.yaml
  # Sweep points
  projects/sid_sft/configs/train/v6e-8/muon_sweep/sid_sft_jax_qwen25_1p5b_base_industrial_v6e8_e3_muon_lr1e3_aux3e4_train.yaml
  projects/sid_sft/configs/train/v6e-8/muon_sweep/sid_sft_jax_qwen25_1p5b_base_industrial_v6e8_e3_muon_lr3e3_aux1e4_train.yaml
  projects/sid_sft/configs/train/v6e-8/sid_sft_jax_qwen25_1p5b_base_industrial_v6e8_e3_muon_lr3e3_aux3e4_train.yaml
  projects/sid_sft/configs/train/v6e-8/muon_sweep/sid_sft_jax_qwen25_1p5b_base_industrial_v6e8_e3_muon_lr3e3_aux1e3_train.yaml
  projects/sid_sft/configs/train/v6e-8/muon_sweep/sid_sft_jax_qwen25_1p5b_base_industrial_v6e8_e3_muon_lr6e3_aux3e4_train.yaml
)

for cfg in "${configs[@]}"; do
  if [[ ! -f "$cfg" ]]; then
    echo "Config not found: $cfg" >&2
    exit 2
  fi
done

for cfg in "${configs[@]}"; do
  echo
  echo "============================================================"
  echo "RUN: $cfg"
  echo "============================================================"
  python -u scripts/run_sid_sft.py --config "$cfg" --run-mode train
done

