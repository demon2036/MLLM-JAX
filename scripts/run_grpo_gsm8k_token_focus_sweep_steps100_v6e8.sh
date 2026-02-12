#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE' >&2
Run a token-focus (prob_threshold, max_tokens_per_sequence) sweep for GRPO/GSM8K (TPU v6e-8).

This variant runs longer (100 steps) and performs full-split eval every 50 steps
via `eval_full_every_steps: 50` in each YAML.

Usage (from repo root, on TPU VM):
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate mllm-jax
  if [ -f /root/.env ]; then set -a; source /root/.env; set +a; fi
  bash scripts/run_grpo_gsm8k_token_focus_sweep_steps100_v6e8.sh

Notes:
  - This script does NOT override hyperparams. Edit YAMLs under:
      projects/gsm8k_grpo/configs/token_focus_sweep_steps100/
  - W&B logging comes from each YAML (wandb_mode=online by default).
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

configs=(
  projects/gsm8k_grpo/configs/token_focus_sweep_steps100/grpo_gsm8k_qwen25_3b_token_focus_sweep_batch16_roll8_p0p3_k1_steps100_evalfull50_v6e8.yaml
  projects/gsm8k_grpo/configs/token_focus_sweep_steps100/grpo_gsm8k_qwen25_3b_token_focus_sweep_batch16_roll8_p0p3_k5_steps100_evalfull50_v6e8.yaml
  projects/gsm8k_grpo/configs/token_focus_sweep_steps100/grpo_gsm8k_qwen25_3b_token_focus_sweep_batch16_roll8_p0p3_k10_steps100_evalfull50_v6e8.yaml
  projects/gsm8k_grpo/configs/token_focus_sweep_steps100/grpo_gsm8k_qwen25_3b_token_focus_sweep_batch16_roll8_p0p3_k20_steps100_evalfull50_v6e8.yaml
  projects/gsm8k_grpo/configs/token_focus_sweep_steps100/grpo_gsm8k_qwen25_3b_token_focus_sweep_batch16_roll8_p0p6_k1_steps100_evalfull50_v6e8.yaml
  projects/gsm8k_grpo/configs/token_focus_sweep_steps100/grpo_gsm8k_qwen25_3b_token_focus_sweep_batch16_roll8_p0p6_k5_steps100_evalfull50_v6e8.yaml
  projects/gsm8k_grpo/configs/token_focus_sweep_steps100/grpo_gsm8k_qwen25_3b_token_focus_sweep_batch16_roll8_p0p6_k10_steps100_evalfull50_v6e8.yaml
  projects/gsm8k_grpo/configs/token_focus_sweep_steps100/grpo_gsm8k_qwen25_3b_token_focus_sweep_batch16_roll8_p0p6_k20_steps100_evalfull50_v6e8.yaml
  projects/gsm8k_grpo/configs/token_focus_sweep_steps100/grpo_gsm8k_qwen25_3b_token_focus_sweep_batch16_roll8_p0p8_k1_steps100_evalfull50_v6e8.yaml
  projects/gsm8k_grpo/configs/token_focus_sweep_steps100/grpo_gsm8k_qwen25_3b_token_focus_sweep_batch16_roll8_p0p8_k5_steps100_evalfull50_v6e8.yaml
  projects/gsm8k_grpo/configs/token_focus_sweep_steps100/grpo_gsm8k_qwen25_3b_token_focus_sweep_batch16_roll8_p0p8_k10_steps100_evalfull50_v6e8.yaml
  projects/gsm8k_grpo/configs/token_focus_sweep_steps100/grpo_gsm8k_qwen25_3b_token_focus_sweep_batch16_roll8_p0p8_k20_steps100_evalfull50_v6e8.yaml
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
  python -u projects/gsm8k_grpo/scripts/run_train.py --config "$cfg"
done

