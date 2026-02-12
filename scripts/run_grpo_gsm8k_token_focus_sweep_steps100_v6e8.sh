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

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  if [[ -f "./.env" ]]; then
    set -a
    source "./.env"
    set +a
  elif [[ -f "/root/.env" ]]; then
    set -a
    source "/root/.env"
    set +a
  fi
fi

ENTITY="${WANDB_ENTITY:-johntitordemon2036}"
PROJECT="mllm-jax-grpo-gsm8k-tokenfocus-sweep-steps100"
FILTER_SUBSTR="projects/gsm8k_grpo/configs/token_focus_sweep_steps100/"

finished_configs=""
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "Querying W&B for finished runs to resume sweep..."
  set +e
  finished_configs="$(python -u scripts/wandb_list_finished_runs_config_paths.py \
    --entity "$ENTITY" \
    --project "$PROJECT" \
    --filter-config-substr "$FILTER_SUBSTR" 2>/dev/null)"
  status="$?"
  set -e
  if [[ "$status" -ne 0 ]]; then
    echo "WARNING: failed to query W&B finished runs; sweep will run all configs."
    finished_configs=""
  else
    finished_count="$(printf '%s\n' "$finished_configs" | grep -c '.' || true)"
    echo "wandb_entity=$ENTITY wandb_project=$PROJECT finished_configs=$finished_count"
  fi
else
  echo "WANDB_API_KEY not set; sweep will run all configs."
fi

for cfg in "${configs[@]}"; do
  if [[ -n "$finished_configs" ]] && printf '%s\n' "$finished_configs" | grep -Fxq "$cfg"; then
    echo
    echo "============================================================"
    echo "SKIP (already finished): $cfg"
    echo "============================================================"
    continue
  fi
  echo
  echo "============================================================"
  echo "RUN: $cfg"
  echo "============================================================"
  python -u projects/gsm8k_grpo/scripts/run_train.py --config "$cfg"
done
