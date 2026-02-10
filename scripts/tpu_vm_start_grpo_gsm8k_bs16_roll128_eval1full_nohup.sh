#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Start GRPO (bs=16, n=128, eval_rollout_n=1 + full eval sweep) via nohup.

Usage:
  bash scripts/tpu_vm_start_grpo_gsm8k_bs16_roll128_eval1full_nohup.sh [--env-name mllm-jax]
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ENV_NAME="mllm-jax"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="${2:-}"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

bash scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh \
  --env-name "$ENV_NAME" \
  --config projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_batch16_roll128_eval1full_v6e8.yaml
