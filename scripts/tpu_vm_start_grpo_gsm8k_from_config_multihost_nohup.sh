#!/usr/bin/env bash
set -euo pipefail

# Start GRPO/GSM8K training from a YAML config via `nohup` on a multi-host TPU VM.
#
# IMPORTANT:
# - Run this script on *all* TPU workers (e.g. `gcloud ... --worker=all`) so
#   `jax.distributed.initialize()` can form the multi-host JAX runtime.
# - This wrapper sets `REQUIRE_MULTIHOST=1` so the training runner will fail
#   fast if only one worker was launched.
#
# Environment variables (optional overrides):
# - REQUIRE_MULTIHOST (default: 1; require jax.process_count()>1)
# - REQUIRE_JAX_PROCESS_COUNT (optional; require an exact process_count)
# - ENV_NAME / CONFIG_PATH / WANDB_* / PY_ARGS (same as the underlying script)

export REQUIRE_MULTIHOST="${REQUIRE_MULTIHOST:-1}"

bash scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh
