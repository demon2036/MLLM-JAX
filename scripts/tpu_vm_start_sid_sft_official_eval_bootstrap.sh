#!/bin/sh
set -eu

LOG_FILE="/var/log/tpu_startup_sid_sft_official_eval_bootstrap.log"
exec >"$LOG_FILE" 2>&1

BOOT_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ || date)"
mkdir -p /var/lib/google/guest-attributes/sid_sft_bootstrap 2>/dev/null || true
printf "%s" "bootstrap_begin ${BOOT_TS}" > /var/lib/google/guest-attributes/sid_sft_bootstrap/status 2>/dev/null || true

if ! command -v curl >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update
    apt-get install -y curl
  fi
fi

SCRIPT_URL="https://raw.githubusercontent.com/demon2036/MLLM-JAX/john/sid3-fixed-prefill-single-compile/scripts/tpu_vm_start_sid_sft_official_eval_startup.sh"
DEST="/root/tpu_start_sid_sft_official_eval.sh"

if command -v curl >/dev/null 2>&1; then
  curl -fsSL "$SCRIPT_URL" -o "$DEST"
else
  printf "%s" "bootstrap_missing_curl" > /var/lib/google/guest-attributes/sid_sft_bootstrap/status 2>/dev/null || true
  exit 1
fi

if [ -x /bin/bash ]; then
  /bin/bash "$DEST"
elif command -v bash >/dev/null 2>&1; then
  bash "$DEST"
else
  printf "%s" "bootstrap_missing_bash" > /var/lib/google/guest-attributes/sid_sft_bootstrap/status 2>/dev/null || true
  exit 1
fi
