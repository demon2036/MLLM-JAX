#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Validate plugins2 GRPO observability web app on TPU VM with retry-on-preemption.

Usage:
  bash scripts/tpu_vm_validate_plugins2_observability_retry.sh \
    --config <yaml> [--type v4-8] [--zone us-central2-b] [--project <gcp-project>] \
    [--git-ref <branch-or-sha>] [--max-attempts 0]

Defaults:
  --config projects/plugins2_grpo_observability/configs/plugins2_grpo_observability_gsm8k_qwen25_0p5b_v4_8_no_wandb.yaml
  --type v4-8
  --zone us-central2-b
  --git-ref <current local branch>
  --max-attempts 0  (0 means infinite retry until success)

Result artifacts (local):
  memory/20260208_plugins2_grpo_tpu_observability/evidence/
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

CONFIG_PATH="projects/plugins2_grpo_observability/configs/plugins2_grpo_observability_gsm8k_qwen25_0p5b_v4_8_no_wandb.yaml"
TPU_TYPE="v4-8"
ZONE="us-central2-b"
PROJECT=""
GIT_REF="$(git rev-parse --abbrev-ref HEAD)"
MAX_ATTEMPTS="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="${2:-}"; shift 2 ;;
    --type)
      TPU_TYPE="${2:-}"; shift 2 ;;
    --zone)
      ZONE="${2:-}"; shift 2 ;;
    --project)
      PROJECT="${2:-}"; shift 2 ;;
    --git-ref)
      GIT_REF="${2:-}"; shift 2 ;;
    --max-attempts)
      MAX_ATTEMPTS="${2:-}"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

if [[ -z "$PROJECT" ]]; then
  PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
fi
if [[ -z "$PROJECT" ]]; then
  echo "Cannot detect gcloud project; pass --project" >&2
  exit 1
fi

EVID_DIR="memory/20260208_plugins2_grpo_tpu_observability/evidence"
mkdir -p "$EVID_DIR"

attempt=1
while :; do
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  tpu_name="mllm-jax-plugins2-${TPU_TYPE}-${ts,,}"
  log_prefix="$EVID_DIR/attempt${attempt}_${ts}"

  echo "[attempt $attempt] creating TPU $tpu_name ($TPU_TYPE, $ZONE, project=$PROJECT)"

  cleanup() {
    echo "[attempt $attempt] deleting TPU $tpu_name"
    bash scripts/delete_tpu_vm.sh --name "$tpu_name" --zone "$ZONE" --project "$PROJECT" || true
  }

  set +e
  bash scripts/create_tpu_vm.sh --type "$TPU_TYPE" --zone "$ZONE" --project "$PROJECT" --name "$tpu_name" >"${log_prefix}_create.log" 2>&1
  create_ec=$?
  set -e

  if [[ $create_ec -ne 0 ]]; then
    echo "[attempt $attempt] create failed, retrying"
    if [[ "$MAX_ATTEMPTS" != "0" && $attempt -ge $MAX_ATTEMPTS ]]; then
      echo "Reached max attempts ($MAX_ATTEMPTS)" >&2
      exit 1
    fi
    attempt=$((attempt + 1))
    continue
  fi

  run_ec=0
  {
    bash scripts/bootstrap_miniconda_on_tpu_vm.sh --name "$tpu_name" --zone "$ZONE" --project "$PROJECT"

    bash scripts/ssh_tpu_vm_root.sh --name "$tpu_name" --zone "$ZONE" --project "$PROJECT" --command '
      set -euo pipefail
      REPO_URL="https://github.com/demon2036/MLLM-JAX.git"
      REPO_DIR="/root/MLLM-JAX"
      if [ ! -d "$REPO_DIR/.git" ]; then
        rm -rf "$REPO_DIR"
        git clone "$REPO_URL" "$REPO_DIR"
      fi
      cd "$REPO_DIR"
      git fetch --all --prune
      git checkout "'"$GIT_REF"'"
      git pull --ff-only || true
      git status -sb
      git rev-parse HEAD
    '

    bash scripts/ssh_tpu_vm_root.sh --name "$tpu_name" --zone "$ZONE" --project "$PROJECT" --command '
      set -euo pipefail
      rm -f /tmp/libtpu_lockfile || true
      source /root/miniconda3/etc/profile.d/conda.sh
      conda activate mllm-jax
      python -m pip install -U pip
      python -m pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
      python -m pip install -U torch --index-url https://download.pytorch.org/whl/cpu
      cd /root/MLLM-JAX
      python -m pip install -U -r requirements-tpu.txt
      python - <<"PY"
import jax, jaxlib
print("backend", jax.default_backend())
print("process", jax.process_index(), jax.process_count())
print("devices", jax.device_count(), jax.local_device_count())
print("jax", jax.__version__, "jaxlib", jaxlib.__version__)
PY
    '

    bash scripts/ssh_tpu_vm_root.sh --name "$tpu_name" --zone "$ZONE" --project "$PROJECT" --command '
      set -euo pipefail
      rm -f /tmp/libtpu_lockfile || true
      source /root/miniconda3/etc/profile.d/conda.sh
      conda activate mllm-jax
      cd /root/MLLM-JAX
      nohup python -u projects/plugins2_grpo_observability/scripts/run_web.py \
        --config '"$CONFIG_PATH"' \
        > /tmp/plugins2_web.log 2>&1 &
      echo $! > /tmp/plugins2_web.pid

      for i in $(seq 1 180); do
        if curl -fsS http://127.0.0.1:8080/api/healthz >/tmp/plugins2_health.json 2>/dev/null; then
          break
        fi
        sleep 2
      done

      curl -fsS http://127.0.0.1:8080/ >/tmp/plugins2_index.html
      curl -fsS -H "Content-Type: application/json" \
        -d "{\"system_prompt\":\"You are a helpful math assistant.\",\"user_prompt\":\"If Alice has 3 apples and buys 5 more, how many apples does she have?\",\"label\":\"8\",\"k\":8}" \
        http://127.0.0.1:8080/api/run >/tmp/plugins2_run.json

      python - <<"PY"
import json
from pathlib import Path
payload = json.loads(Path('/tmp/plugins2_run.json').read_text())
assert 'summary' in payload, 'missing summary'
assert 'samples' in payload and len(payload['samples']) == 8, 'expected 8 samples'
for sample in payload['samples']:
    assert 'tokens' in sample and len(sample['tokens']) > 0, 'sample has no token rows'
print('validated_samples=', len(payload['samples']))
print('loss_before_update=', payload['summary'].get('loss_before_update'))
print('grad_l2=', payload['summary'].get('grad_l2'))
PY

      kill $(cat /tmp/plugins2_web.pid) || true
      sleep 1
      pkill -f "projects/plugins2_grpo_observability/scripts/run_web.py" || true

      echo "==health=="
      cat /tmp/plugins2_health.json
      echo
      echo "==summary=="
      python - <<"PY"
import json
from pathlib import Path
payload = json.loads(Path('/tmp/plugins2_run.json').read_text())
print(json.dumps(payload.get('summary', {}), ensure_ascii=False, indent=2))
PY
    ' >"${log_prefix}_run.log" 2>&1

    bash scripts/ssh_tpu_vm_root.sh --name "$tpu_name" --zone "$ZONE" --project "$PROJECT" --command '
      set -euo pipefail
      cat /tmp/plugins2_run.json
    ' >"${log_prefix}_run_response.json" 2>&1

    bash scripts/ssh_tpu_vm_root.sh --name "$tpu_name" --zone "$ZONE" --project "$PROJECT" --command '
      set -euo pipefail
      cat /tmp/plugins2_web.log | tail -n 200
    ' >"${log_prefix}_web.log" 2>&1
  } || run_ec=$?

  cleanup

  if [[ $run_ec -eq 0 ]]; then
    echo "[attempt $attempt] validation succeeded"
    echo "Evidence: ${log_prefix}_run.log and ${log_prefix}_run_response.json"
    exit 0
  fi

  echo "[attempt $attempt] validation failed with exit_code=$run_ec, retrying"
  if [[ "$MAX_ATTEMPTS" != "0" && $attempt -ge $MAX_ATTEMPTS ]]; then
    echo "Reached max attempts ($MAX_ATTEMPTS)" >&2
    exit "$run_ec"
  fi
  attempt=$((attempt + 1))
  sleep 5
done
