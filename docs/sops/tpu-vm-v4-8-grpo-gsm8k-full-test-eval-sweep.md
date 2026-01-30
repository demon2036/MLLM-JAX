# TPU VM v4-8 GRPO/GSM8K full test-set eval sweep (Qwen2.5-3B, W&B)

- **Title**: SOP: Run GRPO (100 steps) + full GSM8K `test` split eval (1319 Qs) on TPU `v4-8` with W&B online
  **Prereqs**: `gcloud` configured; TPU quota/capacity in target zone; WANDB API key available (stored in local `.env`, synced to TPU as `/root/.env`)
  **Environment (verified)**:
  - TPU VM: `mllm-jax-gsm8k-grpo-full-eval-v4-8-spot-260130085243` (`v4-8`, spot), zone `us-central2-b`, project `civil-rarity-482610-s5`
  - Python: `3.12.12`
  - JAX/JAXLIB: `0.9.0` / `0.9.0` (backend `tpu`, `device_count=4`)
  - Branch: `john/remove-shims-split-gsm8k-grpo-20260130`
  - Commit: `e3a2cf0`
  - W&B run: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/vfyynzmy

## Goal

Run an end-to-end GRPO training loop and then compute stable **full test-set** GSM8K accuracy (1319 questions, each once), logged as:
- `eval_full/accuracy`
- `eval_full/questions_global` (expected `1319`)
- `eval_full/samples_per_question` (expected `1`)

## Steps (commands actually used)

### 0) Create a fresh TPU VM (spot)

```bash
cd /root/github/MLLM-JAX-worktrees/nano-gpt
TPU_NAME=mllm-jax-gsm8k-grpo-full-eval-v4-8-spot-$(date +%y%m%d%H%M%S)
ZONE=us-central2-b
scripts/create_tpu_vm.sh --type v4-8 --zone "$ZONE" --spot --name "$TPU_NAME"
```

### 1) Bootstrap Miniconda + create env

```bash
scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --command \
  'set -euo pipefail; if [ ! -d /root/miniconda3 ]; then curl -fsSL -o /root/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; bash /root/miniconda.sh -b -p /root/miniconda3; rm -f /root/miniconda.sh; fi; /root/miniconda3/bin/conda --version'

scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --command \
  'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true; conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true; ENV_NAME=mllm-jax; if ! conda env list | awk "{print \\$1}" | grep -qx "$ENV_NAME"; then conda create -y -n "$ENV_NAME" python=3.12; fi; conda activate "$ENV_NAME"; python --version; pip install -U pip'
```

### 2) Git-sync repo on TPU (no SCP for code)

```bash
scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --command \
  'set -euo pipefail; if [ ! -d /root/MLLM-JAX/.git ]; then rm -rf /root/MLLM-JAX; git clone https://github.com/demon2036/MLLM-JAX.git /root/MLLM-JAX; fi; cd /root/MLLM-JAX; git fetch origin; git checkout john/remove-shims-split-gsm8k-grpo-20260130; git reset --hard origin/john/remove-shims-split-gsm8k-grpo-20260130; git rev-parse --short HEAD'
```

Expected: prints `e3a2cf0`.

### 3) Install TPU runtime deps

```bash
scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --command \
  'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; pip install -U -r requirements-tpu.txt'
```

### 4) Sync W&B key to TPU (secret-only SCP)

```bash
cd /root/github/MLLM-JAX-worktrees/nano-gpt
scripts/sync_env_to_tpu_vm.sh --name "$TPU_NAME" --zone "$ZONE" --src .env --dest /root/.env --worker all
```

### 5) Launch GRPO + full test eval sweep (via nohup)

```bash
scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --command \
  'set -euo pipefail; cd /root/MLLM-JAX; set -a; source /root/.env; set +a; export EVAL_FULL_SWEEP=1; bash scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh --env-name mllm-jax --config projects/gsm8k_grpo/configs/rl_gsm8k_qwen25_3b_v4_8_grpo_steps100_full_eval.yaml'
```

### 6) Verify completion + full-sweep output

```bash
scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --command \
  'set -euo pipefail; cd /root/MLLM-JAX; cat logs/nohup_rl_gsm8k_qwen25_3b_v4_8_grpo_steps100_full_eval_latest.exit; grep -n "^eval_full " logs/nohup_rl_gsm8k_qwen25_3b_v4_8_grpo_steps100_full_eval_latest.log'
```

Expected:
- Exit file contains `0`.
- Log contains a line like:
  - `eval_full split=test questions=1319 samples_per_question=1 accuracy=... t=...s`

### 7) Delete TPU VM (stop billing)

```bash
cd /root/github/MLLM-JAX-worktrees/nano-gpt
scripts/delete_tpu_vm.sh --name "$TPU_NAME" --zone "$ZONE"
```

## Expected Result (Run `vfyynzmy`)

- `eval_full split=test questions=1319 samples_per_question=1 accuracy=0.6285 t=491.15s`
- Exit code: `0`
- W&B: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/vfyynzmy
