# SOP: TPU VM v4-8 run RL/GSM8K `reinforce++` (100 steps, W&B online)

- **Title**: SOP: TPU VM v4-8 run RL/GSM8K `reinforce++` (100 steps, W&B online)
  **Prereqs**: `gcloud` installed + authenticated; TPU quota/capacity; outbound internet from TPU VM (HF + datasets); repo pushed to GitHub; TPU has `/root/.env` containing `WANDB_API_KEY`
  **Environment (verified)**:
  - Project: `civil-rarity-482610-s5`
  - Zone: `us-central2-b`
  - TPU VM: `mllm-jax-v4-8-260122100610` (`v4-8`, single host)
  - Conda env: `sglang-jax`; Python `3.12.12`
  - JAX: `0.8.1`, jaxlib `0.8.1`
  - JAX devices: `device_count=4` (megacore)
  - Repo ref on TPU (verified): `f8e7cd0`
  - Config used: `plugins/training/configs/rl_gsm8k_qwen25_3b_v4_8_reinforcepp_steps100.yaml`

## Steps (commands actually used)

### 0) Confirm TPU VM state

- `gcloud alpha compute tpus tpu-vm describe mllm-jax-v4-8-260122100610 --project civil-rarity-482610-s5 --zone us-central2-b --format='value(state,health,acceleratorType)'`

### 1) Sync repo on TPU and checkout the commit

- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260122100610 --zone us-central2-b --project civil-rarity-482610-s5 --command 'set -euo pipefail; cd /root/MLLM-JAX; echo \"==remote==\"; git remote -v; echo \"==fetch==\"; git fetch --all --prune; echo \"==checkout==\"; git checkout f8e7cd0; git status -sb; echo \"HEAD=$(git rev-parse --short HEAD)\"'`

### 2) Verify TPU runtime + W&B key (without printing it)

- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260122100610 --zone us-central2-b --project civil-rarity-482610-s5 --env-file /root/.env --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; cd /root/MLLM-JAX; python - <<\"PY\"\nimport os, sys\nimport jax\nprint(\"python\", sys.version.split()[0])\nprint(\"jax_backend\", jax.default_backend())\nprint(\"WANDB_API_KEY_set=\", bool(os.environ.get(\"WANDB_API_KEY\")))\nPY'`

### 3) (Optional) Print resolved config on TPU (no JAX run)

- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260122100610 --zone us-central2-b --project civil-rarity-482610-s5 --env-file /root/.env --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; cd /root/MLLM-JAX; python -u scripts/run_grpo_gsm8k_training.py --print-config --config plugins/training/configs/rl_gsm8k_qwen25_3b_v4_8_reinforcepp_steps100.yaml | sed -n \"1,120p\"'`

### 4) Run 100 steps (W&B online + log file)

- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260122100610 --zone us-central2-b --project civil-rarity-482610-s5 --env-file /root/.env --command 'set -euo pipefail; cd /root/MLLM-JAX; CFG=plugins/training/configs/rl_gsm8k_qwen25_3b_v4_8_reinforcepp_steps100.yaml; LOG_DIR=/root/MLLM-JAX/logs; mkdir -p \"$LOG_DIR\"; RUN_TS=$(date -u +%Y%m%d_%H%M%S); COMMIT=$(git rev-parse --short HEAD); TAG=$(basename \"$CFG\" .yaml); LOG_FILE=\"$LOG_DIR/${TAG}_${COMMIT}_${RUN_TS}.log\"; LATEST=\"$LOG_DIR/${TAG}_latest.log\"; ln -sf \"$LOG_FILE\" \"$LATEST\"; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; export HF_HUB_ENABLE_HF_TRANSFER=1; export TOKENIZERS_PARALLELISM=false; echo \"CFG=$CFG\"; echo \"LOG_FILE=$LOG_FILE\"; set +e; python -u scripts/run_grpo_gsm8k_training.py --config \"$CFG\" 2>&1 | tee \"$LOG_FILE\"; status=${PIPESTATUS[0]}; set -e; echo \"exit_status=${status}\"; exit ${status}'`

### 5) Sanity-check the log for tracebacks

- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260122100610 --zone us-central2-b --project civil-rarity-482610-s5 --command 'set -euo pipefail; cd /root/MLLM-JAX; LOG=/root/MLLM-JAX/logs/rl_gsm8k_qwen25_3b_v4_8_reinforcepp_steps100_latest.log; echo \"LOG=$LOG\"; ls -la \"$LOG\"; echo \"==tail==\"; tail -n 25 \"$LOG\"; echo \"==traceback_grep==\"; if grep -n \"Traceback\" -n \"$LOG\" >/dev/null 2>&1; then echo \"traceback_found=1\"; grep -n \"Traceback\" -n \"$LOG\" | head; exit 1; else echo \"traceback_found=0\"; fi'`

## Expected Result

- The command in step (4) prints `exit_status=0`.
- The log contains `step=99 ...` (100 steps) and includes a W&B run URL.

## Results (this run)

- Commit on TPU: `f8e7cd0`
- Log symlink: `/root/MLLM-JAX/logs/rl_gsm8k_qwen25_3b_v4_8_reinforcepp_steps100_latest.log`
- Concrete log: `/root/MLLM-JAX/logs/rl_gsm8k_qwen25_3b_v4_8_reinforcepp_steps100_f8e7cd0_20260123_162452.log`
- W&B run: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/biv1ho6f`

## Troubleshooting

- `Unable to initialize backend 'tpu' ... already in use`:
  - `rm -f /tmp/libtpu_lockfile` and ensure no other training process is running.
- Script warns about deprecated env overrides (e.g. `WANDB_MODE`):
  - This repo ignores env-var hyperparameter overrides; prefer YAML configs.

