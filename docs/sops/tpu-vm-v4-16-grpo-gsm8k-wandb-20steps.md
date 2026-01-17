# TPU VM v4-16 (2-host) GRPO/GSM8K 20-step Train with W&B (sync `.env` to all workers)

- **Title**: SOP: Run `GRPO + GSM8K` for 20 steps on a 2-host TPU VM (v4-16) with W&B logging via `.env`
  **Prereqs**: `gcloud` installed + authenticated; TPU v4 capacity/quota; outbound internet from TPU VM (HF + datasets + wandb); local changes pushed to GitHub; local `.env` contains `WANDB_API_KEY` and is gitignored
  **Environment (verified)**:
  - TPU VM: `v4-16` (2 hosts), zone `us-central2-b`
  - Conda env: `mllm-jax` (Python `3.12.12`)
  - JAX: `0.8.2`, jaxlib `0.8.2`, libtpu `0.0.32`
  - JAX process count: `2`, device count: `8` (megacore), local devices per host: `4`

## Steps (commands actually used)

### 0) Prepare local secrets (never commit)

- Ensure repo ignores `.env` (this repo already does).
- Create local `.env` (example template: `.env.example`) with:
  - `WANDB_API_KEY=...`

### 1) Push code to GitHub (TPU pulls from Git)

- `cd /home/john/github/MLLM-JAX`
- `git status -sb`
- `git rev-parse --short HEAD`
- `git push origin main`

### 2) Sync repo on all TPU workers (reset local edits)

- `TPU_NAME=mllm-jax-v4-16-260117125029; ZONE=us-central2-b; PROJECT=civil-rarity-482610-s5`
- `cd /home/john/github/MLLM-JAX`
- `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --project "$PROJECT" --worker all --command 'set -euo pipefail; REPO_DIR=/root/MLLM-JAX; cd "$REPO_DIR"; git fetch --all --prune; git checkout main; git reset --hard origin/main; git clean -fd; echo "HEAD=$(git rev-parse --short HEAD)"; git status -sb'`

### 3) Copy `.env` to all workers (via `gcloud tpu-vm scp`)

- `cd /home/john/github/MLLM-JAX`
- `scripts/sync_env_to_tpu_vm.sh --name "$TPU_NAME" --zone "$ZONE" --project "$PROJECT" --worker all`
  - Default destination: `/root/.env` on each worker.

### 4) Bootstrap conda env on all workers

- `cd /home/john/github/MLLM-JAX`
- `scripts/bootstrap_miniconda_on_tpu_vm.sh --name "$TPU_NAME" --zone "$ZONE" --project "$PROJECT" --worker all --env-name mllm-jax --python 3.12`

### 5) Install runtime deps on all workers

- `cd /home/john/github/MLLM-JAX`
- `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --project "$PROJECT" --worker all --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; pip install -U pip; pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; pip install -U torch --index-url https://download.pytorch.org/whl/cpu; cd /root/MLLM-JAX; pip install -U -r requirements-tpu.txt; python - <<\"PY\"\nimport jax, jaxlib\nprint(\"jax\", jax.__version__, \"jaxlib\", jaxlib.__version__)\nprint(\"backend\", jax.default_backend())\nprint(\"process\", jax.process_index(), \"/\", jax.process_count())\nprint(\"device_count\", jax.device_count(), \"local\", len(jax.local_devices()))\nPY'`

### 6) Run 20-step GRPO/GSM8K training with W&B (all workers; only process 0 logs)

- `cd /home/john/github/MLLM-JAX`
- `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --project "$PROJECT" --worker all --env-file /root/.env --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; export HF_HUB_ENABLE_HF_TRANSFER=1; export TOKENIZERS_PARALLELISM=false; export WANDB_MODE=online; export WANDB_PROJECT=mllm-jax-grpo-gsm8k; export WANDB_NAME=grpo_gsm8k_v4-16_56a8a9d_steps20; export MODEL_PATH=\"Qwen/Qwen2.5-7B-Instruct\"; export STEPS=20; export BATCH_SIZE=1; export NUM_PRE_Q=8; export GLOBAL_LENGTH=512; export MAX_LENGTH_SAMPLE=64; export PPO_EPOCHS=1; export GRAD_ACCUM_STEPS=1; export BETA=0.0; python -u scripts/run_grpo_gsm8k_training.py'`

## Expected Result

- TPU logs show `step=0 ...` through `step=19 ...` and exit with code `0`.
- W&B shows a single run under project `mllm-jax-grpo-gsm8k` with logged metrics (`loss`, `entropy`, `reward_mean`, `dt`, ...).

## Troubleshooting

- `git pull` fails due to local modifications: use the reset step in (2).
- `wandb` shows “not logged in”: ensure `.env` is synced to `/root/.env` on all workers and `--env-file /root/.env` is passed.
- `jax.distributed.initialize()` hangs: ensure you run with `--worker all` (both hosts must launch).

## References

- `scripts/sync_env_to_tpu_vm.sh`
- `scripts/ssh_tpu_vm_root.sh`
- `scripts/bootstrap_miniconda_on_tpu_vm.sh`
- `scripts/run_grpo_gsm8k_training.py`
- `plugins/training/runner/grpo_gsm8k.py`

