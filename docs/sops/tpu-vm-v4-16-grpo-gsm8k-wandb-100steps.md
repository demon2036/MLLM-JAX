# TPU VM v4-16 (2-host) GRPO/GSM8K 100-step Train with W&B (sync `.env` to all workers)

- **Title**: SOP: Run `GRPO + GSM8K` for 100 steps on a 2-host TPU VM (v4-16) with W&B logging via `.env`
  **Prereqs**: TPU VM already bootstrapped (see `docs/sops/tpu-vm-v4-16-grpo-gsm8k-wandb-20steps.md`); outbound internet from TPU VM (HF + datasets + wandb); local changes pushed to GitHub; local `.env` contains `WANDB_API_KEY` and is gitignored
  **Environment (verified)**:
  - TPU VM: `mllm-jax-v4-16-260117125029` (`v4-16`, 2 hosts), zone `us-central2-b`, project `civil-rarity-482610-s5`
  - Conda env: `mllm-jax`
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

### 4) Run 100-step GRPO/GSM8K training + eval with W&B (all workers; only process 0 logs)

- `cd /home/john/github/MLLM-JAX`
- `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --project "$PROJECT" --worker all --env-file /root/.env --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; export HF_HUB_ENABLE_HF_TRANSFER=1; export TOKENIZERS_PARALLELISM=false; export WANDB_MODE=online; export WANDB_PROJECT=mllm-jax-grpo-gsm8k; export WANDB_NAME=grpo_gsm8k_v4-16_c7e0f7e_steps100_eval10_20260117_143424; export MODEL_PATH=\"Qwen/Qwen2.5-7B-Instruct\"; export STEPS=100; export BATCH_SIZE=1; export NUM_PRE_Q=8; export GLOBAL_LENGTH=512; export MAX_LENGTH_SAMPLE=64; export PPO_EPOCHS=1; export GRAD_ACCUM_STEPS=1; export BETA=0.0; export EVAL_EVERY_STEPS=10; export EVAL_BATCHES=1; export EVAL_SPLIT=test; python -u scripts/run_grpo_gsm8k_training.py'`

## Expected Result

- TPU logs show `step=0 ...` through `step=99 ...` and exit with code `0`.
- W&B shows a single run under project `mllm-jax-grpo-gsm8k` with logged metrics + timings.
  - Run URL (from this 100-step verification run): `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/gxaov98z`

## Troubleshooting

- `wandb` shows “not logged in”: ensure `.env` is synced to `/root/.env` on all workers and `--env-file /root/.env` is passed.
- `jax.distributed.initialize()` hangs: ensure you run with `--worker all` (both hosts must launch).
- TPU busy / libtpu lock errors: try `rm -f /tmp/libtpu_lockfile` before running.

## References

- `docs/sops/tpu-vm-v4-16-grpo-gsm8k-wandb-20steps.md`
- `scripts/sync_env_to_tpu_vm.sh`
- `scripts/ssh_tpu_vm_root.sh`
- `scripts/run_grpo_gsm8k_training.py`
- `plugins/training/runner/grpo_gsm8k.py`
