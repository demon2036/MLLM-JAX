# TPU VM v4-16 (2-host) GRPO/GSM8K OOM Sweep @ `max_length=2048` (Qwen2.5-7B) with W&B

- **Title**: SOP: Find the largest GRPO batch that does not OOM at `max_length=2048` on a 2-host TPU VM (v4-16), model `Qwen/Qwen2.5-7B-Instruct`
  **Prereqs**: `docs/sops/tpu-vm-v4-16-grpo-gsm8k-wandb-100steps.md` already works; `/root/.env` synced to all workers (contains `WANDB_API_KEY`); outbound internet from TPU VM
  **Environment (verified)**:
  - TPU VM: `mllm-jax-v4-16-260117125029` (`v4-16`, 2 hosts), zone `us-central2-b`, project `civil-rarity-482610-s5`
  - Repo commit on TPU: `8c45109`
  - Model: `Qwen/Qwen2.5-7B-Instruct`
  - JAX process count: `2`, device count: `8` (megacore), local devices per host: `4`

## Key idea (how to interpret “batch size” here)

This runner’s true memory driver is the **effective local batch**:

- `local_batch = batch_size * num_pre_q`

Because `MLLM_JAX.utils._form_global_array()` splits the local batch across local devices, we must also satisfy:

- `local_batch % local_device_count == 0` (here: `local_device_count=4`), so `local_batch` must be a multiple of 4.

## What we tested (commands actually used)

### 0) Ensure TPU repo is on latest `main`

- `TPU_NAME=mllm-jax-v4-16-260117125029; ZONE=us-central2-b; PROJECT=civil-rarity-482610-s5`
- `cd /home/john/github/MLLM-JAX`
- `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --project "$PROJECT" --worker all --command 'set -euo pipefail; cd /root/MLLM-JAX; git fetch --all --prune; git checkout main; git reset --hard origin/main; git clean -fd; echo \"HEAD=$(git rev-parse --short HEAD)\"; git status -sb'`

### 1) Sync `.env` to all workers (WANDB_API_KEY)

- `cd /home/john/github/MLLM-JAX`
- `scripts/sync_env_to_tpu_vm.sh --name "$TPU_NAME" --zone "$ZONE" --project "$PROJECT" --worker all`

### 2) OOM case (fails): `GLOBAL_LENGTH=2048`, `MAX_LENGTH_TOTAL=2048`, `BATCH_SIZE=1`, `NUM_PRE_Q=8`

- `cd /home/john/github/MLLM-JAX`
- `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --project "$PROJECT" --worker all --env-file /root/.env --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; mkdir -p workdir/logs/oom_sweep_len2048; export HF_HUB_ENABLE_HF_TRANSFER=1; export TOKENIZERS_PARALLELISM=false; export WANDB_MODE=online; export WANDB_PROJECT=mllm-jax-grpo-gsm8k; export WANDB_GROUP=oom_sweep_len2048_20260117; export WANDB_JOB_TYPE=oom_sweep; export RUN_TS=$(date -u +%Y%m%d_%H%M%S); export COMMIT=$(git rev-parse --short HEAD); export MODEL_PATH=\"Qwen/Qwen2.5-7B-Instruct\"; export STEPS=2; export BATCH_SIZE=1; export NUM_PRE_Q=8; export GLOBAL_LENGTH=2048; export MAX_LENGTH_TOTAL=2048; export MAX_LENGTH_SAMPLE=64; export PPO_EPOCHS=1; export GRAD_ACCUM_STEPS=1; export BETA=0.0; export EVAL_EVERY_STEPS=0; export WANDB_NAME=grpo_gsm8k_v4-16_len2048_bs${BATCH_SIZE}_npq${NUM_PRE_Q}_${COMMIT}_${RUN_TS}; set +e; python -u scripts/run_grpo_gsm8k_training.py 2>&1 | tee workdir/logs/oom_sweep_len2048/bs${BATCH_SIZE}_${RUN_TS}.log; status=${PIPESTATUS[0]}; set -e; echo \"exit_status=${status}\"; tail -n 40 workdir/logs/oom_sweep_len2048/bs${BATCH_SIZE}_${RUN_TS}.log; exit ${status}'`

Observed:

- Result: **OOM / non-zero exit**
- Error: `RESOURCE_EXHAUSTED: ... Ran out of memory in memory space hbm ...`
- W&B run: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/rgqoyxda`

### 3) Safe case (passes): `GLOBAL_LENGTH=2048`, `MAX_LENGTH_TOTAL=2048`, `BATCH_SIZE=1`, `NUM_PRE_Q=4`

2-step smoke:

- `cd /home/john/github/MLLM-JAX`
- `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --project "$PROJECT" --worker all --env-file /root/.env --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; mkdir -p workdir/logs/oom_sweep_len2048; export HF_HUB_ENABLE_HF_TRANSFER=1; export TOKENIZERS_PARALLELISM=false; export WANDB_MODE=online; export WANDB_PROJECT=mllm-jax-grpo-gsm8k; export WANDB_GROUP=oom_sweep_len2048_20260117; export WANDB_JOB_TYPE=oom_sweep; export RUN_TS=$(date -u +%Y%m%d_%H%M%S); export COMMIT=$(git rev-parse --short HEAD); export MODEL_PATH=\"Qwen/Qwen2.5-7B-Instruct\"; export STEPS=2; export BATCH_SIZE=1; export NUM_PRE_Q=4; export GLOBAL_LENGTH=2048; export MAX_LENGTH_TOTAL=2048; export MAX_LENGTH_SAMPLE=64; export PPO_EPOCHS=1; export GRAD_ACCUM_STEPS=1; export BETA=0.0; export EVAL_EVERY_STEPS=0; export WANDB_NAME=grpo_gsm8k_v4-16_len2048_bs${BATCH_SIZE}_npq${NUM_PRE_Q}_${COMMIT}_${RUN_TS}; set +e; python -u scripts/run_grpo_gsm8k_training.py 2>&1 | tee workdir/logs/oom_sweep_len2048/bs${BATCH_SIZE}_npq${NUM_PRE_Q}_${RUN_TS}.log; status=${PIPESTATUS[0]}; set -e; echo \"exit_status=${status}\"; tail -n 40 workdir/logs/oom_sweep_len2048/bs${BATCH_SIZE}_npq${NUM_PRE_Q}_${RUN_TS}.log; exit ${status}'`

Observed:

- Result: **OK / exit 0**
- W&B run: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/tvy77hov`

20-step confirmation:

- `cd /home/john/github/MLLM-JAX`
- `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --project "$PROJECT" --worker all --env-file /root/.env --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; mkdir -p workdir/logs/oom_sweep_len2048; export HF_HUB_ENABLE_HF_TRANSFER=1; export TOKENIZERS_PARALLELISM=false; export WANDB_MODE=online; export WANDB_PROJECT=mllm-jax-grpo-gsm8k; export WANDB_GROUP=oom_sweep_len2048_20260117; export WANDB_JOB_TYPE=oom_confirm; export RUN_TS=$(date -u +%Y%m%d_%H%M%S); export COMMIT=$(git rev-parse --short HEAD); export MODEL_PATH=\"Qwen/Qwen2.5-7B-Instruct\"; export STEPS=20; export BATCH_SIZE=1; export NUM_PRE_Q=4; export GLOBAL_LENGTH=2048; export MAX_LENGTH_TOTAL=2048; export MAX_LENGTH_SAMPLE=64; export PPO_EPOCHS=1; export GRAD_ACCUM_STEPS=1; export BETA=0.0; export EVAL_EVERY_STEPS=0; export WANDB_NAME=grpo_gsm8k_v4-16_len2048_confirm_bs${BATCH_SIZE}_npq${NUM_PRE_Q}_steps${STEPS}_${COMMIT}_${RUN_TS}; set +e; python -u scripts/run_grpo_gsm8k_training.py 2>&1 | tee workdir/logs/oom_sweep_len2048/confirm_bs${BATCH_SIZE}_npq${NUM_PRE_Q}_steps${STEPS}_${RUN_TS}.log; status=${PIPESTATUS[0]}; set -e; echo \"exit_status=${status}\"; tail -n 40 workdir/logs/oom_sweep_len2048/confirm_bs${BATCH_SIZE}_npq${NUM_PRE_Q}_steps${STEPS}_${RUN_TS}.log; exit ${status}'`

Observed:

- Result: **OK / exit 0**
- W&B run: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/adcws9h3`

## Result summary (what “max batch size” means)

Under:

- `MODEL_PATH=Qwen/Qwen2.5-7B-Instruct`
- `GLOBAL_LENGTH=2048` (rollout prefill bucket becomes 2048)
- `MAX_LENGTH_TOTAL=2048`
- decode bucket is `128`, so the model sees sequence length `2048 + 128 = 2176`
- TPU: v4-16 (2 hosts) → global devices `8`, local devices per host `4`

We observed:

- **Max safe**: `local_batch = batch_size * num_pre_q = 4` (global batch = 8, per-device batch = 1)
- **First OOM**: `local_batch = 8` (global batch = 16, per-device batch = 2)

So if you insist on `num_pre_q=8`, then `batch_size=1` already implies `local_batch=8` → **OOM** at length 2048 on this setup.

## References

- `scripts/run_grpo_gsm8k_training.py`
- `plugins/training/runner/grpo_gsm8k.py`
- `docs/sops/tpu-vm-v4-16-grpo-gsm8k-wandb-100steps.md`
