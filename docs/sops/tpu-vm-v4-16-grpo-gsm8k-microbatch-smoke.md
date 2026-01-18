# TPU VM v4-16 (2-host) GRPO/GSM8K micro-batched smoke run (rollout vs train configs, per-device batching)

- **Title**: SOP: Smoke-run `GRPO + GSM8K` on a 2-host TPU VM (v4-16) with separated rollout/train config (micro-batching)
  **Prereqs**: `gcloud` installed + authenticated; a reachable 2-host TPU VM; outbound internet from TPU VM (HF + datasets); local changes pushed to GitHub
  **Environment (verified)**:
  - Local: Ubuntu Linux; Python `3.12.2`
  - TPU VM: `mllm-jax-v4-16-260117125029` (`v4-16`, 2 hosts), zone `us-central2-b`, project `civil-rarity-482610-s5`
  - Conda env: `mllm-jax`; Python `3.12.12`
  - JAX: `0.8.2`, jaxlib `0.8.2`
  - JAX process count: `2`, global `device_count=8` (megacore), local devices per host: `4`
  - Git refs on TPU (verified): `821fe33`, `7002897`
  **What “verified” means here**:
  - Runs the full loop end-to-end (rollout → reward → advantages → PPO/GRPO update) for `STEPS=1` on `v4-16`.
  - Confirms rollout vs train batch size separation by running `BATCH_SIZE=2` with `TRAIN_MICRO_BATCH_SIZE=4` (runner prints effective `train.grad_accum_steps=4`).
  - Confirms per-device batching + auto-pass inference by running `ROLLOUT_GLOBAL_BATCH_SIZE=16` with `ROLLOUT_PER_DEVICE_BATCH_SIZE=1` (runner prints derived `rollout.batch_size=4` and `passes_per_step=2` on v4-16).
  - Confirms per-device micro-batching + auto grad-accum inference by running `TRAIN_PER_DEVICE_MICRO_BATCH_SIZE=1` (runner prints derived `train.micro_batch_size=4` and `train.grad_accum_steps=4` on v4-16).

## Steps (commands actually used)

### 0) Local preflight (no TPU required)

- `cd /home/john/github/MLLM-JAX`
- `python scripts/run_grpo_gsm8k_training.py --print-config`
- `python tests/test_jit8_schema_and_cli.py`
- `python -m compileall -q plugins scripts MLLM_JAX`

### 1) Push code to GitHub (TPU pulls from Git)

- `cd /home/john/github/MLLM-JAX`
- `git status -sb`
- `git rev-parse --short HEAD`
- `git push origin main`

### 2) Sync repo on all TPU workers (reset local edits)

- `cd /home/john/github/MLLM-JAX`
- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-16-260117125029 --zone us-central2-b --project civil-rarity-482610-s5 --worker all --command 'set -euo pipefail; REPO_DIR=/root/MLLM-JAX; if [ ! -d \"$REPO_DIR/.git\" ]; then rm -rf \"$REPO_DIR\"; git clone https://github.com/demon2036/MLLM-JAX.git \"$REPO_DIR\"; fi; cd \"$REPO_DIR\"; git fetch --all --prune; git checkout main; git reset --hard origin/main; git clean -fd; echo \"HEAD=$(git rev-parse --short HEAD)\"; git status -sb; python -V'`

### 3) Verify TPU JAX runtime (optional sanity check)

- `cd /home/john/github/MLLM-JAX`
- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-16-260117125029 --zone us-central2-b --project civil-rarity-482610-s5 --worker all --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; python - <<\"PY\"\nimport sys\nimport jax, jaxlib\nprint(\"python\", sys.version.split()[0])\nprint(\"jax\", jax.__version__, \"jaxlib\", jaxlib.__version__)\nprint(\"backend\", jax.default_backend())\nprint(\"process\", jax.process_index(), \"/\", jax.process_count())\nprint(\"device_count\", jax.device_count(), \"local\", len(jax.local_devices()))\nPY'`

### 4a) Run 1-step GRPO/GSM8K micro-batched training (backward-compatible env vars; all workers)

- `cd /home/john/github/MLLM-JAX`
- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-16-260117125029 --zone us-central2-b --project civil-rarity-482610-s5 --worker all --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; export HF_HUB_ENABLE_HF_TRANSFER=1; export WANDB_MODE=disabled; export TOKENIZERS_PARALLELISM=false; export MODEL_PATH=\"Qwen/Qwen2.5-7B-Instruct\"; export STEPS=1; export BATCH_SIZE=2; export NUM_PRE_Q=8; export TRAIN_MICRO_BATCH_SIZE=4; export GLOBAL_LENGTH=512; export MAX_LENGTH_SAMPLE=64; export PPO_EPOCHS=1; export BETA=0.0; python -u scripts/run_grpo_gsm8k_training.py'`

### 4b) Run 1-step GRPO/GSM8K with per-device batching + global rollout target (auto passes + auto grad-accum; all workers)

- `cd /home/john/github/MLLM-JAX`
- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-16-260117125029 --zone us-central2-b --project civil-rarity-482610-s5 --worker all --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; export HF_HUB_ENABLE_HF_TRANSFER=1; export WANDB_MODE=disabled; export TOKENIZERS_PARALLELISM=false; export MODEL_PATH=\"Qwen/Qwen2.5-7B-Instruct\"; export STEPS=1; export ROLLOUT_GLOBAL_BATCH_SIZE=16; export ROLLOUT_PER_DEVICE_BATCH_SIZE=1; export NUM_PRE_Q=2; export TRAIN_PER_DEVICE_MICRO_BATCH_SIZE=1; export GLOBAL_LENGTH=512; export MAX_LENGTH_SAMPLE=64; export PPO_EPOCHS=1; export BETA=0.0; python -u scripts/run_grpo_gsm8k_training.py'`

## Expected Result

- Both workers print `backend=tpu process=0/2` and `backend=tpu process=1/2`.
- Runner prints an effective config that includes:
  - `local_batch=...`
  - `rollout.passes_per_step=...` (when `ROLLOUT_GLOBAL_BATCH_SIZE` is set)
  - `train.micro_batch_size=...` + `train.grad_accum_steps=...` (when `TRAIN_MICRO_BATCH_SIZE` or `TRAIN_PER_DEVICE_MICRO_BATCH_SIZE` is set)
- Log includes `step=0 ...` and the command exits with code `0` (no Python tracebacks).

## Troubleshooting

- `jax.distributed.initialize()` hangs: ensure you run with `--worker all` (both hosts must launch).
- `Unable to initialize backend 'tpu' ... already in use`: try `rm -f /tmp/libtpu_lockfile` and ensure no other training process is running on the TPU VM.
- `train.micro_batch_size=... must divide local_batch=...`: ensure `TRAIN_MICRO_BATCH_SIZE` divides `BATCH_SIZE * NUM_PRE_Q`.
  - For per-device settings: `train.micro_batch_size` divides `rollout_passes * rollout.batch_size * rollout.num_pre_q` (all per-process counts).

## References

- `scripts/run_grpo_gsm8k_training.py`
- `plugins/training/runner/grpo_gsm8k.py`
- `plugins/training/configs/grpo_gsm8k_default.yaml`
- `docs/sops/tpu-vm-repo-sync.md`
