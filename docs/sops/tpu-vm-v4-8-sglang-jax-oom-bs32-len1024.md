# SOP: Diagnose sglang-jax OOM on v4-8 (bs32 len1024) and verify a small smoke run

- **Title**: SOP: Diagnose sglang-jax OOM on v4-8 (bs32 len1024) and verify a small smoke run
  **Prereqs**: `gcloud` installed + authenticated; TPU VM running; repo present at `/root/MLLM-JAX`; WANDB disabled (or a valid `WANDB_API_KEY` if enabled)
  **Environment (verified)**:
  - Project: `civil-rarity-482610-s5`
  - Zone: `us-central2-b`
  - TPU VM: `mllm-jax-v4-8-260117090531` (`v4-8`, single host)
  - JAX: `0.8.2`, jaxlib `0.8.2`
  - sglang-jax: `0.0.2`
  - Git ref on TPU (verified): `15154fc`

## Steps (commands actually used)

### 1) Inspect the existing bs32/len1024 sglang-jax OOM log

- `gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v4-8-260117090531 --project civil-rarity-482610-s5 --zone us-central2-b --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:V3Ugj8P2EOHhKueQ7iZ1p+PUuMvI7OHufNpEqm6b0Yw --command "set -euo pipefail; tail -n 160 /root/MLLM-JAX/logs/nohup_grpo_gsm8k_bs32_sglang_jax_20260119_084640.log"`

Observed:

- `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: ... Error loading program 'jit_training_step' ... Attempting to reserve 7.82G ... 5.68G free`

### 2) Run a small sglang-jax smoke with a reduced footprint

- `gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v4-8-260117090531 --project civil-rarity-482610-s5 --zone us-central2-b --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:V3Ugj8P2EOHhKueQ7iZ1p+PUuMvI7OHufNpEqm6b0Yw --command "set -euo pipefail; cd /root/MLLM-JAX; LOG_DIR=/root/MLLM-JAX/logs; mkdir -p \"${LOG_DIR}\"; RUN_TS=$(date -u +%Y%m%d_%H%M%S); COMMIT=$(git rev-parse --short HEAD); LOG_FILE=\"${LOG_DIR}/oom_investigate_smoke_${COMMIT}_${RUN_TS}.log\"; LATEST=\"${LOG_DIR}/oom_investigate_smoke_latest.log\"; ln -sf \"${LOG_FILE}\" \"${LATEST}\"; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; export HF_HUB_ENABLE_HF_TRANSFER=1; export TOKENIZERS_PARALLELISM=false; export WANDB_MODE=disabled; export MODEL_PATH=Qwen/Qwen2.5-7B-Instruct; export STEPS=1; export ROLLOUT_BACKEND=sglang_jax; export BATCH_SIZE=1; export NUM_PRE_Q=4; export GLOBAL_LENGTH=512; export MAX_LENGTH_SAMPLE=64; export PPO_EPOCHS=1; export GRAD_ACCUM_STEPS=1; export BETA=0.0; export EVAL_EVERY_STEPS=0; export SGLANG_JAX_MEM_FRACTION_STATIC=0.5; echo \"LOG_FILE=${LOG_FILE}\"; set +e; python -u scripts/run_grpo_gsm8k_training.py 2>&1 | tee \"${LOG_FILE}\"; status=${PIPESTATUS[0]}; set -e; echo \"exit_status=${status}\"; tail -n 40 \"${LOG_FILE}\"; exit ${status}"`

Observed:

- `exit_status=0`
- A line like `step=0 loss=0.000000 ...` appears with no tracebacks.

### 3) Attempt naive-aligned bs32/len1024 with sglang-jax (still OOM on v4-8)

Default sglang-jax envs (chunking defaults + mem_fraction_static=0.5), 1 step:

- `gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v4-8-260117090531 --project civil-rarity-482610-s5 --zone us-central2-b --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:V3Ugj8P2EOHhKueQ7iZ1p+PUuMvI7OHufNpEqm6b0Yw --command "set -euo pipefail; cd /root/MLLM-JAX; LOG_DIR=/root/MLLM-JAX/logs; mkdir -p \"${LOG_DIR}\"; RUN_TS=$(date -u +%Y%m%d_%H%M%S); COMMIT=$(git rev-parse --short HEAD); LOG_FILE=\"${LOG_DIR}/sglang_jax_align_naive_bs32_len1024_${COMMIT}_${RUN_TS}.log\"; LATEST=\"${LOG_DIR}/sglang_jax_align_naive_bs32_len1024_latest.log\"; ln -sf \"${LOG_FILE}\" \"${LATEST}\"; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; export HF_HUB_ENABLE_HF_TRANSFER=1; export TOKENIZERS_PARALLELISM=false; export WANDB_MODE=disabled; export STEPS=1; echo \"LOG_FILE=${LOG_FILE}\"; set +e; python -u scripts/run_grpo_gsm8k_training.py --config plugins/training/configs/grpo_gsm8k_bs32_sglang_jax.yaml 2>&1 | tee \"${LOG_FILE}\"; status=${PIPESTATUS[0]}; set -e; echo \"exit_status=${status}\"; tail -n 40 \"${LOG_FILE}\"; exit ${status}"`

Observed:

- `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Error loading program 'jit_training_step' ... 5.68G free`

More aggressive KV limits (chunk size 1 + max_total_tokens=1536):

- `gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v4-8-260117090531 --project civil-rarity-482610-s5 --zone us-central2-b --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:V3Ugj8P2EOHhKueQ7iZ1p+PUuMvI7OHufNpEqm6b0Yw --command "set -euo pipefail; cd /root/MLLM-JAX; LOG_DIR=/root/MLLM-JAX/logs; mkdir -p \"${LOG_DIR}\"; RUN_TS=$(date -u +%Y%m%d_%H%M%S); COMMIT=$(git rev-parse --short HEAD); LOG_FILE=\"${LOG_DIR}/sglang_jax_align_naive_bs32_len1024_${COMMIT}_${RUN_TS}.log\"; LATEST=\"${LOG_DIR}/sglang_jax_align_naive_bs32_len1024_latest.log\"; ln -sf \"${LOG_FILE}\" \"${LATEST}\"; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; export HF_HUB_ENABLE_HF_TRANSFER=1; export TOKENIZERS_PARALLELISM=false; export WANDB_MODE=disabled; export STEPS=1; export SGLANG_JAX_ROLLOUT_CHUNK_SIZE=1; export SGLANG_JAX_MAX_RUNNING_REQUESTS=1; export SGLANG_JAX_MAX_TOTAL_TOKENS=1536; export SGLANG_JAX_MEM_FRACTION_STATIC=0.4; echo \"LOG_FILE=${LOG_FILE}\"; set +e; python -u scripts/run_grpo_gsm8k_training.py --config plugins/training/configs/grpo_gsm8k_bs32_sglang_jax.yaml 2>&1 | tee \"${LOG_FILE}\"; status=${PIPESTATUS[0]}; set -e; echo \"exit_status=${status}\"; tail -n 40 \"${LOG_FILE}\"; exit ${status}"`

Observed:

- `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Error loading program 'jit_training_step' ... 5.90G free`

If mem_fraction_static is reduced further (e.g., `0.35`), the Engine fails to initialize with:

- `RuntimeError: Not enough memory. Please try to increase --mem-fraction-static.`

## Expected Result

- The bs32/len1024 log shows a `RESOURCE_EXHAUSTED` error for `jit_training_step`.
- The small smoke run finishes with `exit_status=0` and prints `step=0 ...`.
- Naive-aligned bs32/len1024 with co-located sglang-jax still OOMs on v4-8 (see step 3).

## Troubleshooting

- TPU init error (`Operation not permitted` on `/dev/accel*`):
  - Kill stale training jobs and remove lockfile:
    - `gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v4-8-260117090531 --project civil-rarity-482610-s5 --zone us-central2-b --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:V3Ugj8P2EOHhKueQ7iZ1p+PUuMvI7OHufNpEqm6b0Yw --command "set -euo pipefail; ps -ef | grep run_grpo_gsm8k_training.py | grep -v grep | awk '{print $2}' | xargs -r kill -9 || true; rm -f /tmp/libtpu_lockfile || true"`
- `ValueError: Unable to shard batch=1 across local (dp,fsdp) shards=4`:
  - Ensure `train.micro_batch_size` is a multiple of the local device count (v4-8 uses `4` local devices).
- `RuntimeError: Not enough memory. Please try to increase --mem-fraction-static.`:
  - Increase `SGLANG_JAX_MEM_FRACTION_STATIC` or reduce `SGLANG_JAX_MAX_TOTAL_TOKENS`/rollout chunk size.

## Conclusion

- With `Qwen/Qwen2.5-7B-Instruct`, `bs=32`, `max_length_sample=1024`, and co-located rollout+train on v4-8, sglang-jax still fails due to HBM limits. To align naive settings, use a larger TPU (v4-16+) or separate rollout host.

## References

- `plugins/training/runner/grpo_gsm8k.py`
- `plugins/training/rollout_backends/sglang_jax.py`
- `docs/sops/tpu-vm-v4-8-grpo-gsm8k-rollout-backend-sglang-jax-smoke.md`
