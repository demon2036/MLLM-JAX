# SOP: TPU VM v4-8 GRPO/GSM8K 20-step Train with `rollout.backend=sglang_jax` + W&B

- **Title**: SOP: Run GRPO/GSM8K for 20 steps on a single-host TPU VM (v4-8) with `rollout.backend=sglang_jax` and W&B logging via `.env`
  **Prereqs**: `gcloud` installed + authenticated; TPU quota/capacity; outbound internet from TPU VM (HF + datasets + wandb); repo pushed to GitHub; local `.env` contains a valid `WANDB_API_KEY` and is gitignored
  **Environment (verified)**:
  - Project: `civil-rarity-482610-s5`
  - Zone: `us-central2-b`
  - TPU VM: `mllm-jax-v4-8-260117090531` (`v4-8`, single host)
  - Conda env: `mllm-jax`; Python `3.12.12`
  - JAX: `0.8.2`, jaxlib `0.8.2`
  - sglang-jax: `0.0.2`
  - wandb: `0.24.0` (logged in run output)
  - Git ref on TPU (verified): `e2c72a1`

## Steps (commands actually used)

### 0) Prepare local secrets (never commit)

- Copy `.env.example` to `.env` and fill:
  - `WANDB_API_KEY=<your_wandb_api_key>` (must be 40+ chars)
  - Optional:
    - `WANDB_PROJECT=mllm-jax-grpo-gsm8k`
    - `WANDB_MODE=online`

### 1) Sync repo on the TPU VM (git-sync, no SCP)

- `TPU_NAME=mllm-jax-v4-8-260117090531; ZONE=us-central2-b; PROJECT=civil-rarity-482610-s5`
- `gcloud alpha compute tpus tpu-vm ssh root@$TPU_NAME --project $PROJECT --zone $ZONE --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:V3Ugj8P2EOHhKueQ7iZ1p+PUuMvI7OHufNpEqm6b0Yw --command "set -euo pipefail; cd /root/MLLM-JAX; git fetch --all --prune; git checkout mllm-jax-sglang; git reset --hard origin/mllm-jax-sglang; git clean -fd; echo HEAD=\$(git rev-parse --short HEAD); git status -sb"`

### 2) Copy `.env` to the TPU VM (secret sync allowed)

- `TPU_NAME=mllm-jax-v4-8-260117090531; ZONE=us-central2-b; PROJECT=civil-rarity-482610-s5`
- `gcloud alpha compute tpus tpu-vm scp .env root@$TPU_NAME:/root/.env --project $PROJECT --zone $ZONE --worker 0 --scp-flag=-batch --scp-flag=-hostkey --scp-flag=SHA256:V3Ugj8P2EOHhKueQ7iZ1p+PUuMvI7OHufNpEqm6b0Yw`

Verify key length (optional):

- `gcloud alpha compute tpus tpu-vm ssh root@$TPU_NAME --project $PROJECT --zone $ZONE --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:V3Ugj8P2EOHhKueQ7iZ1p+PUuMvI7OHufNpEqm6b0Yw --command "set -euo pipefail; echo -n WANDB_API_KEY_len=; grep '^WANDB_API_KEY=' /root/.env | head -n 1 | cut -d= -f2- | tr -d '\\r\\n' | wc -c"`

### 3) Run 20 steps via nohup helper (recommended on Windows + plink)

This avoids `gcloud` retry/SSH disconnects accidentally starting multiple foreground runs.

- `cd <repo>`
- `$ts=(Get-Date).ToUniversalTime().ToString('yyyyMMdd_HHmmss'); gcloud alpha compute tpus tpu-vm ssh root@$TPU_NAME --project $PROJECT --zone $ZONE --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:V3Ugj8P2EOHhKueQ7iZ1p+PUuMvI7OHufNpEqm6b0Yw --command "set -euo pipefail; cd /root/MLLM-JAX; export WANDB_MODE=online; export WANDB_PROJECT=mllm-jax-grpo-gsm8k; export WANDB_NAME=grpo_gsm8k_sglang_jax_v4-8_e2c72a1_steps20_$ts; export STEPS=20; export ROLLOUT_BACKEND=sglang_jax; bash scripts/tpu_vm_start_grpo_gsm8k_sglang_jax_smoke.sh"`

### 4) Monitor progress and verify exit code

- `gcloud alpha compute tpus tpu-vm ssh root@$TPU_NAME --project $PROJECT --zone $ZONE --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:V3Ugj8P2EOHhKueQ7iZ1p+PUuMvI7OHufNpEqm6b0Yw --command "set -euo pipefail; cd /root/MLLM-JAX; grep -n 'step=' logs/nohup_grpo_gsm8k_sglang_jax_smoke_latest.log | tail -n 5 || true"`
- `gcloud alpha compute tpus tpu-vm ssh root@$TPU_NAME --project $PROJECT --zone $ZONE --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:V3Ugj8P2EOHhKueQ7iZ1p+PUuMvI7OHufNpEqm6b0Yw --command "set -euo pipefail; cd /root/MLLM-JAX; cat logs/nohup_grpo_gsm8k_sglang_jax_smoke_latest.exit"`

Observed success tail:

- `step=19 loss=0.000000 entropy=0.4395 reward_mean=0.1250 dt=19.32s`
- `logs/nohup_grpo_gsm8k_sglang_jax_smoke_latest.exit` contains `0`

## Expected Result

- Log contains `wandb: ... Loaded credentials ... from WANDB_API_KEY.` and a run URL.
- Log prints `step=0 ...` through `step=19 ...` without Python tracebacks.
- Exit file exists and contains `0`.

## Troubleshooting

- `wandb disabled due to init error: WANDB_API_KEY invalid ...`:
  - Refresh key at https://wandb.ai/authorize and re-sync `.env`.
- Hang/`libtpu` lock:
  - `rm -f /tmp/libtpu_lockfile` and ensure no other training process is running.
- OOM with co-located train + rollout:
  - Reduce sglang-jax footprint via env knobs:
    - `SGLANG_JAX_MEM_FRACTION_STATIC`, `SGLANG_JAX_MAX_TOTAL_TOKENS`, `SGLANG_JAX_CONTEXT_LENGTH`, `SGLANG_JAX_DTYPE`

## References

- `docs/sops/tpu-vm-repo-sync.md`
- `docs/sops/tpu-vm-v4-8-grpo-gsm8k-rollout-backend-sglang-jax-smoke.md`
- `scripts/tpu_vm_start_grpo_gsm8k_sglang_jax_smoke.sh`
- `scripts/run_grpo_gsm8k_training.py`
- `plugins/training/rollout_backends/sglang_jax.py`
