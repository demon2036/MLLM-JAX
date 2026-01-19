# SOP: TPU VM v4-8 smoke-run `scripts/run_grpo_gsm8k_training.py` with `rollout.backend=sglang_jax`

- **Title**: SOP: TPU VM v4-8 smoke-run GRPO/GSM8K runner with `rollout.backend=sglang_jax`
  **Prereqs**: `gcloud` installed + authenticated; TPU quota/capacity; outbound internet from TPU VM (HF + datasets + wandb); repo pushed to GitHub; local `.env` is gitignored
  **Environment (verified)**:
  - Project: `civil-rarity-482610-s5`
  - Zone: `us-central2-b`
  - TPU VM: `mllm-jax-v4-8-260117090531` (`v4-8`, single host)
  - Conda env: `mllm-jax`; Python `3.12.12`
  - JAX: `0.8.2`, jaxlib `0.8.2`
  - sglang-jax: `0.0.2`
  - Git ref on TPU (verified): `d837682`

## Steps (commands actually used)

### 0) Prepare local secrets (never commit)

- Copy `.env.example` to `.env` and fill:
  - `WANDB_API_KEY=<your_wandb_api_key>`
  - Keep `.env` gitignored (this repo already ignores it).

### 1) Sync repo on the TPU VM (git-sync, no SCP)

- `TPU_NAME=mllm-jax-v4-8-260117090531; ZONE=us-central2-b; PROJECT=civil-rarity-482610-s5`
- `gcloud alpha compute tpus tpu-vm ssh root@$TPU_NAME --project $PROJECT --zone $ZONE --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:V3Ugj8P2EOHhKueQ7iZ1p+PUuMvI7OHufNpEqm6b0Yw --command "set -euo pipefail; cd /root/MLLM-JAX; git fetch --all --prune; git checkout mllm-jax-sglang; git reset --hard origin/mllm-jax-sglang; git clean -fd; echo HEAD=\$(git rev-parse --short HEAD); git status -sb"`

### 2) Copy `.env` to the TPU VM (secret sync allowed)

- `TPU_NAME=mllm-jax-v4-8-260117090531; ZONE=us-central2-b; PROJECT=civil-rarity-482610-s5`
- `gcloud alpha compute tpus tpu-vm scp .env root@$TPU_NAME:/root/.env --project $PROJECT --zone $ZONE --worker 0 --scp-flag=-batch --scp-flag=-hostkey --scp-flag=SHA256:V3Ugj8P2EOHhKueQ7iZ1p+PUuMvI7OHufNpEqm6b0Yw`

### 3) (Optional) Verify runtime versions on TPU

- `TPU_NAME=mllm-jax-v4-8-260117090531; ZONE=us-central2-b; PROJECT=civil-rarity-482610-s5`
- `gcloud alpha compute tpus tpu-vm ssh root@$TPU_NAME --project $PROJECT --zone $ZONE --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:V3Ugj8P2EOHhKueQ7iZ1p+PUuMvI7OHufNpEqm6b0Yw --command "set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; python -V; python -c 'import jax, jaxlib; print(jax.__version__, jaxlib.__version__); print(jax.default_backend(), jax.device_count())'; pip show sglang-jax | sed -n '1,8p'"`

### 4) Run the 1-step smoke (`rollout.backend=sglang_jax`) via nohup helper

- `TPU_NAME=mllm-jax-v4-8-260117090531; ZONE=us-central2-b; PROJECT=civil-rarity-482610-s5`
- `gcloud alpha compute tpus tpu-vm ssh root@$TPU_NAME --project $PROJECT --zone $ZONE --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:V3Ugj8P2EOHhKueQ7iZ1p+PUuMvI7OHufNpEqm6b0Yw --command "set -euo pipefail; cd /root/MLLM-JAX; export WANDB_MODE=online; export WANDB_PROJECT=mllm-jax-grpo-gsm8k; export WANDB_NAME=grpo_gsm8k_sglang_jax_v4-8_d837682_steps1; bash scripts/tpu_vm_start_grpo_gsm8k_sglang_jax_smoke.sh"`

### 5) Monitor logs and check exit code

- `TPU_NAME=mllm-jax-v4-8-260117090531; ZONE=us-central2-b; PROJECT=civil-rarity-482610-s5`
- `gcloud alpha compute tpus tpu-vm ssh root@$TPU_NAME --project $PROJECT --zone $ZONE --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:V3Ugj8P2EOHhKueQ7iZ1p+PUuMvI7OHufNpEqm6b0Yw --command "set -euo pipefail; cd /root/MLLM-JAX; tail -n 40 logs/nohup_grpo_gsm8k_sglang_jax_smoke_latest.log"`
- `gcloud alpha compute tpus tpu-vm ssh root@$TPU_NAME --project $PROJECT --zone $ZONE --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:V3Ugj8P2EOHhKueQ7iZ1p+PUuMvI7OHufNpEqm6b0Yw --command "set -euo pipefail; cd /root/MLLM-JAX; cat logs/nohup_grpo_gsm8k_sglang_jax_smoke_latest.exit"`

Observed success line:

- `step=0 loss=0.000000 entropy=0.2363 reward_mean=0.1250 dt=122.87s`

## Expected Result

- Logs contain an effective config with `rollout.backend: sglang_jax`.
- A line like `step=0 ...` appears with no Python tracebacks.
- `logs/nohup_grpo_gsm8k_sglang_jax_smoke_latest.exit` exists and contains `0`.

## Troubleshooting

- W&B says `WANDB_API_KEY invalid ... has <N>`:
  - Check key length (must be 40+ chars):
    - `grep '^WANDB_API_KEY=' /root/.env | head -n 1 | cut -d= -f2 | tr -d '\r\n' | wc -c`
  - Refresh the API key at https://wandb.ai/authorize and re-sync `.env`.
- TPU backend already in use:
  - `rm -f /tmp/libtpu_lockfile` and ensure no other training process is running.
- OOM when co-locating train + rollout:
  - Reduce sglang-jax footprint via env knobs:
    - `SGLANG_JAX_MEM_FRACTION_STATIC`, `SGLANG_JAX_MAX_TOTAL_TOKENS`, `SGLANG_JAX_CONTEXT_LENGTH`, `SGLANG_JAX_DTYPE`

## References

- `docs/sops/tpu-vm-repo-sync.md`
- `docs/sops/tpu-vm-v4-8-grpo-gsm8k-rollout-backend-naive-smoke.md`
- `scripts/tpu_vm_start_grpo_gsm8k_sglang_jax_smoke.sh`
- `plugins/training/rollout_backends/sglang_jax.py`
- `plugins/training/runner/grpo_gsm8k.py`
- `scripts/run_grpo_gsm8k_training.py`
