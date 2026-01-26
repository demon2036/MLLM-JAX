# SOP: GRPO Pallas kernel gradcheck (Qwen2.5-1.5B) on TPU

- **Title**: SOP: Run a logits-level GRPO loss+grad equivalence check between reference JAX and a Pallas kernel on a TPU VM
  **Prereqs**: TPU VM reachable via `gcloud ... tpu-vm ssh`; outbound internet on TPU (HF + W&B); repo synced via Git (no SCP); `WANDB_API_KEY` available if `wandb_mode=online`
  **Environment (verified)**:
  - TPU VM: `mllm-jax-v6e-8-spot-260124132428` (`v6e-8`, spot), zone `us-east1-d`
  - Conda env: `mllm-jax` (Python `3.12.12`)
  - JAX/jaxlib: `0.9.0` / `0.9.0` (TPU)
  - Repo: `https://github.com/demon2036/MLLM-JAX.git`, branch `tiled-ce-pallas` @ `d7aae9e`
  - W&B run: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-kernel/runs/j71299gc`

## Goal

- Validate that `plugins/training/kernels/grpo_loss_pallas.py` matches the existing reference GRPO computation:
  - Forward: `loss_ref ≈ loss_kernel`
  - Backward: `dlogits_ref ≈ dlogits_kernel`

## Steps (commands actually used)

### 1) Git-sync repo to TPU VM (no SCP)

- TPU already had a checkout at `/root/MLLM-JAX`; we fast-forwarded it:
  - `echo y | gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --worker=0 --quiet --command "set -euo pipefail; cd /root/MLLM-JAX; git fetch --all --prune; git pull --ff-only; git rev-parse --short HEAD"`

### 2) Ensure runtime deps on TPU VM

- This TPU used an existing conda env `mllm-jax` with JAX + deps installed.
- Record versions:
  - `echo y | gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --worker=0 --quiet --command "/root/miniconda3/bin/conda run -n mllm-jax python -c 'import sys, jax, jaxlib; print(sys.version); print(jax.__version__); print(jaxlib.__version__)'"`

### 3) Run gradcheck script (W&B online)

- `echo y | gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --worker=0 --quiet --command "set -euo pipefail; cd /root/MLLM-JAX; /root/miniconda3/bin/conda run -n mllm-jax python -u scripts/grpo_kernel_gradcheck.py --config plugins/training/configs/grpo_kernel_gradcheck_qwen25_1p5b.yaml"`

## Expected Result

- Script exits `0`.
- Output prints max/mean diffs and they are within the configured tolerances.

## Troubleshooting

- If W&B init fails: confirm `/root/.env` has `WANDB_API_KEY` and `wandb_mode=online` in the YAML config.
- If pallas compilation fails: try smaller `seq_len`/`batch_size` in the gradcheck config to reduce compile pressure.

## References

- `memory/20260125_grpo-pallas-kernel/README.md`

