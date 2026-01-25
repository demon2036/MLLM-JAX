# SOP: GRPO Pallas kernel gradcheck (Qwen2.5-1.5B) on TPU

- **Title**: SOP: Run a logits-level GRPO loss+grad equivalence check between reference JAX and a Pallas kernel on a TPU VM
  **Prereqs**: TPU VM reachable via `gcloud ... tpu-vm ssh`; outbound internet on TPU (HF + W&B); repo synced via Git (no SCP); `WANDB_API_KEY` available if `wandb_mode=online`
  **Environment (to fill once verified)**:
  - TPU VM: `<TPU_NAME>` (`v6e-8`, spot), zone `<ZONE>`
  - Conda env: `<env>` (Python `<ver>`)
  - JAX/jaxlib/libtpu: `<versions>`
  - Repo: `https://github.com/demon2036/MLLM-JAX.git`, branch/commit `<branch>@<sha>`
  - W&B run: `<url>`

## Goal

- Validate that `plugins/training/kernels/grpo_loss_pallas.py` matches the existing reference GRPO computation:
  - Forward: `loss_ref ≈ loss_kernel`
  - Backward: `dlogits_ref ≈ dlogits_kernel`

## Steps (placeholders until verified)

### 1) Git-sync repo to TPU VM (no SCP)

- See `docs/sops/tpu-vm-repo-sync.md`

### 2) Ensure runtime deps on TPU VM

- `<install commands actually used>`

### 3) Run gradcheck script (W&B online)

- `cd /root/MLLM-JAX`
- `python -u scripts/grpo_kernel_gradcheck.py --config plugins/training/configs/grpo_kernel_gradcheck_qwen25_1p5b.yaml`

## Expected Result

- Script exits `0`.
- Output prints max/mean diffs and they are within the configured tolerances.

## Troubleshooting

- If W&B init fails: confirm `/root/.env` has `WANDB_API_KEY` and `wandb_mode=online` in the YAML config.
- If pallas compilation fails: try smaller `seq_len`/`batch_size` in the gradcheck config to reduce compile pressure.

## References

- `memory/20260125_grpo-pallas-kernel/README.md`

