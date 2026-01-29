# SOP: TPU v6e-8 unified-api-refactor SFT + nanoGPT regression (W&B online)

- **Title**: SOP: TPU v6e-8 unified-api-refactor SFT + nanoGPT regression (W&B online)
- **Prereqs**:
  - TPU VM reachable (v6e-8)
  - Repo synced to TPU via Git (no SCP)
  - Conda env exists with JAX TPU wheel + deps installed
  - `/root/.env` contains `WANDB_API_KEY` (do not commit)
- **Environment (verified)**:
  - TPU VM: `unified-api-refactor-sft-v6e-8-260129095800` (zone `us-east1-d`, 8 devices)
  - OS: Ubuntu 24.04 (kernel `6.11.0-1015-gcp`)
  - Conda env: `mllm-jax` (Python `3.12.12`)
  - JAX: `0.9.0` (`jax.default_backend()==tpu`, `len(jax.devices())==8`)
  - Flax/Optax: `flax 0.12.3`, `optax 0.2.6`
  - W&B: `0.24.0` (mode `online`)

## Goal

- Validate the unified `plugins/api` + `plugins/training/{core,rl,sft}` refactor can run:
  - SID SFT (JAX) on TPU with W&B online
  - nanoGPT (Tiny Shakespeare char) regression on TPU with W&B online

## Steps (commands actually run)

### 0) Sync repo to the target branch

- `scripts/ssh_tpu_vm_root.sh --name unified-api-refactor-sft-v6e-8-260129095800 --zone us-east1-d --command 'set -euo pipefail; cd /root/MLLM-JAX; git fetch --all --prune; git checkout john/unified-api-refactor-20260129; git pull; git rev-parse --short HEAD'` (exit `0`)

### 1) (Optional) Print TPU env versions

- `scripts/ssh_tpu_vm_root.sh --name unified-api-refactor-sft-v6e-8-260129095800 --zone us-east1-d --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; python - <<\"PY\"\\nimport jax, flax, optax, wandb\\nprint(\"jax\", jax.__version__)\\nprint(\"jax_backend\", jax.default_backend())\\nprint(\"devices\", len(jax.devices()))\\nprint(\"flax\", flax.__version__)\\nprint(\"optax\", optax.__version__)\\nprint(\"wandb\", wandb.__version__)\\nPY'` (exit `0`)

### 2) Run SID SFT (3 steps, step-time config)

On TPU, W&B recorded the following invocation (see `wandb-metadata.json` in the run directory):

- `python -u scripts/run_sid_sft.py --config projects/sid_sft/configs/sid_sft_jax_qwen25_1p5b_instruct_industrial_v6e8_step_time.yaml --run-mode train` (exit `0`)

Results:

- W&B run: `https://wandb.ai/johntitordemon2036/minionerec-sid-sft/runs/vu5sk3k4`
- Summary JSON: `runs/sid_sft_jax_qwen25_1p5b_instruct_industrial_v6e8_step_time/run_summary.json`

### 3) Run nanoGPT regression (1500 steps)

- `scripts/ssh_tpu_vm_root.sh --name unified-api-refactor-sft-v6e-8-260129095800 --zone us-east1-d --env-file /root/.env --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; python -u projects/nano_gpt/run.py --config projects/nano_gpt/configs/tinyshakespeare_char_v6e_8_refactor_regression_1500steps.yaml'` (exit `0`)

Results:

- W&B run: `https://wandb.ai/johntitordemon2036/nano-gpt-jax/runs/y5rtmtaz`
- Summary JSON: `runs/nano_gpt_tinyshakespeare_v6e_8_refactor_regression_1500steps/run_summary.json`
- Observed: `eval/loss=1.4672` at `step=1500`

## Expected Result

- Both runs exit `0` with no traceback.
- W&B shows metrics and the run URLs above.
- nanoGPT `eval/loss` at `step=1500` is close to the historical baseline (`~1.47`).

## Troubleshooting

- If W&B auth fails, strip CRLF in `/root/.env` (`sed -i \"s/\\r$//\" /root/.env`).
- If TPU reports `rope_theta` missing on Qwen2 configs: ensure the refactor branch includes `plugins/training/core/io/hf_config.ensure_rope_theta()`.

## References

- `docs/sops/tpu-vm-repo-sync.md`
- `docs/sops/tpu-vm-v6e-8-nanogpt-tinyshakespeare-standard.md`
- `memory/20260129_unified-api-refactor/README.md`
