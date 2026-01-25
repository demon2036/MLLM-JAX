# SOP: Cross-entropy Pallas kernel gradcheck (Qwen2.5-1.5B) on TPU

- **Title**: SOP: Run a logits-level CE loss+grad equivalence check between reference JAX and a Pallas kernel on a TPU VM
  **Prereqs**: TPU VM reachable via `gcloud ... tpu-vm ssh`; outbound internet on TPU (HF + W&B); repo synced via Git (no SCP); `WANDB_API_KEY` available when `wandb_mode=online`
  **Environment (verified)**:
  - TPU VM: `mllm-jax-v6e-8-spot-260124132428` (`v6e-8`, spot), zone `us-east1-d`
  - Python: `3.12.3` (venv at `/root/venvs/mllm-jax`)
  - JAX/jaxlib/libtpu: `0.9.0` / `0.9.0` / `0.0.34`
  - Repo: `https://github.com/demon2036/MLLM-JAX.git` @ `cef0a7e`
  - W&B run: `https://wandb.ai/johntitordemon2036/mllm-jax-ce-kernel/runs/1815sclb`

## Goal

- Validate `plugins/training/kernels/tiled_cross_entropy_pallas.py` matches the reference CE computation:
  - Forward: per-token `loss`/`logp` and scalar loss match
  - Backward: `dlogits` matches

## Steps (commands actually used)

### 1) Git-sync repo to TPU VM (no SCP)

- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --command 'set -euo pipefail; REPO_URL=https://github.com/demon2036/MLLM-JAX.git; REPO_DIR=/root/MLLM-JAX; if [ ! -d \"$REPO_DIR/.git\" ]; then git clone \"$REPO_URL\" \"$REPO_DIR\"; fi; cd \"$REPO_DIR\"; git fetch --all --prune; git checkout cef0a7e; git status -sb; git rev-parse --short HEAD'`

### 2) Sync secrets (`WANDB_API_KEY`) to the TPU VM

- `scripts/sync_env_to_tpu_vm.sh --name mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --worker all`

### 3) Create a venv + install deps on TPU VM (PEP 668 safe)

- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --command 'set -euo pipefail; apt-get update -y; apt-get install -y python3.12-venv'`
- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --command 'set -euo pipefail; VENV=/root/venvs/mllm-jax; rm -rf \"$VENV\"; python3 -m venv \"$VENV\"; \"$VENV/bin/python\" -V; \"$VENV/bin/pip\" --version'`
- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --command 'set -euo pipefail; VENV=/root/venvs/mllm-jax; \"$VENV/bin/pip\" install -U pip; \"$VENV/bin/pip\" install -U \"jax[tpu]\" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html'`
- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --command 'set -euo pipefail; VENV=/root/venvs/mllm-jax; \"$VENV/bin/pip\" install -U flax optax chex transformers tqdm wandb pyyaml einops \"huggingface_hub[hf_transfer]\" safetensors'`
- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --command 'set -euo pipefail; VENV=/root/venvs/mllm-jax; \"$VENV/bin/pip\" install -U torch'`
- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --command 'set -euo pipefail; VENV=/root/venvs/mllm-jax; \"$VENV/bin/pip\" install -U webdataset datasets math_verify'`

### 4) Run the gradcheck script (W&B online)

- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --env-file /root/.env --command 'set -euo pipefail; cd /root/MLLM-JAX; /root/venvs/mllm-jax/bin/python -u scripts/cross_entropy_kernel_gradcheck.py --config plugins/training/configs/cross_entropy_kernel_gradcheck_qwen25_1p5b.yaml'`

## Expected Result

- Script exits `0`.
- Printed metrics show tight diffs, e.g.:
  - `abs_diff_loss == 0.0`
  - `dlogits_max_abs` on the order of `1e-6`

## Troubleshooting

- `externally-managed-environment`: use a venv (don’t `pip install` into system Python).
- Missing Python deps: install the missing wheel into the venv and re-run the script.
- Slow startup warning about transparent hugepages: optional; see the warning text for the sysfs toggle.

## References

- `memory/20260125_tiled-ce-pallas-kernel/README.md`
- `scripts/cross_entropy_kernel_gradcheck.py`
- `plugins/training/kernels/tiled_cross_entropy_pallas.py`

