# Qwen Attention Refactor + TPU v4-8 Validation (GRPO/GSM8K, W&B)

- **Title**: SOP: Refactor Qwen attention (borrow patterns from MaxText) and validate by running GRPO/GSM8K on a TPU `v4-8` with W&B enabled
  **Prereqs**:
  - Local: Windows PowerShell; `git`; `python`; `pytest`; `gcloud` authenticated
  - TPU VM: reachable via `gcloud alpha compute tpus tpu-vm ssh`; outbound internet (HF + datasets + wandb)
  - W&B: `WANDB_API_KEY` available (do **not** commit it; use `/root/.env` on TPU)
  **Environment (verified)**:
  - Local: gcloud `551.0.0`; project `civil-rarity-482610-s5`
  - TPU VM: `vllm-tpu-v4-8-bench` (zone `us-central2-b`), Ubuntu `22.04.2`, JAX device_count `4`
  - Repo: `https://github.com/demon2036/MLLM-JAX.git`, branch `refactor-attention` (`819a212`)

## Steps (commands actually used)

### 1) (Local) Study MaxText attention structure (reference-only)

- `git clone --depth 1 https://github.com/google/maxtext.git memo/maxtext`

### 2) (Local) Refactor attention + clean deprecated plugin + run tests

- `python -m py_compile MLLM_JAX/language/attention.py MLLM_JAX/language/llama/llama.py MLLM_JAX/language/qwen2/modular_qwen2.py`
- `python -m pytest -q`
- `git push origin refactor-attention`

### 3) (TPU) Bootstrap Miniconda + env

Host key note (Windows/Plink):
- If you hit a hostkey prompt/mismatch, pass it explicitly (example used in this run):
  - `--ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:+bwuNJF6xP/ysxYGNxZbQNPaH/vjTYB+GCqVxhuFveI`

Install Miniconda:
- `gcloud alpha compute tpus tpu-vm ssh root@vllm-tpu-v4-8-bench --project civil-rarity-482610-s5 --zone us-central2-b --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:+bwuNJF6xP/ysxYGNxZbQNPaH/vjTYB+GCqVxhuFveI --command "set -euo pipefail; if [ ! -d /root/miniconda3 ]; then curl -fsSL -o /root/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; bash /root/miniconda.sh -b -p /root/miniconda3; rm -f /root/miniconda.sh; fi; /root/miniconda3/bin/conda --version"`

Create env:
- `gcloud alpha compute tpus tpu-vm ssh root@vllm-tpu-v4-8-bench --project civil-rarity-482610-s5 --zone us-central2-b --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:+bwuNJF6xP/ysxYGNxZbQNPaH/vjTYB+GCqVxhuFveI --command "set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true; conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true; if ! conda env list | awk '{print $1}' | grep -qx mllm-jax; then conda create -y -n mllm-jax python=3.12; fi; conda activate mllm-jax; python --version; python -m pip install -U pip"`

Install deps (JAX TPU + requirements):
- `gcloud alpha compute tpus tpu-vm ssh root@vllm-tpu-v4-8-bench --project civil-rarity-482610-s5 --zone us-central2-b --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:+bwuNJF6xP/ysxYGNxZbQNPaH/vjTYB+GCqVxhuFveI --command "set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; python -m pip install -U pip; python -m pip install -U \"jax[tpu]\" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; python -m pip install -U torch --index-url https://download.pytorch.org/whl/cpu; cd /root/MLLM-JAX; python -m pip install -U -r requirements-tpu.txt"`

### 4) (TPU) Git sync + write `/root/.env` (W&B) + start GRPO

Git sync:
- `gcloud alpha compute tpus tpu-vm ssh root@vllm-tpu-v4-8-bench --project civil-rarity-482610-s5 --zone us-central2-b --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:+bwuNJF6xP/ysxYGNxZbQNPaH/vjTYB+GCqVxhuFveI --command "set -euo pipefail; REPO_URL=https://github.com/demon2036/MLLM-JAX.git; REPO_DIR=/root/MLLM-JAX; if [ ! -d \"$REPO_DIR/.git\" ]; then rm -rf \"$REPO_DIR\"; git clone \"$REPO_URL\" \"$REPO_DIR\"; fi; cd \"$REPO_DIR\"; git fetch --all --prune; git checkout refactor-attention; git reset --hard origin/refactor-attention; git clean -fd; git rev-parse --short HEAD"`

Write W&B key (replace the placeholder):
- `gcloud alpha compute tpus tpu-vm ssh root@vllm-tpu-v4-8-bench --project civil-rarity-482610-s5 --zone us-central2-b --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:+bwuNJF6xP/ysxYGNxZbQNPaH/vjTYB+GCqVxhuFveI --command "set -euo pipefail; umask 077; cat > /root/.env <<'EOF'\nWANDB_API_KEY=<YOUR_WANDB_API_KEY>\nEOF\nchmod 600 /root/.env; ls -la /root/.env"`

Start training (nohup):
- `gcloud alpha compute tpus tpu-vm ssh root@vllm-tpu-v4-8-bench --project civil-rarity-482610-s5 --zone us-central2-b --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:+bwuNJF6xP/ysxYGNxZbQNPaH/vjTYB+GCqVxhuFveI --command "set -euo pipefail; cd /root/MLLM-JAX; export HF_HUB_ENABLE_HF_TRANSFER=1; export TOKENIZERS_PARALLELISM=false; bash scripts/tpu_vm_start_grpo_gsm8k_qwen25_3b_bs128_steps100_nohup.sh"`

Monitor:
- `gcloud alpha compute tpus tpu-vm ssh root@vllm-tpu-v4-8-bench --project civil-rarity-482610-s5 --zone us-central2-b --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:+bwuNJF6xP/ysxYGNxZbQNPaH/vjTYB+GCqVxhuFveI --command "set -euo pipefail; cd /root/MLLM-JAX; grep -n \"^step=\" logs/nohup_grpo_gsm8k_qwen25_3b_bs128_steps100_latest.log | tail -n 10 || true"`
- `gcloud alpha compute tpus tpu-vm ssh root@vllm-tpu-v4-8-bench --project civil-rarity-482610-s5 --zone us-central2-b --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:+bwuNJF6xP/ysxYGNxZbQNPaH/vjTYB+GCqVxhuFveI --command "set -euo pipefail; cd /root/MLLM-JAX; cat logs/nohup_grpo_gsm8k_qwen25_3b_bs128_steps100_latest.exit 2>/dev/null || true"`  # expect `0` when complete

## Expected Result

- A W&B run appears under `johntitordemon2036/mllm-jax-grpo-gsm8k`.
- TPU log prints `step=<n> ...` lines and eventually writes a `*.exit` file containing `0`.

## Troubleshooting

- Host key mismatch (Windows/Plink): re-run with `--ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:...`.
- TPU OOM: reduce `train.micro_batch_size` (this repo uses `16` for v4-8 in `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml`).

## References

- MaxText repo: https://github.com/google/maxtext
- `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml`
- `scripts/tpu_vm_start_grpo_gsm8k_qwen25_3b_bs128_steps100_nohup.sh`

