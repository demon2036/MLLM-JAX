# SOP: MiniOneRec SID SFT Muon (high-LR) on TPU v6e-8 (W&B online)

- **Title**: SOP: MiniOneRec SID SFT Muon (high-LR) on TPU v6e-8 (W&B online)
- **Prereqs**:
  - gcloud auth + TPU quota for `v6e-8`
  - Repo pushed to GitHub (no SCP for code)
  - `/root/.env` on TPU VM contains `WANDB_API_KEY`
- **Environment (verified)**:
  - Commit: `john/sid-sft-muon-20260130` @ `5f53cca`
  - JAX: `0.9.0` (`jax.default_backend()==tpu`)
  - W&B: `0.24.1` (mode `online`)

## Goal

- Run SID SFT baseline (AdamW) + dp8 eval.
- Run SID SFT with Muon optimizer (Muon LR≈10× AdamW aux LR) + dp8 eval.

## Steps (commands actually run)

### 0) Create a v6e-8 TPU VM

- `TPU_NAME="sid-sft-muon-v6e-8-260130014837"; ./scripts/create_tpu_vm.sh --type v6e-8 --zone us-east1-d --name "$TPU_NAME"` (exit `0`)

### 1) Bootstrap Miniconda env

- `./scripts/bootstrap_miniconda_on_tpu_vm.sh --name "$TPU_NAME" --zone us-east1-d --env-name mllm-jax --python 3.12` (exit `0`)

### 2) Sync repo to branch + install deps

- Clone + checkout:
  - `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east1-d --command 'set -euo pipefail; if [ ! -d /root/MLLM-JAX/.git ]; then git clone https://github.com/demon2036/MLLM-JAX.git /root/MLLM-JAX; fi; cd /root/MLLM-JAX; git fetch --all --prune; git checkout john/sid-sft-muon-20260130; git pull --ff-only; git rev-parse --short HEAD'` (exit `0`)
- Install TPU deps (JAX TPU + torch CPU + requirements):
  - `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east1-d --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; python -m pip install -U pip; python -m pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; python -m pip install -U torch --index-url https://download.pytorch.org/whl/cpu; cd /root/MLLM-JAX; python -m pip install -U -r requirements-tpu.txt; python -m pip install -U fire pandas'` (exit `0`)

### 3) Clone upstream MiniOneRec (data paths)

- `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east1-d --command 'set -euo pipefail; cd /root/MLLM-JAX; mkdir -p workdir; if [ ! -d workdir/MiniOneRec/.git ]; then git clone https://github.com/AkaliKong/MiniOneRec workdir/MiniOneRec; fi'` (exit `0`)

### 4) Sync W&B `.env` to TPU

- `./scripts/sync_env_to_tpu_vm.sh --name "$TPU_NAME" --zone us-east1-d --src .env --dest /root/.env --worker all` (exit `0`)

### 5) AdamW train (3-epoch, Industrial)

- `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east1-d --env-file /root/.env --command 'set -euo pipefail; export PYTHONUNBUFFERED=1; export HF_HUB_ENABLE_HF_TRANSFER=1; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; python -u scripts/run_sid_sft.py --config projects/sid_sft/configs/train/v6e-8/sid_sft_jax_qwen25_1p5b_base_industrial_v6e8_e3_adamw_train.yaml --run-mode train'` (exit `0`)
- W&B run: `https://wandb.ai/johntitordemon2036/minionerec-sid-sft/runs/nil84on8`

### 6) AdamW eval (dp8/bs8)

- `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east1-d --env-file /root/.env --command 'set -euo pipefail; export PYTHONUNBUFFERED=1; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; python -u scripts/run_sid_sft.py --config projects/sid_sft/configs/eval/v6e-8/sid_sft_jax_qwen25_1p5b_base_industrial_v6e8_e3_adamw_last_eval_dp8_bs8.yaml --run-mode eval'` (exit `0`)
- W&B run: `https://wandb.ai/johntitordemon2036/minionerec-sid-sft/runs/boct4a05`

### 7) Muon train attempt (preempted; partial)

- `scripts/ssh_tpu_vm_root.sh --name sid-sft-muon-v6e-8-260130024717 --zone us-east5-b --env-file /root/.env --command 'set -euo pipefail; export PYTHONUNBUFFERED=1; export HF_HUB_ENABLE_HF_TRANSFER=1; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; python -u scripts/run_sid_sft.py --config projects/sid_sft/configs/train/v6e-8/sid_sft_jax_qwen25_1p5b_base_industrial_v6e8_e3_muon_lr3e3_aux3e4_train.yaml --run-mode train'` (TPU `PREEMPTED` mid-run; did not complete)
- W&B run (partial): `https://wandb.ai/johntitordemon2036/minionerec-sid-sft/runs/8ea5iqmw`

## Expected Result

- AdamW train + eval exit `0` and log to W&B.
- Muon train + eval should exit `0` and log to W&B (requires a stable TPU; spot preemption can prevent completion).

## Troubleshooting

- dp8 eval + padded-vocab checkpoint mismatch:
  - If you see `ScopeParamShapeError ... expected vocab ... got padded_vocab ...`, ensure you are on commit `5f53cca` or later (supports shrinking padded vocab when loading a checkpoint).

