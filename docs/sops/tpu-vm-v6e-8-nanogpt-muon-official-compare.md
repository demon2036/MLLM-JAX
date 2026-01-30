# SOP: TPU v6e-8 nanoGPT Tiny Shakespeare Muon vs AdamW (W&B online)

- **Title**: SOP: TPU v6e-8 nanoGPT Tiny Shakespeare Muon vs AdamW (W&B online)
- **Prereqs**:
  - gcloud auth + TPU quota for `v6e-8`
  - Repo pushed to GitHub (no SCP for code)
  - `/root/.env` on TPU VM contains `WANDB_API_KEY`
- **Environment (verified)**:
  - Commit: `nano-gpt` / `origin/nano-gpt-sft` @ `ba1752e`
  - TPU runtime: `v6e-ubuntu-2404`
  - JAX: `0.9.0` (`jax.default_backend()==tpu`)
  - W&B: `0.24.1` (mode `online`)

## Goal

- Run the nanoGPT Tiny Shakespeare (char) “Muon official params” config and the corresponding AdamW baseline config on TPU v6e-8, both logging to W&B online, then compare best `eval/loss` and the step it occurs.

## Notes

- On 2026-01-30, multiple on-demand v6e-8 allocations failed due to capacity/quota; a spot TPU in `us-east5-b` worked.

## Steps (commands actually run)

### 0) Create a v6e-8 TPU VM (spot)

- `TPU_NAME="nanogpt-muon-v6e-8-spot-260130061859"; ./scripts/create_tpu_vm.sh --type v6e-8 --zone us-east5-b --name "$TPU_NAME" --spot` (exit `0`)

### 1) Bootstrap Miniconda env

- `./scripts/bootstrap_miniconda_on_tpu_vm.sh --name "$TPU_NAME" --zone us-east5-b --env-name mllm-jax --python 3.12` (exit `0`)

### 2) Sync repo to branch + install deps

- Clone + checkout:
  - `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east5-b --command 'set -euo pipefail; if [ ! -d /root/MLLM-JAX/.git ]; then git clone https://github.com/demon2036/MLLM-JAX.git /root/MLLM-JAX; fi; cd /root/MLLM-JAX; git fetch --all --prune; git checkout john/sid-sft-muon-20260130; git pull --ff-only; git rev-parse --short HEAD'` (exit `0`)
- Install TPU deps (JAX TPU + torch CPU + requirements):
  - `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east5-b --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; python -m pip install -U pip; python -m pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; python -m pip install -U torch --index-url https://download.pytorch.org/whl/cpu; cd /root/MLLM-JAX; python -m pip install -U -r requirements-tpu.txt; python -m pip install -U fire pandas; python - <<\"PY\"\nimport jax\nprint(\"jax\", jax.__version__, \"backend\", jax.default_backend(), \"devices\", len(jax.devices()))\nPY'` (exit `0`)

### 3) Sync W&B `.env` to TPU

- `./scripts/sync_env_to_tpu_vm.sh --name "$TPU_NAME" --zone us-east5-b --src .env --dest /root/.env --worker all` (exit `0`)

### 4) AdamW baseline run (official baseline params)

- `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east5-b --env-file /root/.env --command 'set -euo pipefail; export PYTHONUNBUFFERED=1; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; python -u projects/nano_gpt/run.py --config projects/nano_gpt/configs/tinyshakespeare_char_v6e_8_adamw_muon_official_baseline.yaml'` (exit `0`)
- W&B run: `https://wandb.ai/johntitordemon2036/nano-gpt-jax/runs/k4hj07in`
- Best eval (from printed JSON): `best_eval_loss=1.48699 @ step=3250`

### 5) Muon run (official params)

- `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east5-b --env-file /root/.env --command 'set -euo pipefail; export PYTHONUNBUFFERED=1; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; python -u projects/nano_gpt/run.py --config projects/nano_gpt/configs/tinyshakespeare_char_v6e_8_muon_official.yaml'` (exit `0`)
- W&B run: `https://wandb.ai/johntitordemon2036/nano-gpt-jax/runs/4eut52t2`
- Best eval (from printed JSON): `best_eval_loss=1.45772 @ step=2250`

### 6) Delete TPU VM

- `./scripts/delete_tpu_vm.sh --name "$TPU_NAME" --zone us-east5-b` (exit `0`)

## Expected Result

- Both runs finish (exit `0`) and sync metrics to W&B online.
- Muon run reaches a lower best `eval/loss` than the AdamW baseline (on 2026-01-30 it did: `1.4577` vs `1.4870`).

