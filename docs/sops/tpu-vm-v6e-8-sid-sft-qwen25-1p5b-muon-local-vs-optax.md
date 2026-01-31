# SOP: TPU v6e-8 Qwen2.5-1.5B SID SFT — Muon (local) vs Optax

- **Title**: SOP: Run a strict old-vs-optax Muon comparison on Qwen2.5-1.5B SID SFT (Industrial) on TPU v6e-8 (W&B online)
  **Prereqs**: `gcloud` authenticated + project set; W&B API key available in a local `.env`; TPU has egress to HuggingFace; repo changes pushed to GitHub (for running non-main branches)
  **Environment (verified)**:
  - TPU VM: `v6e-8` spot, Ubuntu `24.04` runtime `v6e-ubuntu-2404`
  - Python: `3.12` (conda env `mllm-jax`)
  - JAX: `0.9.0` + `libtpu 0.0.34`
  - Optax: `0.2.6`

## Steps (commands actually run)

- Create TPU (on-demand may fail with capacity; spot works):
  - `TPU_NAME="check-muon-v6e-8-spot-$(date +%y%m%d%H%M%S)"; ./scripts/create_tpu_vm.sh --type v6e-8 --zone us-east5-b --name "$TPU_NAME" --spot`

- Bootstrap Miniconda + env:
  - `./scripts/bootstrap_miniconda_on_tpu_vm.sh --name "$TPU_NAME" --zone us-east5-b --env-name mllm-jax --python 3.12`

- Clone repo to TPU (via Git) + install deps:
  - `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east5-b --command 'set -euo pipefail; REPO_URL=https://github.com/demon2036/MLLM-JAX.git; REPO_DIR=/root/MLLM-JAX; if [ ! -d "$REPO_DIR/.git" ]; then git clone "$REPO_URL" "$REPO_DIR"; fi; cd "$REPO_DIR"; git fetch --all --prune; git status -sb'`
  - `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east5-b --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; python -m pip install -U pip; python -m pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; python -m pip install -U torch --index-url https://download.pytorch.org/whl/cpu; cd /root/MLLM-JAX; python -m pip install -U -r requirements-tpu.txt; python -m pip install -U fire pandas'`

- Ensure upstream `MiniOneRec` exists under the repo’s ignored `workdir/`:
  - `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east5-b --command 'set -euo pipefail; cd /root/MLLM-JAX; mkdir -p workdir; if [ ! -d workdir/MiniOneRec/.git ]; then git clone https://github.com/AkaliKong/MiniOneRec workdir/MiniOneRec; fi'`

- Sync local `.env` to TPU (W&B online):
  - `./scripts/sync_env_to_tpu_vm.sh --name "$TPU_NAME" --zone us-east5-b --src ../nano-gpt/.env --dest /root/.env --worker all`

- Create two TPU worktrees (same env, same configs):
  - `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east5-b --command 'set -euo pipefail; cd /root/MLLM-JAX; git fetch --all --prune; git worktree add /root/MLLM-JAX-localmuon 0001ab7; git worktree add /root/MLLM-JAX-optaxmuon 9aebbfc; rm -rf /root/MLLM-JAX-localmuon/workdir /root/MLLM-JAX-optaxmuon/workdir; ln -s /root/MLLM-JAX/workdir /root/MLLM-JAX-localmuon/workdir; ln -s /root/MLLM-JAX/workdir /root/MLLM-JAX-optaxmuon/workdir; git worktree list'`

- Run train + eval (same YAMLs, different worktree dirs so outputs don’t collide):
  - Train config:
    - `projects/sid_sft/configs/train/v6e-8/sid_sft_jax_qwen25_1p5b_base_industrial_v6e8_e3_muon_lr3e3_aux3e4_train.yaml`
  - Eval config (loads `save_last`):
    - `projects/sid_sft/configs/eval/v6e-8/sid_sft_jax_qwen25_1p5b_base_industrial_v6e8_e3_muon_lr3e3_aux3e4_last_eval_dp8_bs8.yaml`

  - Local/custom Muon (commit `0001ab7`):
    - Train:
      - `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east5-b --env-file /root/.env --command 'set -euo pipefail; export PYTHONUNBUFFERED=1; export HF_HUB_ENABLE_HF_TRANSFER=1; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX-localmuon; ./scripts/run_sid_sft.sh --config projects/sid_sft/configs/train/v6e-8/sid_sft_jax_qwen25_1p5b_base_industrial_v6e8_e3_muon_lr3e3_aux3e4_train.yaml --run-mode train'`
    - Eval:
      - `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east5-b --env-file /root/.env --command 'set -euo pipefail; export PYTHONUNBUFFERED=1; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX-localmuon; ./scripts/run_sid_sft.sh --config projects/sid_sft/configs/eval/v6e-8/sid_sft_jax_qwen25_1p5b_base_industrial_v6e8_e3_muon_lr3e3_aux3e4_last_eval_dp8_bs8.yaml --run-mode eval'`

  - Optax Muon (commit `9aebbfc`):
    - Train:
      - `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east5-b --env-file /root/.env --command 'set -euo pipefail; export PYTHONUNBUFFERED=1; export HF_HUB_ENABLE_HF_TRANSFER=1; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX-optaxmuon; ./scripts/run_sid_sft.sh --config projects/sid_sft/configs/train/v6e-8/sid_sft_jax_qwen25_1p5b_base_industrial_v6e8_e3_muon_lr3e3_aux3e4_train.yaml --run-mode train'`
    - Eval:
      - `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east5-b --env-file /root/.env --command 'set -euo pipefail; export PYTHONUNBUFFERED=1; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX-optaxmuon; ./scripts/run_sid_sft.sh --config projects/sid_sft/configs/eval/v6e-8/sid_sft_jax_qwen25_1p5b_base_industrial_v6e8_e3_muon_lr3e3_aux3e4_last_eval_dp8_bs8.yaml --run-mode eval'`

## Verified results (2026-01-31)

- Train final loss (`train.final_loss` in `run_summary.json`, 234 steps, effective_bs=1024):
  - local/custom Muon (`0001ab7`): `0.347595`
  - optax Muon (`9aebbfc`): `0.341043`
- Eval (dp8, bs=8, beams=50, samples=4533, invalid=0):
  - local/custom Muon (`0001ab7`):
    - HR@K: 1=`0.07633`, 3=`0.10324`, 5=`0.11824`, 10=`0.14935`, 20=`0.18509`, 50=`0.23914`
    - NDCG@K: 1=`0.07633`, 3=`0.09201`, 5=`0.09814`, 10=`0.10809`, 20=`0.11709`, 50=`0.12781`
  - optax Muon (`9aebbfc`):
    - HR@K: 1=`0.07942`, 3=`0.10876`, 5=`0.12619`, 10=`0.15200`, 20=`0.19016`, 50=`0.24487`
    - NDCG@K: 1=`0.07942`, 3=`0.09649`, 5=`0.10368`, 10=`0.11190`, 20=`0.12155`, 50=`0.13252`

## Cleanup

- Delete TPU to stop billing:
  - `./scripts/delete_tpu_vm.sh --name "$TPU_NAME" --zone us-east5-b`

