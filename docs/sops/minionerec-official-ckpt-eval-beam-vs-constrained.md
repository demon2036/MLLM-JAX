# SOP: Eval official MiniOneRec HF checkpoints — pure beam vs constrained beam (TPU)

- **Title**: SOP: Eval **official MiniOneRec** HF checkpoints with **pure beam search** vs **SID-trie constrained beam search** on TPU (`v4-8`)
- **Prereqs**: `gcloud` authenticated; TPU API enabled; a local `.env` with `WANDB_API_KEY`; network access (HF download)
- **Environment (verified)**:
  - TPU: `v4-8` (spot), zone `us-central2-b`, runtime `tpu-ubuntu2204-base`
  - Python: `3.12.12` (conda env `mllm-jax`)
  - JAX: `0.9.0` + `jaxlib 0.9.0` + `libtpu 0.0.34`
  - Repo: `demon2036/MLLM-JAX` branch `john/sft-muon-20260129`, commit `2c1413d`

## Steps (commands actually run)

### 1) Create TPU VM (spot)

- Attempt `v6e-8` spot (failed due to spot quota=0 in `us-central2-b`):
  - `./scripts/create_tpu_vm.sh --type v6e-8 --zone us-central2-b --name "mllm-jax-v6e-8-minionerec-official-beam-260130022623"`
- Create a `v4-8` spot instead:
  - `TPU_NAME="mllm-jax-v4-8-minionerec-official-beam-260130022636"`
  - `./scripts/create_tpu_vm.sh --type v4-8 --zone us-central2-b --name "$TPU_NAME"`

### 2) Bootstrap conda + clone repos on TPU

- Bootstrap Miniconda + Python env:
  - `./scripts/bootstrap_miniconda_on_tpu_vm.sh --name "$TPU_NAME" --zone us-central2-b --env-name mllm-jax --python 3.12`

- Clone this repo and checkout the branch:
  - `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-central2-b --command 'set -euo pipefail; if [ ! -d /root/MLLM-JAX/.git ]; then git clone https://github.com/demon2036/MLLM-JAX.git /root/MLLM-JAX; fi; cd /root/MLLM-JAX; git fetch --all; git checkout john/sft-muon-20260129; git pull; git rev-parse --short HEAD'`

- Clone upstream MiniOneRec into `workdir/`:
  - `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-central2-b --command 'set -euo pipefail; cd /root/MLLM-JAX; mkdir -p workdir; if [ ! -d workdir/MiniOneRec/.git ]; then git clone https://github.com/AkaliKong/MiniOneRec workdir/MiniOneRec; fi; cd workdir/MiniOneRec; git rev-parse --short HEAD'`

### 3) Install TPU deps + sync W&B key

- Install JAX TPU + deps:
  - `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-central2-b --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; python -m pip install -U pip; python -m pip install -U \"jax[tpu]\" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; python -m pip install -U torch --index-url https://download.pytorch.org/whl/cpu; cd /root/MLLM-JAX; python -m pip install -U -r requirements-tpu.txt; python - <<\"PY\"\nimport jax, jaxlib\nprint(\"jax\", jax.__version__, \"jaxlib\", jaxlib.__version__)\nprint(\"backend\", jax.default_backend())\nprint(\"process\", jax.process_index(), \"/\", jax.process_count())\nprint(\"device_count\", jax.device_count(), \"local\", jax.local_device_count())\nPY'`

- Sync local secrets (`WANDB_API_KEY`) to TPU:
  - `./scripts/sync_env_to_tpu_vm.sh --name "$TPU_NAME" --zone us-central2-b --src .env --dest /root/.env --worker all`

### 4) Download official HF checkpoints (Industrial/Office)

- Download `kkknight/MiniOneRec` ckpt subfolders into `workdir/hf_ckpts/kkknight_MiniOneRec/`:
  - `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-central2-b --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; export HF_HUB_ENABLE_HF_TRANSFER=1; cd /root/MLLM-JAX; python - <<\"PY\"\nfrom huggingface_hub import snapshot_download\nsnapshot_download(repo_id=\"kkknight/MiniOneRec\", allow_patterns=[\"Industrial_ckpt/*\"], local_dir=\"workdir/hf_ckpts/kkknight_MiniOneRec\")\nsnapshot_download(repo_id=\"kkknight/MiniOneRec\", allow_patterns=[\"Office_ckpt/*\"], local_dir=\"workdir/hf_ckpts/kkknight_MiniOneRec\")\nPY'`

### 5) Run eval (constrained vs pure beam)

- Industrial (constrained, dp4):
  - `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-central2-b --env-file /root/.env --command 'set -euo pipefail; export PYTHONUNBUFFERED=1; rm -f /tmp/libtpu_lockfile || true; export HF_HUB_ENABLE_HF_TRANSFER=1; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; ./scripts/run_sid_sft.sh --config projects/minionerec/sft/configs/official_eval/sid_sft_jax_eval_official_minionerec_industrial_ckpt_dp4.yaml --run-mode eval'`
  - W&B: https://wandb.ai/johntitordemon2036/minionerec-sid-sft/runs/1r7lmu5d

- Industrial (pure beam, dp4):
  - `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-central2-b --env-file /root/.env --command 'set -euo pipefail; export PYTHONUNBUFFERED=1; rm -f /tmp/libtpu_lockfile || true; export HF_HUB_ENABLE_HF_TRANSFER=1; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; ./scripts/run_sid_sft.sh --config projects/minionerec/sft/configs/official_eval/sid_sft_jax_eval_official_minionerec_industrial_ckpt_dp4_pure_beam.yaml --run-mode eval'`
  - W&B: https://wandb.ai/johntitordemon2036/minionerec-sid-sft/runs/gndfjwwa

- Office (constrained, dp4):
  - `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-central2-b --env-file /root/.env --command 'set -euo pipefail; export PYTHONUNBUFFERED=1; rm -f /tmp/libtpu_lockfile || true; export HF_HUB_ENABLE_HF_TRANSFER=1; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; ./scripts/run_sid_sft.sh --config projects/minionerec/sft/configs/official_eval/sid_sft_jax_eval_official_minionerec_office_ckpt_dp4.yaml --run-mode eval'`
  - W&B: https://wandb.ai/johntitordemon2036/minionerec-sid-sft/runs/qfhr8gk1

- Office (pure beam, dp4):
  - `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-central2-b --env-file /root/.env --command 'set -euo pipefail; export PYTHONUNBUFFERED=1; rm -f /tmp/libtpu_lockfile || true; export HF_HUB_ENABLE_HF_TRANSFER=1; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; ./scripts/run_sid_sft.sh --config projects/minionerec/sft/configs/official_eval/sid_sft_jax_eval_official_minionerec_office_ckpt_dp4_pure_beam.yaml --run-mode eval'`
  - W&B: https://wandb.ai/johntitordemon2036/minionerec-sid-sft/runs/pzxpyyxj

## Expected result

- All 4 eval runs exit `0` and write `eval_predictions.json` + `eval_predictions.metrics.json` under their `output_dir`.
- Constrained decoding matches the “official” verified metrics (invalid=0).
- Pure beam search shows lower HR/NDCG due to invalid SID predictions.

## Verified results (2026-01-30)

- Industrial (constrained): HR@10=0.15839, NDCG@10=0.11772, invalid=0
- Industrial (pure beam): HR@10=0.12530, NDCG@10=0.10045, invalid=155129 (~68.44% of 4533*50)
- Office (constrained): HR@10=0.15845, NDCG@10=0.12462, invalid=0
- Office (pure beam): HR@10=0.11447, NDCG@10=0.09882, invalid=175011 (~71.93% of 4866*50)

## Troubleshooting

- `v6e-8` spot quota is 0 in your zone:
  - Fall back to `v4-8` spot, or request v6e spot quota.
- HF Hub throttling:
  - Provide `HF_TOKEN` if you hit rate limits.
- TPU busy (`libtpu_lockfile`):
  - Stop the stale job and `rm -f /tmp/libtpu_lockfile`.

