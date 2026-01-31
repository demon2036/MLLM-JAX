# SOP: TPU v4-8 MiniOneRec SID SFT (Muon) after plugins refactor

- **Title**: SOP: Run Qwen2.5-1.5B MiniOneRec SID SFT with Muon on TPU v4-8 and report HR@K/NDCG@K
- **Prereqs**: `gcloud` authenticated + project set; TPU VM egress to HuggingFace; repo changes pushed to GitHub (if not on `main`)
- **Environment (verified)**:
  - Date: 2026-01-31
  - TPU VM: `v4-8` spot, zone `us-central2-b`, runtime `tpu-ubuntu2204-base`
  - Python: `3.12.12` (conda env `mllm-jax`)
  - JAX: `0.9.0` + `jaxlib 0.9.0`, `libtpu 0.0.34`, `device_count=4`
  - Repo branch/commit: `nano-gpt-sft` @ `68c73a1`

## Notes

- W&B: this run used `wandb.mode=online` in YAML, but the TPU VM did not have `WANDB_API_KEY`,
  so `wandb.init(...)` failed and the helper auto-disabled W&B (no crash).
- TPU capacity: `v6e-8` spot was unavailable in `us-east5-b` (no capacity) and spot quota was `0`
  in `us-central2-b`, so we ran on `v4-8`.

## Steps (commands actually run)

### 1) Create TPU VM (v4-8 spot)

- `TPU_NAME="plugins-refactor-sid-sft-muon-260131052355"`
- `./scripts/create_tpu_vm.sh --type v4-8 --zone us-central2-b --name "$TPU_NAME" --spot`

### 2) Bootstrap Miniconda + env

- `./scripts/bootstrap_miniconda_on_tpu_vm.sh --name "$TPU_NAME" --zone us-central2-b --env-name mllm-jax --python 3.12`

### 3) Clone repo on TPU (Git sync, no SCP)

- `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-central2-b --command 'set -euo pipefail; REPO_URL=https://github.com/demon2036/MLLM-JAX.git; REPO_DIR=/root/MLLM-JAX; if [ ! -d \"$REPO_DIR/.git\" ]; then git clone \"$REPO_URL\" \"$REPO_DIR\"; fi; cd \"$REPO_DIR\"; git fetch --all --prune; git checkout nano-gpt-sft; git pull; git rev-parse --short HEAD; git status -sb'`

### 4) Install TPU deps (JAX TPU + torch CPU + repo deps)

- `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-central2-b --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; python -m pip install -U pip; python -m pip install -U \"jax[tpu]\" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; python -m pip install -U torch --index-url https://download.pytorch.org/whl/cpu; cd /root/MLLM-JAX; python -m pip install -U -r requirements-tpu.txt; python -m pip install -U fire pandas pytest; python - <<\"PY\"\nimport jax, jaxlib\nprint(\"jax\", jax.__version__, \"jaxlib\", jaxlib.__version__)\nprint(\"backend\", jax.default_backend())\nprint(\"process\", jax.process_index(), \"/\", jax.process_count())\nprint(\"device_count\", jax.device_count(), \"local\", len(jax.local_devices()))\nPY'`

### 5) Ensure upstream MiniOneRec repo exists under `workdir/`

- `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-central2-b --command 'set -euo pipefail; cd /root/MLLM-JAX; mkdir -p workdir; if [ ! -d workdir/MiniOneRec/.git ]; then git clone https://github.com/AkaliKong/MiniOneRec workdir/MiniOneRec; fi; (cd workdir/MiniOneRec && git rev-parse --short HEAD)'`

### 6) Train (Muon)

- Train config:
  - `projects/sid_sft/configs/train/v4-8/sid_sft_jax_qwen25_1p5b_base_industrial_v4_8_e3_muon_refactor_20260131_train.yaml`
- `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-central2-b --command 'set -euo pipefail; export PYTHONUNBUFFERED=1; export HF_HUB_ENABLE_HF_TRANSFER=1; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; ./scripts/run_sid_sft.sh --config projects/sid_sft/configs/train/v4-8/sid_sft_jax_qwen25_1p5b_base_industrial_v4_8_e3_muon_refactor_20260131_train.yaml --run-mode train'`

### 7) Eval (constrained decoding HR@K/NDCG@K)

- Eval config:
  - `projects/sid_sft/configs/eval/v4-8/sid_sft_jax_qwen25_1p5b_base_industrial_v4_8_e3_muon_refactor_20260131_last_eval_dp4_bs4.yaml`
- `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-central2-b --command 'set -euo pipefail; export PYTHONUNBUFFERED=1; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; ./scripts/run_sid_sft.sh --config projects/sid_sft/configs/eval/v4-8/sid_sft_jax_qwen25_1p5b_base_industrial_v4_8_e3_muon_refactor_20260131_last_eval_dp4_bs4.yaml --run-mode eval'`

## Expected result

- Training writes:
  - `runs/sid_sft_jax_qwen25_1p5b_base_industrial_v4_8_e3_muon_refactor_20260131_train/sft_state_last.msgpack`
  - `runs/sid_sft_jax_qwen25_1p5b_base_industrial_v4_8_e3_muon_refactor_20260131_train/run_summary.json`
- Eval writes:
  - `runs/sid_sft_jax_qwen25_1p5b_base_industrial_v4_8_e3_muon_refactor_20260131_last_eval_dp4_bs4/eval_predictions.json`
  - `runs/sid_sft_jax_qwen25_1p5b_base_industrial_v4_8_e3_muon_refactor_20260131_last_eval_dp4_bs4/eval_predictions.metrics.json`
  - `runs/sid_sft_jax_qwen25_1p5b_base_industrial_v4_8_e3_muon_refactor_20260131_last_eval_dp4_bs4/run_summary.json`

## Verified results (2026-01-31)

- Train (234 steps, effective_bs=1024): final_loss=`0.346560`
- Eval (samples=4533, invalid=0):
  - HR@K: 1=`0.07809`, 3=`0.10346`, 5=`0.11780`, 10=`0.14869`, 20=`0.18685`, 50=`0.24664`
  - NDCG@K: 1=`0.07809`, 3=`0.09271`, 5=`0.09857`, 10=`0.10846`, 20=`0.11806`, 50=`0.12988`

## Run 2: train_eval on validation + save best checkpoint

- Repo branch/commit: `nano-gpt-sft` @ `c2c3e2c`
  - Adds `train.save_best`/`train.save_best_metric` and `eval.split` to `projects/sid_sft`.
- Train+eval config (valid split, beams=20):
  - `projects/sid_sft/configs/train/v4-8/sid_sft_jax_qwen25_1p5b_base_industrial_v4_8_e3_muon_refactor_20260131_train_bestval.yaml`
- Command:
  - `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-central2-b --command 'set -euo pipefail; export PYTHONUNBUFFERED=1; export HF_HUB_ENABLE_HF_TRANSFER=1; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; ./scripts/run_sid_sft.sh --config projects/sid_sft/configs/train/v4-8/sid_sft_jax_qwen25_1p5b_base_industrial_v4_8_e3_muon_refactor_20260131_train_bestval.yaml --run-mode train_eval'`
- Output dir:
  - `runs/sid_sft_jax_qwen25_1p5b_base_industrial_v4_8_e3_muon_refactor_20260131_train_bestval/`
  - Best checkpoint:
    - `runs/sid_sft_jax_qwen25_1p5b_base_industrial_v4_8_e3_muon_refactor_20260131_train_bestval/sft_state_best.msgpack`
- W&B run: `johntitordemon2036/minionerec-sid-sft/runs/jiykssr3` (mode=online)
- Eval (valid split, beams=20, samples=4532, invalid=0):
  - HR@K: 1=`0.08694`, 3=`0.11584`, 5=`0.12842`, 10=`0.15181`, 20=`0.17829`
  - NDCG@K: 1=`0.08694`, 3=`0.10350`, 5=`0.10865`, 10=`0.11617`, 20=`0.12287`

## Cleanup

- Per task requirement, this run did **not** delete the TPU VM.
  When you want to stop billing:
  - `./scripts/delete_tpu_vm.sh --name "$TPU_NAME" --zone us-central2-b`
