# SOP: TPU VM v4-8 smoke-run `scripts/run_grpo_gsm8k_training.py` with `rollout.backend=naive`

- **Title**: SOP: TPU VM v4-8 smoke-run GRPO/GSM8K runner with `rollout.backend=naive`
  **Prereqs**: `gcloud` installed + authenticated; TPU quota/capacity; outbound internet from TPU VM (HF + datasets); repo pushed to GitHub
  **Environment (verified)**:
  - Project: `civil-rarity-482610-s5`
  - Zone: `us-central2-b`
  - TPU VM: `v4-8` (single host), runtime `tpu-ubuntu2204-base`
  - Conda env: `mllm-jax`; Python `3.12.12`
  - JAX: `0.8.2`, jaxlib `0.8.2`, libtpu `0.0.32`
  - JAX devices: `4` (megacore)
  - Git ref on TPU (verified): `efa7631`

## Steps (commands actually used)

### 0) Local: record the commit you want to run on TPU

- `cd /home/john/works/MLLM-JAX-mllm-jax-sglang`
- `git rev-parse --short HEAD`  # -> `efa7631`

### 1) Create a TPU VM

Attempted spot first (preempted immediately due to a maintenance event):

- `cd /home/john/works/MLLM-JAX-mllm-jax-sglang`
- `scripts/create_tpu_vm.sh --type v4-8 --zone us-central2-b`  # -> `mllm-jax-v4-8-260118175721`
- `gcloud alpha compute tpus tpu-vm describe mllm-jax-v4-8-260118175721 --zone us-central2-b --project civil-rarity-482610-s5 --format='yaml(state,health,healthDescription)'`
- `scripts/delete_tpu_vm.sh --name mllm-jax-v4-8-260118175721 --zone us-central2-b --project civil-rarity-482610-s5`

Create on-demand (verified run):

- `cd /home/john/works/MLLM-JAX-mllm-jax-sglang`
- `scripts/create_tpu_vm.sh --type v4-8 --zone us-central2-b --on-demand`  # -> `mllm-jax-v4-8-260118180552`
- `gcloud alpha compute tpus tpu-vm describe mllm-jax-v4-8-260118180552 --zone us-central2-b --project civil-rarity-482610-s5 --format='value(state,acceleratorType)'`

### 2) Bootstrap Miniconda + conda env on the TPU VM

- `cd /home/john/works/MLLM-JAX-mllm-jax-sglang`
- `scripts/bootstrap_miniconda_on_tpu_vm.sh --name mllm-jax-v4-8-260118180552 --zone us-central2-b --project civil-rarity-482610-s5`

### 3) Clone/pull the repo on the TPU VM (git-sync, no SCP) and checkout the commit

- `cd /home/john/works/MLLM-JAX-mllm-jax-sglang`
- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260118180552 --zone us-central2-b --project civil-rarity-482610-s5 --command 'set -euo pipefail; REPO_URL=https://github.com/demon2036/MLLM-JAX.git; REPO_DIR=/root/MLLM-JAX; if [ ! -d \"$REPO_DIR/.git\" ]; then rm -rf \"$REPO_DIR\"; git clone \"$REPO_URL\" \"$REPO_DIR\"; fi; cd \"$REPO_DIR\"; git fetch --all --prune; git checkout efa7631; git status -sb; echo \"HEAD=$(git rev-parse --short HEAD)\"'`

### 4) Install TPU runtime deps (JAX + requirements)

- `cd /home/john/works/MLLM-JAX-mllm-jax-sglang`
- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260118180552 --zone us-central2-b --project civil-rarity-482610-s5 --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; python -V; python -m pip install -U pip; python -m pip install -U \"jax[tpu]\" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; python -m pip install -U torch --index-url https://download.pytorch.org/whl/cpu; cd /root/MLLM-JAX; python -m pip install -U -r requirements-tpu.txt; python - <<\"PY\"\nimport jax, jaxlib\nprint(\"jax\", jax.__version__, \"jaxlib\", jaxlib.__version__)\nprint(\"backend\", jax.default_backend())\nprint(\"process\", jax.process_index(), \"/\", jax.process_count())\nprint(\"device_count\", jax.device_count(), \"local\", len(jax.local_devices()))\nPY'`

### 5) Run a 1-step smoke training with `ROLLOUT_BACKEND=naive`

- `cd /home/john/works/MLLM-JAX-mllm-jax-sglang`
- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260118180552 --zone us-central2-b --project civil-rarity-482610-s5 --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; export HF_HUB_ENABLE_HF_TRANSFER=1; export WANDB_MODE=disabled; export TOKENIZERS_PARALLELISM=false; export MODEL_PATH=\"Qwen/Qwen2.5-7B-Instruct\"; export STEPS=1; export ROLLOUT_BACKEND=naive; export BATCH_SIZE=1; export NUM_PRE_Q=4; export GLOBAL_LENGTH=512; export MAX_LENGTH_SAMPLE=64; export PPO_EPOCHS=1; export GRAD_ACCUM_STEPS=1; export BETA=0.0; python -u scripts/run_grpo_gsm8k_training.py'`

Observed success line:

- `step=0 loss=0.000000 entropy=0.2734 reward_mean=0.1250 dt=118.86s`

### 6) Delete the TPU VM (to stop billing)

- `cd /home/john/works/MLLM-JAX-mllm-jax-sglang`
- `scripts/delete_tpu_vm.sh --name mllm-jax-v4-8-260118180552 --zone us-central2-b --project civil-rarity-482610-s5`

## Expected Result

- TPU VM reaches `READY`.
- `python -u scripts/run_grpo_gsm8k_training.py` prints an effective config containing `rollout.backend: naive`.
- The run prints `step=0 ...` and exits with code `0` (no Python tracebacks).

## Troubleshooting

- TPU VM `state: PREEMPTED` (spot):
  - Delete it and re-create (or use `--on-demand`).
- SSH hangs or times out:
  - Check TPU state: `gcloud alpha compute tpus tpu-vm describe ... --format='value(state)'`.
- `Unable to initialize backend 'tpu' ... already in use`:
  - `rm -f /tmp/libtpu_lockfile` and ensure no other training process is running.

## References

- `answers/rollout-backend-abstraction.md`
- `docs/sops/tpu-vm-repo-sync.md`
- `docs/sops/tpu-vm-create-v4-8-or-v6e-8.md`
- `scripts/create_tpu_vm.sh`
- `scripts/delete_tpu_vm.sh`
- `scripts/bootstrap_miniconda_on_tpu_vm.sh`
- `scripts/ssh_tpu_vm_root.sh`
- `scripts/run_grpo_gsm8k_training.py`

