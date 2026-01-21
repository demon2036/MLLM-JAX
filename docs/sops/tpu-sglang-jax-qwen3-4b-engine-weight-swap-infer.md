# TPU: sglang-jax Qwen3-4B weight-swap inference (Engine param replacement)

- **Title**: SOP: Run Qwen/Qwen3-4B on TPU via sglang-jax, then swap weights into a dummy Engine and generate “你是谁”
  **Prereqs**:
  - Local machine has `gcloud` authenticated and project set.
  - TPU VM v4-8 quota in target zone.
  - TPU VM runtime: `tpu-ubuntu2204-base`.
  - This repo pushed to GitHub (for TPU Git sync).
  - **Windows note**: `gcloud ... tpu-vm ssh` may use plink and require hostkey handling.
  **Environment (observed in this run)**:
  - Project: `civil-rarity-482610-s5`
  - Zone: `us-central2-b`
  - TPU: `v4-8`
  - TPU VM OS: Ubuntu `22.04.2`
  - Conda env: `sglang-jax` (Python `3.12.12`)
  - JAX: `0.8.1` (TPU backend)
  - sglang-jax commit (pinned): `bd09a87fc6e86c21ce14edd66948ac5dea3a4360`
  - Repo commit (runner script): `d70e7ff`

## Steps

### 1) Create a v4-8 TPU VM (local)

Commands run:

```bash
gcloud auth list --format='table(account,status)'
gcloud config get-value project
gcloud compute tpus accelerator-types list --zone=us-central2-b --format='value(name)'
gcloud alpha compute tpus tpu-vm create mllm-jax-v4-8-260121114749 \
  --project=civil-rarity-482610-s5 \
  --zone=us-central2-b \
  --accelerator-type=v4-8 \
  --version=tpu-ubuntu2204-base \
  --spot \
  --quiet
gcloud alpha compute tpus tpu-vm describe mllm-jax-v4-8-260121114749 \
  --project=civil-rarity-482610-s5 \
  --zone=us-central2-b \
  --format='value(state,acceleratorType)'
```

### 2) SSH to TPU VM as root (local)

Commands run (Windows/plink hostkey pin for this TPU instance):

```bash
gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v4-8-260121114749 \
  --project=civil-rarity-482610-s5 \
  --zone=us-central2-b \
  --quiet \
  --ssh-flag='-hostkey' \
  --ssh-flag='SHA256:ct25E/bBAKMPmAYdkuZUesmzNqZWV7XzseQKxTkc26o' \
  --command 'whoami; hostname; cat /etc/os-release | head -n 5'
```

### 3) Bootstrap Python 3.12 + JAX TPU (on TPU VM)

Commands run:

```bash
# Install Miniconda
curl -fsSL -o /root/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /root/miniconda.sh -b -p /root/miniconda3
rm -f /root/miniconda.sh

# Create env
source /root/miniconda3/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
conda create -y -n sglang-jax python=3.12
conda activate sglang-jax
pip install -U pip

# Install JAX TPU
pip install -U jax[tpu]==0.8.1 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python -c 'import jax; print(jax.__version__); print(jax.device_count()); print(jax.local_device_count())'

# Optional: enable faster HF downloads
pip install -U huggingface-hub==0.34.3
pip install -U hf_transfer
```

### 4) Clone repos on TPU via Git (no SCP)

Commands run:

```bash
# Clone this repo
git clone https://github.com/demon2036/MLLM-JAX.git /root/MLLM-JAX
cd /root/MLLM-JAX
git fetch --all --prune
git checkout d70e7ff
mkdir -p /root/MLLM-JAX/workdir

# Clone sglang-jax into local scratch
git clone https://github.com/sgl-project/sglang-jax.git /root/MLLM-JAX/workdir/sglang-jax
cd /root/MLLM-JAX/workdir/sglang-jax
git fetch --all --prune
git checkout bd09a87fc6e86c21ce14edd66948ac5dea3a4360
```

### 5) Install sglang-jax (editable) on TPU

Commands run:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sglang-jax
pip install -e /root/MLLM-JAX/workdir/sglang-jax/python
python -c 'import sgl_jax; from sgl_jax.version import __version__; print(__version__)'
```

### 6) Run the Engine weight-swap inference script (on TPU)

Commands run:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sglang-jax
cd /root/MLLM-JAX
PYTHONUNBUFFERED=1 timeout 7200 python -u tests/run_sglang_jax_qwen3_4b_param_swap.py
```

## Expected Result

- Script prints:
  - `{"phase": "engine_ready_dummy", ...}` (Engine initialized with dummy weights)
  - `{"phase": "weights_swapped", ...}` (weights loaded from `Qwen/Qwen3-4B` safetensors and injected into `model_runner.model_state_leaves`)
  - `{"phase": "generate_result", "prompt": "你是谁", ...}` with a non-empty `text`
- Exit code `0`

## How sglang-jax does this (minimal architecture notes)

- Engine composition (see sglang-jax `python/sgl_jax/srt/entrypoints/engine.py`):
  - `Engine` launches `TokenizerManager` + `Scheduler` + `DetokenizerManager`.
- Weight loading path:
  - `ModelRunner.load_model()` uses `model_loader.load_model()` which calls `Qwen3ForCausalLM.load_weights()` via `WeightLoader.load_weights_from_safetensors()`.
  - Qwen3 HF key mapping lives in `python/sgl_jax/srt/models/qwen3.py` (`_create_qwen3_weight_mappings`).
- Why runtime replacement works:
  - `ModelRunner.initialize_jit()` exports `model_state_leaves` (flattened nnx state). The compiled function reconstructs the model state each call using these leaves.
  - Replacing `model_state_leaves` with a same-structure leaf list effectively swaps parameters without changing compiled shapes.

## Troubleshooting

- **TPU “already in use by pid …”**
  - Find and kill the stuck process:
    - `ps -fp <pid>`
    - `kill -9 <pid>`
- **Windows `gcloud tpu-vm ssh` host key mismatch (plink prompt)**
  - Re-run once to see the new `SHA256:...` fingerprint, then pass:
    - `--ssh-flag='-hostkey' --ssh-flag='SHA256:<fingerprint>'`
  - Or remove the cached key and reconnect.

## References

- `docs/sops/tpu-vm-create-v4-8-or-v6e-8.md`
- `docs/sops/tpu-vm-repo-sync.md`
- sglang-jax repo: https://github.com/sgl-project/sglang-jax
