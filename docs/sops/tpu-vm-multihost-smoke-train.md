# TPU VM Multi-host Smoke Train (v4-16)

- **Title**: SOP: Create v4-16 (2-host) TPU VM and run MLLM-JAX smoke train
  **Prereqs**: gcloud configured/authenticated; TPU v4 capacity/quota in the target zone; outbound network for Conda/PyPI/Hugging Face downloads
  **Steps**:
  - Confirm you are in the right project:
    - `gcloud config get-value project`
  - (Optional) Find existing TPU VMs across zones (includes zone in output):
    - `gcloud compute tpus locations list --format='value(locationId)' | xargs -P6 -I{} bash -lc 'z="{}"; gcloud compute tpus tpu-vm list --zone="$z" --format="value(name,acceleratorType,state)" 2>/dev/null | sed "s|^|$z\t|"'`
  - Create a 2-host TPU VM (v4-16, spot):
    - `TPU_NAME=mllm-jax-v4-16-260117125029; ZONE=us-central2-b; ACCELERATOR_TYPE=v4-16; RUNTIME_VERSION=tpu-ubuntu2204-base; gcloud alpha compute tpus tpu-vm create "$TPU_NAME" --zone="$ZONE" --accelerator-type="$ACCELERATOR_TYPE" --version="$RUNTIME_VERSION" --spot --quiet`
  - Install Miniconda on all workers:
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --worker=all --quiet --command 'set -euo pipefail; if [ ! -d /root/miniconda3 ]; then curl -fsSL -o /root/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; bash /root/miniconda.sh -b -p /root/miniconda3; rm -f /root/miniconda.sh; fi; /root/miniconda3/bin/conda --version'`
  - Create a conda env and upgrade pip:
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --worker=all --quiet --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true; conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true; ENV_NAME=mllm-jax; if ! conda env list | awk "{print \\$1}" | grep -qx "$ENV_NAME"; then conda create -y -n "$ENV_NAME" python=3.12; fi; conda activate "$ENV_NAME"; python --version; pip install -U pip'`
  - Install runtime deps on all workers:
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --worker=all --quiet --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; python -m pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; python -m pip install -U flax optax chex transformers datasets tqdm wandb math_verify "huggingface_hub[hf_transfer]" cloud_tpu_client webdataset einops; python -m pip install -U torch --index-url https://download.pytorch.org/whl/cpu; python -c "import jax; print(\\"jax\\", jax.__version__, \\"backend\\", jax.default_backend(), \\"process\\", jax.process_index(), \\"/\\", jax.process_count(), \\"device_count\\", jax.device_count(), \\"local\\", len(jax.local_devices()))"'`
  - Clone the repo on all workers:
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --worker=all --quiet --command 'set -euo pipefail; if [ ! -d /root/MLLM-JAX/.git ]; then rm -rf /root/MLLM-JAX; git clone --depth=1 https://github.com/demon2036/MLLM-JAX.git /root/MLLM-JAX; fi; cd /root/MLLM-JAX; git rev-parse --short HEAD'`
  - Work around an unused TF import in older repo snapshots:
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --worker=all --quiet --command 'set -euo pipefail; cd /root/MLLM-JAX; sed -i "/from tensorflow\\.python\\.framework\\.tensor import DenseSpec/d" MLLM_JAX/language/llama/llama.py'`
  - Run 1-step smoke train on all workers:
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --worker=all --quiet --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; export PYTHONPATH=/root/MLLM-JAX:${PYTHONPATH:-}; export WANDB_MODE=disabled; export MODEL_PATH=Qwen/Qwen2.5-0.5B; export STEPS=1; export BATCH_SIZE=1; export SEQ_LEN=64; python -u scripts/run_smoke_train_qwen25_7b.py'`
  **Expected Result**: Both workers print `backend=tpu ...` then each prints `step=0 loss=... dt=...s`.
  **Troubleshooting**:
  - If `jax.distributed.initialize()` hangs: ensure the run command uses `--worker=all` and the TPU is `READY`.
  - If `ModuleNotFoundError: No module named 'MLLM_JAX'`: keep the `PYTHONPATH=/root/MLLM-JAX` export (or install the repo as a package).
  - If `ModuleNotFoundError` for `webdataset` or `einops`: re-run the deps install step.
  **References**: `docs/sops/tpu-vm-lifecycle.md`, `docs/sops/tpu-vm-bootstrap.md`

