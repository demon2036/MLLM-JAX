# SOP: TPU v6e-8 nanoGPT Tiny Shakespeare (char) standard training

- **Title**: SOP: Run nanoGPT Tiny Shakespeare (char-level) standard training on TPU v6e-8
  **Prereqs**:
  - TPU VM reachable (v6e-8)
  - Repo synced to TPU via Git (no code SCP)
  - Conda env exists with JAX TPU wheel + deps installed
  - (Optional) `/root/.env` contains `WANDB_API_KEY` if using W&B online
  **Environment (verified)**:
  - TPU VM: `minionerec-sid-sft-v6e-8-official-260128021636` (zone `europe-west4-a`, 8 devices)
  - OS: Ubuntu 24.04.2 LTS
  - Conda: `conda 25.11.1`
  - Python (conda env `mllm-jax`): `3.12.12`
  - jax/jaxlib: `0.9.0` (`jax.default_backend() == tpu`)
  - flax/optax: `flax 0.12.3`, `optax 0.2.6`
  - wandb: `0.24.0`

## Steps (verified)

### 0) SSH to the TPU VM

This run was executed from Windows using OpenSSH (since `gcloud ... tpu-vm ssh` used PuTTY and prompted for host-key caching).

- `ssh -i ~/.ssh/google_compute_engine -o StrictHostKeyChecking=no root@34.7.101.121`

### 1) Bootstrap Miniconda + conda env

- Install Miniconda (if missing):
  - `if [ ! -d /root/miniconda3 ]; then curl -fsSL -o /root/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; bash /root/miniconda.sh -b -p /root/miniconda3; rm -f /root/miniconda.sh; fi`
- Create env + upgrade pip:
  - `source /root/miniconda3/etc/profile.d/conda.sh`
  - `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true`
  - `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true`
  - `conda create -y -n mllm-jax python=3.12`
  - `conda activate mllm-jax && pip install -U pip`

### 2) Install TPU runtime deps

- `pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html`
- `pip install -U flax optax chex wandb pyyaml numpy`

Sanity check:

- `python -c "import jax; print(jax.__version__); print(jax.default_backend()); print(len(jax.devices()))"`

### 3) Sync repo via Git and checkout the target commit

- `git clone https://github.com/demon2036/MLLM-JAX.git /root/MLLM-JAX`
- `cd /root/MLLM-JAX && git fetch --all --prune && git checkout a258eab4367498fa3be49001329e89dd7fa1f69c`

### 4) (Optional) W&B auth (note)

In this run, the provided `WANDB_API_KEY` failed verification against `https://api.wandb.ai` (401), so the standard training was executed with `wandb.mode=disabled`.

If you intend to run W&B online:

- Put `WANDB_API_KEY` into `/root/.env` (do not commit) and verify:
  - `wandb login --relogin --verify <WANDB_API_KEY>`
- If using a dedicated / self-hosted W&B, also set `WANDB_BASE_URL` (or pass `--host` to `wandb login`).

### 5) TPU smoke (no W&B)

- `cd /root/MLLM-JAX`
- `python -u projects/nano_gpt/run.py --config projects/nano_gpt/configs/tinyshakespeare_char_cpu_smoke.yaml`

### 6) Standard run (no W&B, via nohup)

- Launch:
  - `bash scripts/tpu_vm_start_nanogpt_from_config_nohup.sh --config projects/nano_gpt/configs/tinyshakespeare_char_v6e_8_standard_no_wandb.yaml --env-name mllm-jax`
- Monitor:
  - `tail -n 50 -f logs/nohup_nanogpt_tinyshakespeare_char_v6e_8_standard_no_wandb_latest.log`
- Completion:
  - `cat logs/nohup_nanogpt_tinyshakespeare_char_v6e_8_standard_no_wandb_latest.exit` (expect `0`)

## Expected Result (verified)

- Training completes without traceback (exit code `0`)
- Final `eval/loss` at `step=5000` is printed in the log (observed: `1.6999`)
- Output directory contains `config_resolved.json`, `run_summary.json`, and checkpoints:
  - `runs/nano_gpt_tinyshakespeare_v6e_8_standard_no_wandb/checkpoints/checkpoint_4500`
  - `runs/nano_gpt_tinyshakespeare_v6e_8_standard_no_wandb/checkpoints/checkpoint_5000`

## Troubleshooting

- If `jax.default_backend()` is `cpu`: JAX TPU wheels/libtpu not installed or not detected.
- If you hit `global_batch_size must be divisible by num_devices`: set `train.global_batch_size` accordingly.
- If W&B returns 401: key invalid for the configured host; verify with `wandb login --verify` and/or set `WANDB_BASE_URL`.

## References

- `projects/nano_gpt/README.md`
- `projects/nano_gpt/configs/tinyshakespeare_char_v6e_8_standard.yaml`
- `projects/nano_gpt/configs/tinyshakespeare_char_v6e_8_standard_no_wandb.yaml`
- `scripts/tpu_vm_start_nanogpt_from_config_nohup.sh`
