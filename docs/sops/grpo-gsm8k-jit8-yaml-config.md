# GRPO/GSM8K jit8 YAML Config SOPs

- **Title**: SOP: Run `test_jit8.py` via YAML config
  **Prereqs**: Repo at `/home/john/github/MLLM-JAX`; JAX installed (required for training); PyYAML installed; (optional) Weights & Biases account for logging
  **Environment (verified)**: Ubuntu 6.14.0-37-generic; Python 3.13.9; pyyaml 6.0.3; transformers 4.57.3; jax: import failed
  **Steps**:
  - `cd /home/john/github/MLLM-JAX`
  - Ensure local scratch dir exists (ignored by git):
    - `mkdir -p workdir`
  - Print the merged config (sanity check):
    - `python test_jit8.py --print-config`
  - Override config fields from CLI (repeatable `--set key=value`):
    - `python test_jit8.py --print-config --set training_steps=1 --set wandb_enabled=false`
  - (Placeholder; not run in this session) Start training:
    - `python test_jit8.py --config plugins/jit8_train/configs/gsm8k_default.yaml`
  **Expected Result**: `--print-config` prints the final merged YAML; training starts when `import jax` succeeds.
  **Troubleshooting**: If `ModuleNotFoundError: jax`, install JAX (see `requirements-tpu.txt` header comments / `setup.sh`).
  **References**: `plugins/jit8_train/run.py`; `plugins/jit8_train/config.py`; `plugins/jit8_train/configs/gsm8k_default.yaml`; (study) `git@github.com:demon2036/jax-imagenet-adv.git`

