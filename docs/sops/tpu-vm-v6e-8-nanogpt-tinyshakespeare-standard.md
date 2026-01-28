# SOP: TPU v6e-8 nanoGPT Tiny Shakespeare (char) standard training (W&B online)

- **Title**: SOP: Run nanoGPT Tiny Shakespeare (char-level) standard training on TPU v6e-8
  **Prereqs**:
  - TPU VM reachable (v6e-8)
  - Repo synced to TPU via Git (no code SCP)
  - Conda env exists with JAX TPU wheel + deps installed
  - `/root/.env` contains `WANDB_API_KEY` and W&B egress works
  **Environment (verified)**:
  - (fill) TPU: v6e-8, zone `europe-west4-a`
  - (fill) OS: Ubuntu 24.04
  - (fill) Python:
  - (fill) jax/jaxlib:
  - (fill) flax/optax:
  **Steps**:
  - On the TPU VM (from repo root):
    - (fill) Activate conda env:
      - `source /root/miniconda3/etc/profile.d/conda.sh && conda activate mllm-jax`
    - (fill) Sanity check TPU backend:
      - `python -c "import jax; print(jax.default_backend()); print(len(jax.devices()))"`
    - (fill) Run standard training config (foreground):
      - `bash projects/nano_gpt/scripts/run_train.sh --config projects/nano_gpt/configs/tinyshakespeare_char_v6e_8_standard.yaml`
    - (optional) Run in background via nohup:
      - `bash scripts/tpu_vm_start_nanogpt_from_config_nohup.sh --config projects/nano_gpt/configs/tinyshakespeare_char_v6e_8_standard.yaml`
      - `tail -n 200 -f logs/nohup_nanogpt_tinyshakespeare_char_v6e_8_standard_latest.log`
  **Expected Result**:
  - Training completes without traceback (exit code `0`)
  - W&B run created in `nano-gpt-jax` project with train/eval metrics
  - Output directory contains `config_resolved.json`, `run_summary.json`, and checkpoints
  **Troubleshooting**:
  - If W&B fails auth: verify `/root/.env` has `WANDB_API_KEY` and `wandb.mode=online` in the YAML.
  - If `jax.default_backend()` is `cpu`: JAX TPU wheels/libtpu not installed or not detected.
  - If you hit `global_batch_size must be divisible by num_devices`: set `train.global_batch_size` accordingly.
  **References**:
  - `projects/nano_gpt/README.md`
  - `projects/nano_gpt/configs/tinyshakespeare_char_v6e_8_standard.yaml`
  - `scripts/tpu_vm_start_nanogpt_from_config_nohup.sh`

