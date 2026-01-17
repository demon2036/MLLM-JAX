# GRPO/GSM8K jit8 YAML Config SOPs

- **Title**: SOP: Run `test_jit8.py` via YAML config
  **Prereqs**: Repo at `/home/john/github/MLLM-JAX`; PyYAML installed; (for training) JAX installed; (TPU) `gcloud` authenticated + a reachable TPU VM
  **Environment (verified)**:
  - Local: Ubuntu 6.14.0-37-generic; Python 3.13.9; pyyaml 6.0.3; transformers 4.57.3; JAX not installed (print-config still works)
  - TPU VM: `v4-8` (`us-central2-b`), conda env `mllm-jax`; Python 3.12.12; JAX 0.8.2; libtpu 0.0.32; PyYAML 6.0.3
  - Git ref used on TPU: `a45be03d1d42cb39e56f9effd7d22e9d455dc474`
  **Steps**:
  - `cd /home/john/github/MLLM-JAX`
  - Ensure local scratch dir exists (ignored by git):
    - `mkdir -p workdir`
  - Print the merged config (sanity check):
    - `python test_jit8.py --print-config`
  - Override config fields from CLI (repeatable `--set key=value`):
    - `python test_jit8.py --print-config --set training_steps=1 --set wandb_enabled=false`
  - Sync the repo to the TPU VM via Git (verified; see also `docs/sops/tpu-vm-repo-sync.md`):
    - `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260117090531 --zone us-central2-b --project civil-rarity-482610-s5 --command 'set -euo pipefail; REPO_URL=https://github.com/demon2036/MLLM-JAX.git; REPO_DIR=/root/MLLM-JAX; if [ ! -d \"$REPO_DIR/.git\" ]; then git clone \"$REPO_URL\" \"$REPO_DIR\"; fi; cd \"$REPO_DIR\"; git fetch --all --prune; git checkout a45be03d1d42cb39e56f9effd7d22e9d455dc474; git status -sb'`
  - Run a 1-step TPU smoke training using YAML + CLI overrides (verified):
    - `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260117090531 --zone us-central2-b --project civil-rarity-482610-s5 --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; export HF_HUB_ENABLE_HF_TRANSFER=1; export WANDB_MODE=disabled; export TOKENIZERS_PARALLELISM=false; python -u test_jit8.py --config plugins/jit8_train/configs/gsm8k_default.yaml --set model_path=Qwen/Qwen2.5-0.5B-Instruct --set training_steps=1 --set wandb_enabled=false --set batch_size=1 --set grad_accum_steps=1 --set num_pre_q=4 --set ppo_steps=1 --set max_length_sample=128 --set max_length_extra=512 --set global_length=512'`
  **Expected Result**:
  - `--print-config` prints the final merged YAML.
  - TPU run prints at least one training line like `step=0 ...` and exits without Python tracebacks.
  **Troubleshooting**:
  - If `ModuleNotFoundError: jax`, install JAX (see `requirements-tpu.txt` header comments / `setup.sh`).
  - If padding fails with a negative width, increase `global_length` (GSM8K chat prompts measured ~275–399 tokens; `global_length=512` is safe with current `system_prompt`).
  - If `_form_global_array` throws a split error, ensure `batch_size * num_pre_q` is divisible by local device count (TPU v4-8 megacore: `4`).
  - If shapes mismatch, keep `max_length_extra` aligned with `prefill_length` (with `global_length=512`, `prefill_length=512`, so `max_length_total=max_length_sample+512`).
  **References**: `plugins/jit8_train/run.py`; `plugins/jit8_train/sampling.py`; `plugins/jit8_train/config.py`; `plugins/jit8_train/configs/gsm8k_default.yaml`; `docs/sops/github-push.md`; `docs/sops/tpu-vm-repo-sync.md`; (study) `git@github.com:demon2036/jax-imagenet-adv.git`
