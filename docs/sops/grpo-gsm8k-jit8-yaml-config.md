# GRPO/GSM8K jit8 YAML Config SOPs

> Status: Deprecated/historical. The legacy `jit8_train` code has been moved under `deprecated/` and is not part of the active workflow.

- **Title**: SOP: Run `test_jit8.py` via YAML config
  **Prereqs**: Repo at `/home/john/github/MLLM-JAX`; PyYAML installed; (for training) JAX installed; (TPU) `gcloud` authenticated + a reachable TPU VM
  **Environment (verified)**:
  - Local: Ubuntu Linux; Python 3.12.2; pyyaml 6.0.3; JAX not installed (print-config + local schema test still work)
  - TPU VM: `v4-8` (`us-central2-b`), conda env `mllm-jax`; Python 3.12.12; JAX 0.8.2; libtpu 0.0.32; PyYAML 6.0.3
  - Git ref used on TPU: `bf199f8b6089add232c371c12a4fdde3eb152b71`
  **Steps**:
  - `cd /home/john/github/MLLM-JAX`
  - Ensure local scratch dir exists (ignored by git):
    - `mkdir -p workdir`
  - Print the merged config (sanity check):
    - `python test_jit8.py --print-config`
  - Override config fields from CLI (repeatable `--set key=value`):
    - `python test_jit8.py --print-config --set training_steps=1 --set wandb_enabled=false`
  - Enable runtime batch schema validation (useful while refactoring plugins):
    - `python test_jit8.py --print-config --set validate_schema=true`
  - Run the lightweight local regression test (no JAX required):
    - `python tests/test_jit8_schema_and_cli.py`
  - Ensure the TPU VM has Miniconda + the conda env (verified):
    - `scripts/bootstrap_miniconda_on_tpu_vm.sh --name mllm-jax-v4-8-260117090531 --zone us-central2-b`
  - Sync the repo to the TPU VM via Git (verified; see also `docs/sops/tpu-vm-repo-sync.md`):
    - `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260117090531 --zone us-central2-b --command 'set -euo pipefail; REPO_URL=https://github.com/demon2036/MLLM-JAX.git; REPO_DIR=/root/MLLM-JAX; if [ ! -d \"$REPO_DIR/.git\" ]; then rm -rf \"$REPO_DIR\"; git clone \"$REPO_URL\" \"$REPO_DIR\"; fi; cd \"$REPO_DIR\"; git fetch --all --prune; git checkout bf199f8b6089add232c371c12a4fdde3eb152b71; git status -sb; git rev-parse HEAD'`
  - Install TPU deps (verified):
    - `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260117090531 --zone us-central2-b --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; pip install -U \"jax[tpu]\" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; pip install -U torch --index-url https://download.pytorch.org/whl/cpu; cd /root/MLLM-JAX; pip install -U -r requirements-tpu.txt; python -c \"import jax; print(\\\"jax\\\", jax.__version__, \\\"backend\\\", jax.default_backend(), \\\"device_count\\\", jax.device_count()); print(jax.devices())\"'`
  - Run a 1-step TPU smoke training using YAML + CLI overrides (verified):
    - `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260117090531 --zone us-central2-b --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; export HF_HUB_ENABLE_HF_TRANSFER=1; export WANDB_MODE=disabled; export TOKENIZERS_PARALLELISM=false; python -u test_jit8.py --config plugins/jit8_train/configs/gsm8k_default.yaml --set model_path=Qwen/Qwen2.5-0.5B-Instruct --set training_steps=1 --set wandb_enabled=false --set validate_schema=true --set batch_size=1 --set grad_accum_steps=1 --set num_pre_q=4 --set ppo_steps=1 --set max_length_sample=128 --set max_length_extra=512 --set global_length=512'`
  **Expected Result**:
  - `--print-config` prints the final merged YAML.
  - TPU run prints at least one training line like `step=0 ...` and exits without Python tracebacks.
  **Troubleshooting**:
  - If `ModuleNotFoundError: jax`, install JAX (see `requirements-tpu.txt` header comments / `setup.sh`).
  - If padding fails with a negative width, increase `global_length` (GSM8K chat prompts measured ~275–399 tokens; `global_length=512` is safe with current `system_prompt`).
  - If `_form_global_array` throws a split error, ensure `batch_size * num_pre_q` is divisible by local device count (TPU v4-8 megacore: `4`).
  - If shapes mismatch, keep `max_length_extra` aligned with `prefill_length` (with `global_length=512`, `prefill_length=512`, so `max_length_total=max_length_sample+512`).
  **References**: `plugins/jit8_train/run.py`; `plugins/jit8_train/sampling.py`; `plugins/jit8_train/config.py`; `plugins/jit8_train/configs/gsm8k_default.yaml`; `docs/sops/github-push.md`; `docs/sops/tpu-vm-repo-sync.md`; (study) `git@github.com:demon2036/jax-imagenet-adv.git`
