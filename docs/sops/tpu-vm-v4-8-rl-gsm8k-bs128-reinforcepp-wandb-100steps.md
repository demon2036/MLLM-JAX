# SOP: TPU VM v4-8 run RL/GSM8K bs128 `reinforce++` (100 steps, W&B online)

- **Title**: SOP: TPU VM v4-8 run RL/GSM8K bs128 `reinforce++` (100 steps, W&B online)
  **Prereqs**: `gcloud` installed + authenticated; TPU quota/capacity; outbound internet from TPU VM (HF + datasets); repo pushed to GitHub; TPU has `/root/.env` containing `WANDB_API_KEY`
  **Environment (verified)**:
  - Project: `civil-rarity-482610-s5`
  - Zone: `us-central2-b`
  - TPU VM: `mllm-jax-v4-8-260122100610` (`v4-8`, single host)
  - Conda env: `sglang-jax`; Python `3.12.12`
  - JAX: `0.8.1`
  - JAX backend: `tpu`
  - JAX devices: `device_count=4` (megacore)
  - Repo ref on TPU (verified): `04f5097` (detached HEAD)
  - Base (no-shrink) config: `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml`
  - Algo-only config: `plugins/training/configs/rl_gsm8k_qwen25_3b_bs128_steps100_reinforcepp.yaml`
  - W&B run: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/84sik048`

## Steps (commands actually used)

### 0) Confirm TPU VM state

- `gcloud alpha compute tpus tpu-vm describe mllm-jax-v4-8-260122100610 --project civil-rarity-482610-s5 --zone us-central2-b --format='value(state,health,acceleratorType)'`

### 1) Confirm TPU runtime + W&B key (without printing it)

- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260122100610 --zone us-central2-b --project civil-rarity-482610-s5 --env-file /root/.env --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; python - <<\"PY\"\nimport os, sys\nimport jax\nprint(\"python\", sys.version.split()[0])\nprint(\"jax\", getattr(jax, \"__version__\", None))\nprint(\"jax_backend\", jax.default_backend())\nprint(\"device_count\", jax.device_count())\nprint(\"WANDB_API_KEY_set\", bool(os.environ.get(\"WANDB_API_KEY\")))\nPY'`

### 2) Confirm bs128 hyperparams + algo selection (print resolved config)

- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260122100610 --zone us-central2-b --project civil-rarity-482610-s5 --env-file /root/.env --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; cd /root/MLLM-JAX; python -u scripts/run_grpo_gsm8k_training.py --print-config --config plugins/training/configs/rl_gsm8k_qwen25_3b_bs128_steps100_reinforcepp.yaml | sed -n \"1,60p\"'`

### 3) Verify the bs128 algo YAML is “algo-only” (no shrink)

- `diff -u plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml plugins/training/configs/rl_gsm8k_qwen25_3b_bs128_steps100_reinforcepp.yaml | grep -E '^[+-][^+-]'`
  - Expected output:
    - `-  name: grpo`
    - `+  name: reinforce++`

### 4) Locate the run log + confirm 100 steps

- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260122100610 --zone us-central2-b --project civil-rarity-482610-s5 --command 'set -euo pipefail; cd /root/MLLM-JAX; LOG=$(readlink -f logs/rl_gsm8k_qwen25_3b_bs128_steps100_reinforcepp_latest.log); echo \"LOG=$LOG\"; grep -n \"^step=\" \"$LOG\" | tail -n 5'`

### 5) Sanity-check: no tracebacks + process ended

- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260122100610 --zone us-central2-b --project civil-rarity-482610-s5 --command 'set -euo pipefail; cd /root/MLLM-JAX; LOG=$(readlink -f logs/rl_gsm8k_qwen25_3b_bs128_steps100_reinforcepp_latest.log); echo traceback_count=$(grep -n \"Traceback\" \"$LOG\" | wc -l); echo pgrep_matches:; pgrep -af \"[r]un_grpo_gsm8k_training.py\" || true'`

### 6) Confirm W&B server-side run state is `finished`

- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260122100610 --zone us-central2-b --project civil-rarity-482610-s5 --env-file /root/.env --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; python - <<\"PY\"\nimport wandb\napi = wandb.Api(timeout=30)\nrun = api.run(\"johntitordemon2036/mllm-jax-grpo-gsm8k/84sik048\")\nprint(\"wandb_state\", run.state)\nPY'`

## Expected Result

- The resolved config shows `rollout.batch_size=16`, `rollout.n=8` (`128` seq/step) and `steps=100` (bs128 no-shrink).
- The log contains `step=99 ...` (100 steps) and has `traceback_count=0`.
- `pgrep -af "[r]un_grpo_gsm8k_training.py"` prints nothing after completion.
- W&B API reports `wandb_state finished`.

## Results (this run)

- Commit on TPU: `04f5097`
- Log symlink: `/root/MLLM-JAX/logs/rl_gsm8k_qwen25_3b_bs128_steps100_reinforcepp_latest.log`
- Concrete log: `/root/MLLM-JAX/logs/rl_gsm8k_qwen25_3b_bs128_steps100_reinforcepp_04f5097_20260124_013914.log`
- W&B run: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/84sik048`

## Troubleshooting

- `Unable to initialize backend 'tpu' ... already in use`:
  - Ensure no other training process is running and remove lockfile: `rm -f /tmp/libtpu_lockfile`.
- Warnings about deprecated env overrides (e.g. `WANDB_MODE`):
  - This repo ignores env-var hyperparameter overrides; keep hyperparams in YAML configs.
