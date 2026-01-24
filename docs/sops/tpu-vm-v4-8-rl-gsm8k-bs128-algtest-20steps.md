# SOP: TPU VM v4-8 run RL/GSM8K bs128 algtest (20 steps, W&B online)

- **Title**: SOP: TPU VM v4-8 run RL/GSM8K bs128 algtest (20 steps, W&B online)
  **Prereqs**: `gcloud` installed + authenticated; TPU quota/capacity; outbound internet from TPU VM (HF + datasets); repo pushed to GitHub; TPU has `/root/.env` containing `WANDB_API_KEY`
  **Environment (verified)**:
  - Project: `civil-rarity-482610-s5`
  - Zone: `us-central2-b`
  - TPU VM: `mllm-jax-v4-8-260122100610` (`v4-8`, single host)
  - Conda env: `sglang-jax`; Python `3.12.12`
  - JAX: `0.8.1`
  - JAX backend: `tpu`
  - JAX devices: `device_count=4` (megacore)
  - Branch/commit: `algorithm` @ `66c6c92`
  - W&B project: `algorithm_test`

## Steps (commands actually used)

### 0) Confirm TPU VM state

```bash
gcloud alpha compute tpus tpu-vm describe mllm-jax-v4-8-260122100610 \
  --project civil-rarity-482610-s5 --zone us-central2-b \
  --format='value(state,health,acceleratorType)'
```

### 1) Sync repo on TPU via Git (no SCP)

```bash
scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260122100610 --zone us-central2-b \
  --project civil-rarity-482610-s5 \
  --command 'set -euo pipefail; REPO_URL=https://github.com/demon2036/MLLM-JAX.git; REPO_DIR=/root/MLLM-JAX; if [ ! -d "$REPO_DIR/.git" ]; then git clone "$REPO_URL" "$REPO_DIR"; fi; cd "$REPO_DIR"; git fetch --all --prune; git checkout algorithm; git status -sb; git rev-parse --short HEAD'
```

### 2) Verify TPU runtime + W&B key (without printing it)

```bash
scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260122100610 --zone us-central2-b \
  --project civil-rarity-482610-s5 --env-file /root/.env \
  --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; python - <<"PY"
import os, sys
import jax
print("python", sys.version.split()[0])
print("jax", getattr(jax, "__version__", None))
print("jax_backend", jax.default_backend())
print("device_count", jax.device_count())
print("WANDB_API_KEY_set", bool(os.environ.get("WANDB_API_KEY")))
PY'
```

### 3) Launch PPO (20 steps, algtest)

```bash
scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260122100610 --zone us-central2-b \
  --project civil-rarity-482610-s5 --env-file /root/.env \
  --command 'set -euo pipefail; cd /root/MLLM-JAX; bash scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh --config plugins/training/configs/rl_gsm8k_qwen25_3b_bs128_steps20_ppo_algtest.yaml --env-name sglang-jax'
```

### 4) Wait for PPO completion + verify logs

```bash
scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260122100610 --zone us-central2-b \
  --project civil-rarity-482610-s5 \
  --command 'set -euo pipefail; cd /root/MLLM-JAX; EXIT=logs/nohup_rl_gsm8k_qwen25_3b_bs128_steps20_ppo_algtest_20260124_074428.exit; LOG=logs/nohup_rl_gsm8k_qwen25_3b_bs128_steps20_ppo_algtest_20260124_074428.log; while [ ! -f "$EXIT" ]; do sleep 30; done; echo "EXIT=$(cat $EXIT)"; echo "LOG=$LOG"; echo "steps:"; grep -n "^step=" "$LOG" | tail -n 5; echo "traceback_count=$(grep -n "Traceback" "$LOG" | wc -l)"; echo "wandb_line:"; grep -n "wandb: 🚀 View run" "$LOG" | head -n 1'
```

### 5) Launch REINFORCE (20 steps, algtest)

```bash
scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260122100610 --zone us-central2-b \
  --project civil-rarity-482610-s5 --env-file /root/.env \
  --command 'set -euo pipefail; cd /root/MLLM-JAX; bash scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh --config plugins/training/configs/rl_gsm8k_qwen25_3b_bs128_steps20_reinforce_algtest.yaml --env-name sglang-jax'
```

### 6) Wait for REINFORCE completion + verify logs

```bash
scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260122100610 --zone us-central2-b \
  --project civil-rarity-482610-s5 \
  --command 'set -euo pipefail; cd /root/MLLM-JAX; EXIT=logs/nohup_rl_gsm8k_qwen25_3b_bs128_steps20_reinforce_algtest_20260124_080640.exit; LOG=logs/nohup_rl_gsm8k_qwen25_3b_bs128_steps20_reinforce_algtest_20260124_080640.log; while [ ! -f "$EXIT" ]; do sleep 30; done; echo "EXIT=$(cat $EXIT)"; echo "LOG=$LOG"; echo "steps:"; grep -n "^step=" "$LOG" | tail -n 5; echo "traceback_count=$(grep -n "Traceback" "$LOG" | wc -l)"; echo "wandb_line:"; grep -n "wandb: 🚀 View run" "$LOG" | head -n 1'
```

### 7) Launch RLOO (20 steps, algtest)

```bash
scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260122100610 --zone us-central2-b \
  --project civil-rarity-482610-s5 --env-file /root/.env \
  --command 'set -euo pipefail; cd /root/MLLM-JAX; bash scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh --config plugins/training/configs/rl_gsm8k_qwen25_3b_bs128_steps20_rloo_algtest.yaml --env-name sglang-jax'
```

### 8) Wait for RLOO completion + verify logs

```bash
scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260122100610 --zone us-central2-b \
  --project civil-rarity-482610-s5 \
  --command 'set -euo pipefail; cd /root/MLLM-JAX; EXIT=logs/nohup_rl_gsm8k_qwen25_3b_bs128_steps20_rloo_algtest_20260124_083112.exit; LOG=logs/nohup_rl_gsm8k_qwen25_3b_bs128_steps20_rloo_algtest_20260124_083112.log; while [ ! -f "$EXIT" ]; do sleep 30; done; echo "EXIT=$(cat $EXIT)"; echo "LOG=$LOG"; echo "steps:"; grep -n "^step=" "$LOG" | tail -n 5; echo "traceback_count=$(grep -n "Traceback" "$LOG" | wc -l)"; echo "wandb_line:"; grep -n "wandb: 🚀 View run" "$LOG" | head -n 1'
```

### 9) Launch DAPO (20 steps, algtest)

```bash
scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260122100610 --zone us-central2-b \
  --project civil-rarity-482610-s5 --env-file /root/.env \
  --command 'set -euo pipefail; cd /root/MLLM-JAX; bash scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh --config plugins/training/configs/rl_gsm8k_qwen25_3b_bs128_steps20_dapo_algtest.yaml --env-name sglang-jax'
```

### 10) Wait for DAPO completion + verify logs

```bash
scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260122100610 --zone us-central2-b \
  --project civil-rarity-482610-s5 \
  --command 'set -euo pipefail; cd /root/MLLM-JAX; EXIT=logs/nohup_rl_gsm8k_qwen25_3b_bs128_steps20_dapo_algtest_20260124_085452.exit; LOG=logs/nohup_rl_gsm8k_qwen25_3b_bs128_steps20_dapo_algtest_20260124_085452.log; while [ ! -f "$EXIT" ]; do sleep 30; done; echo "EXIT=$(cat $EXIT)"; echo "LOG=$LOG"; echo "steps:"; grep -n "^step=" "$LOG" | tail -n 5; echo "traceback_count=$(grep -n "Traceback" "$LOG" | wc -l)"; echo "wandb_line:"; grep -n "wandb: 🚀 View run" "$LOG" | head -n 1'
```

### 11) Confirm W&B run states (API)

```bash
scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260122100610 --zone us-central2-b \
  --project civil-rarity-482610-s5 --env-file /root/.env \
  --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; python - <<"PY"
import wandb
api = wandb.Api(timeout=30)
runs = {
    "ppo": "m035cwxl",
    "reinforce": "jwarpwed",
    "rloo": "31l8fq53",
    "dapo": "7y09so5y",
}
for name, run_id in runs.items():
    run = api.run(f"johntitordemon2036/algorithm_test/{run_id}")
    print(name, run.state, run.name)
PY'
```

## Expected Result

- Each algtest run exits `0` and logs `step=19` with `algo=<name>`.
- No `Traceback` appears in any log.
- W&B project `algorithm_test` shows all runs in `finished` state.

## Results (this run)

- PPO: `https://wandb.ai/johntitordemon2036/algorithm_test/runs/m035cwxl`
- REINFORCE: `https://wandb.ai/johntitordemon2036/algorithm_test/runs/jwarpwed`
- RLOO: `https://wandb.ai/johntitordemon2036/algorithm_test/runs/31l8fq53`
- DAPO: `https://wandb.ai/johntitordemon2036/algorithm_test/runs/7y09so5y`

## Troubleshooting

- TPU busy or stuck: `pgrep -af "[r]un_grpo_gsm8k_training.py"` and remove `/tmp/libtpu_lockfile`.
- W&B not syncing: verify `/root/.env` has `WANDB_API_KEY` and rerun the W&B API check.

## References

- `scripts/ssh_tpu_vm_root.sh`
- `scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh`
- `plugins/training/configs/rl_gsm8k_qwen25_3b_bs128_steps20_ppo_algtest.yaml`
- `plugins/training/configs/rl_gsm8k_qwen25_3b_bs128_steps20_reinforce_algtest.yaml`
- `plugins/training/configs/rl_gsm8k_qwen25_3b_bs128_steps20_rloo_algtest.yaml`
- `plugins/training/configs/rl_gsm8k_qwen25_3b_bs128_steps20_dapo_algtest.yaml`
