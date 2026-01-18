# SOP: TPU v4-8 跑 `MAX_LENGTH_SAMPLE=1024` 20 steps，并统计后 10 steps 平均耗时（含 W&B）

- **Title**: SOP: TPU v4-8 timing run (`MAX_LENGTH_SAMPLE=1024`, 20 steps) + average dt over steps 10–19 (with W&B)
  **Prereqs**: `gcloud` 已登录；TPU VM 可 SSH；TPU VM 有 conda env `mllm-jax`；`/root/.env` 内含 `WANDB_API_KEY`
  **Environment (verified)**:
  - Project: `civil-rarity-482610-s5`
  - Zone: `us-central2-b`
  - TPU VM used (already alive): `mllm-jax-v4-8-260117090531` (`v4-8`, single host)
  - Conda env: `mllm-jax`; Python `3.12.12`
  - JAX: `0.8.2`, jaxlib `0.8.2`
  - Repo ref on TPU (verified): `2f2d836` (branch: detached HEAD)

## Steps (commands actually used)

### 0) Find a READY TPU VM (reuse existing)

- `gcloud alpha compute tpus tpu-vm list --project civil-rarity-482610-s5 --zone us-central2-b --format='table(name,acceleratorType,state)'`

### 1) Confirm W&B key is present on TPU (without printing it)

- `cd /home/john/works/MLLM-JAX-mllm-jax-sglang`
- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260117090531 --zone us-central2-b --project civil-rarity-482610-s5 --env-file /root/.env --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; python - <<\"PY\"\nimport os\nprint(\"WANDB_API_KEY_set=\", bool(os.environ.get(\"WANDB_API_KEY\")))\nPY'`

### 2) Sync repo on TPU and checkout the target commit

- `cd /home/john/works/MLLM-JAX-mllm-jax-sglang`
- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260117090531 --zone us-central2-b --project civil-rarity-482610-s5 --command 'set -euo pipefail; cd /root/MLLM-JAX; git fetch --all --prune; git checkout 2f2d836; git status -sb; echo \"HEAD=$(git rev-parse --short HEAD)\"'`

### 3) Run 20 steps with `MAX_LENGTH_SAMPLE=1024` (W&B online + log file)

- `cd /home/john/works/MLLM-JAX-mllm-jax-sglang`
- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260117090531 --zone us-central2-b --project civil-rarity-482610-s5 --env-file /root/.env --command 'set -euo pipefail; cd /root/MLLM-JAX; LOG_DIR=/root/MLLM-JAX/logs; mkdir -p \"$LOG_DIR\"; RUN_TS=$(date -u +%Y%m%d_%H%M%S); COMMIT=$(git rev-parse --short HEAD); LOG_FILE=\"$LOG_DIR/grpo_gsm8k_len1024_steps20_${COMMIT}_${RUN_TS}.log\"; LATEST=\"$LOG_DIR/grpo_gsm8k_len1024_steps20_latest.log\"; ln -sf \"$LOG_FILE\" \"$LATEST\"; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; export HF_HUB_ENABLE_HF_TRANSFER=1; export TOKENIZERS_PARALLELISM=false; export WANDB_MODE=online; export WANDB_PROJECT=mllm-jax-grpo-gsm8k; export WANDB_NAME=\"grpo_gsm8k_v4-8_len1024_steps20_naive_${COMMIT}_${RUN_TS}\"; export MODEL_PATH=\"Qwen/Qwen2.5-7B-Instruct\"; export STEPS=20; export ROLLOUT_BACKEND=naive; export BATCH_SIZE=1; export NUM_PRE_Q=4; export GLOBAL_LENGTH=512; export MAX_LENGTH_SAMPLE=1024; export PPO_EPOCHS=1; export GRAD_ACCUM_STEPS=1; export BETA=0.0; echo \"LOG_FILE=$LOG_FILE\"; echo \"WANDB_PROJECT=$WANDB_PROJECT\"; echo \"WANDB_NAME=$WANDB_NAME\"; set +e; python -u scripts/run_grpo_gsm8k_training.py 2>&1 | tee \"$LOG_FILE\"; status=${PIPESTATUS[0]}; set -e; echo \"exit_status=${status}\"; exit ${status}'`

### 4) Compute average `dt` for steps 10–19 from the log

- `cd /home/john/works/MLLM-JAX-mllm-jax-sglang`
- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260117090531 --zone us-central2-b --project civil-rarity-482610-s5 --command 'set -euo pipefail; cd /root/MLLM-JAX; LOG_FILE=/root/MLLM-JAX/logs/grpo_gsm8k_len1024_steps20_latest.log; ls -la \"$LOG_FILE\"; python - <<PY\nimport re\nfrom pathlib import Path\np = Path(\"/root/MLLM-JAX/logs/grpo_gsm8k_len1024_steps20_latest.log\")\npat = re.compile(r\"^step=(\\d+)\\b.*\\bdt=([0-9.]+)s\\b\")\ndts = {}\nfor line in p.read_text(errors=\"ignore\").splitlines():\n    m = pat.search(line)\n    if m:\n        dts[int(m.group(1))] = float(m.group(2))\nsteps = list(range(10, 20))\nvals = [dts[s] for s in steps if s in dts]\nprint(\"dt_steps_10_19_count=\", len(vals))\nprint(\"dt_avg_10_19_s=\", sum(vals)/len(vals) if vals else None)\nprint(\"dt_steps_10_19=\", [(s, dts[s]) for s in steps if s in dts])\nPY'`

## Results (this run)

- Log file: `/root/MLLM-JAX/logs/grpo_gsm8k_len1024_steps20_2f2d836_20260118_104721.log`
- W&B run: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/gzfha44a`
- Avg dt over steps 10–19: `5.476s`
- dt steps 10–19:
  - `[(10, 4.49), (11, 7.01), (12, 3.45), (13, 4.68), (14, 6.66), (15, 5.85), (16, 7.4), (17, 5.0), (18, 6.08), (19, 4.14)]`

## Notes

- 想在 W&B 里看 breakdown：runner 会打点 `time/train/rollout_s`, `time/train/update_s`, `time/train/step_s` 等（见 `plugins/training/runner/grpo_gsm8k.py`）。
- 前几步包含 JIT/编译开销，按 `steps 10–19` 统计更稳定。

