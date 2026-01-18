# SOP: TPU v4-8 跑 bs=32 / `MAX_LENGTH_SAMPLE=1024` 20 steps，并统计后 10 steps 平均耗时（含 W&B）

- **Title**: SOP: TPU v4-8 timing run (bs=32, len=1024, k=1, micro_batch=4) + average `dt` over steps 10–19 (with W&B)
  **Prereqs**: `gcloud` 已登录；TPU VM 可 SSH；TPU VM 有 conda env `mllm-jax`；`/root/.env` 内含 `WANDB_API_KEY`
  **Environment (verified)**:
  - Project: `civil-rarity-482610-s5`
  - Zone: `us-central2-b`
  - TPU VM used (already alive): `mllm-jax-v4-8-260117090531` (`v4-8`, single host)
  - Conda env: `mllm-jax`; Python `3.12.12`
  - JAX: `0.8.2`, jaxlib `0.8.2`
  - JAX devices: `device_count=4` (megacore)
  - Repo ref on TPU (verified): `8a67d73` (branch: detached HEAD)

## Steps (commands actually used)

### 0) Confirm TPU JAX runtime

- `cd /home/john/works/MLLM-JAX-mllm-jax-sglang`
- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260117090531 --zone us-central2-b --project civil-rarity-482610-s5 --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; python - <<\"PY\"\nimport sys\nimport jax, jaxlib\nprint(\"python\", sys.version.split()[0])\nprint(\"jax\", jax.__version__, \"jaxlib\", jaxlib.__version__)\nprint(\"backend\", jax.default_backend())\nprint(\"process\", jax.process_index(), \"/\", jax.process_count())\nprint(\"device_count\", jax.device_count(), \"local\", len(jax.local_devices()))\nPY'`

### 1) Sync repo on TPU and checkout the target commit

- `cd /home/john/works/MLLM-JAX-mllm-jax-sglang`
- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260117090531 --zone us-central2-b --project civil-rarity-482610-s5 --command 'set -euo pipefail; cd /root/MLLM-JAX; git fetch --all --prune; git checkout 8a67d73; echo \"HEAD=$(git rev-parse --short HEAD)\"; python - <<\"PY\"\nimport yaml\nfrom pathlib import Path\ncfg=yaml.safe_load(Path(\"plugins/training/configs/grpo_gsm8k_bs32.yaml\").read_text())\nprint(\"rollout=\", cfg[\"rollout\"])\nprint(\"train=\", cfg[\"train\"])\nPY'`

### 2) Run 20 steps with W&B online + log file

- `cd /home/john/works/MLLM-JAX-mllm-jax-sglang`
- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260117090531 --zone us-central2-b --project civil-rarity-482610-s5 --env-file /root/.env --command 'set -euo pipefail; cd /root/MLLM-JAX; LOG_DIR=/root/MLLM-JAX/logs; mkdir -p \"$LOG_DIR\"; RUN_TS=$(date -u +%Y%m%d_%H%M%S); COMMIT=$(git rev-parse --short HEAD); LOG_FILE=\"$LOG_DIR/grpo_gsm8k_bs32_len1024_k1_mb4_steps20_${COMMIT}_${RUN_TS}.log\"; LATEST=\"$LOG_DIR/grpo_gsm8k_bs32_len1024_k1_mb4_steps20_latest.log\"; ln -sf \"$LOG_FILE\" \"$LATEST\"; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; export HF_HUB_ENABLE_HF_TRANSFER=1; export TOKENIZERS_PARALLELISM=false; export WANDB_MODE=online; export WANDB_PROJECT=mllm-jax-grpo-gsm8k; export WANDB_NAME=\"grpo_gsm8k_v4-8_bs32_len1024_k1_mb4_steps20_naive_${COMMIT}_${RUN_TS}\"; echo \"LOG_FILE=$LOG_FILE\"; echo \"WANDB_PROJECT=$WANDB_PROJECT\"; echo \"WANDB_NAME=$WANDB_NAME\"; set +e; python -u scripts/run_grpo_gsm8k_training.py --config plugins/training/configs/grpo_gsm8k_bs32.yaml 2>&1 | tee \"$LOG_FILE\"; status=${PIPESTATUS[0]}; set -e; echo \"exit_status=${status}\"; exit ${status}'`

### 3) Compute average `dt` for steps 10–19 from the log

- `cd /home/john/works/MLLM-JAX-mllm-jax-sglang`
- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v4-8-260117090531 --zone us-central2-b --project civil-rarity-482610-s5 --command 'set -euo pipefail; cd /root/MLLM-JAX; LOG_FILE=/root/MLLM-JAX/logs/grpo_gsm8k_bs32_len1024_k1_mb4_steps20_latest.log; ls -la \"$LOG_FILE\"; python - <<\"PY\"\nimport re\nfrom pathlib import Path\np = Path(\"/root/MLLM-JAX/logs/grpo_gsm8k_bs32_len1024_k1_mb4_steps20_latest.log\")\npat = re.compile(r\"^step=(\\d+)\\b.*\\bdt=([0-9.]+)s\\b\")\ndts = {}\nfor line in p.read_text(errors=\"ignore\").splitlines():\n    m = pat.search(line)\n    if m:\n        dts[int(m.group(1))] = float(m.group(2))\nsteps = list(range(10, 20))\nvals = [dts[s] for s in steps if s in dts]\nprint(\"dt_steps_10_19_count=\", len(vals))\nprint(\"dt_avg_10_19_s=\", sum(vals)/len(vals) if vals else None)\nprint(\"dt_steps_10_19=\", [(s, dts[s]) for s in steps if s in dts])\nPY'`

## Results (this run)

- Log file: `/root/MLLM-JAX/logs/grpo_gsm8k_bs32_len1024_k1_mb4_steps20_8a67d73_20260118_115940.log`
- W&B run: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/u8lze515`
- Avg dt over steps 10–19: `22.92s`
- dt steps 10–19:
  - `[(10, 23.86), (11, 21.97), (12, 23.74), (13, 20.48), (14, 21.05), (15, 23.72), (16, 23.78), (17, 24.64), (18, 22.77), (19, 23.19)]`

## Notes

- 本 run 使用 YAML：`plugins/training/configs/grpo_gsm8k_bs32.yaml`（`rollout.batch_size=32`, `rollout.num_pre_q=1`, `rollout.max_length_sample=1024`, `train.micro_batch_size=4`）。
- W&B 里可以看 breakdown：`time/train/rollout_s`, `time/train/update_s`, `time/train/step_s`（见 `plugins/training/runner/grpo_gsm8k.py`）。
- step=0 包含 JIT/编译开销；按 `steps 10–19` 统计更稳定。

