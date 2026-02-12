# SOP: GRPO adv==0 `<think>` window penalty (reward-gap scaled, -0.1)

- **Title**: SOP: Enable balanced adv==0 `<think>` window penalty for GRPO/GSM8K
  **Goal**: Keep standard GRPO updates when `adv!=0`, but for `adv==0` sequences apply a small negative advantage on a `<think>` token window to encourage exploration — while skipping full-score samples.
  **Definition**:
  - Apply only when `advantages==0` (sequence-level).
  - Window mask: find first `<think>` in completion tokens; start after tag; take next `K=window_tokens` tokens. If missing, fallback to first K completion tokens (`no_think_policy=first_tokens`).
  - Penalty advantage: `adv_zero_token = penalty * scale`, where:
    - `penalty = -0.1`
    - `scale = clip((max_total_reward - reward) / max_total_reward, 0, 1)`
    - `max_total_reward = sum(reward_weights)`; full-score ⇒ `scale=0` ⇒ no update.
  - Normalization: `per_sequence`.
  **Prereqs**: repo pushed to GitHub; TPU VM reachable; internet egress (HF + datasets); W&B API key available.

## Files / Config

- Window mask helper: `plugins/training/rl/adv_zero_think_penalty.py`
- Config schema: `plugins/training/rl/algorithms/config.py` (`algo.update.kwargs.adv_zero_think_penalty`)
- Loss injection: `MLLM_JAX/train_modules/__init__.py` (`TrainGRPOModule`)
- Runner wiring + metrics: `projects/gsm8k_grpo/jax/train.py`
- TPU YAML (validated): `projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_adv0_think_penalty_gapscale_pen0p1_batch16_roll8_v6e8.yaml`

## Steps (commands actually used)

### 1) Local: tests, push branch

- `git switch -c test-token`
- `pytest -q`  # exit code 0 (JAX-dependent tests may skip)
- `git commit -m "feat: balance adv0 think penalty with reward-gap scaling"`
- `git push -u origin test-token`

### 2) Create TPU VM (v6e-8, spot)

- `scripts/create_tpu_vm.sh --type v6e-8 --zone us-east1-d --name mllm-jax-grpo-adv0think-balance-v6e-8-useast1d-spot-260212062815 --project civil-rarity-482610-s5 --spot`

### 3) Sync `.env` (W&B online)

Create a local `.env` (do not commit) containing:
- `WANDB_API_KEY=...`

Sync to TPU (worker=all):
- `scripts/sync_env_to_tpu_vm.sh --name mllm-jax-grpo-adv0think-balance-v6e-8-useast1d-spot-260212062815 --zone us-east1-d --project civil-rarity-482610-s5 --src .env`

### 4) Git-sync repo on TPU + checkout commit

- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-grpo-adv0think-balance-v6e-8-useast1d-spot-260212062815 --zone us-east1-d --project civil-rarity-482610-s5 --command 'set -euo pipefail; REPO_URL=https://github.com/demon2036/MLLM-JAX.git; REPO_DIR=/root/MLLM-JAX; if [ ! -d \"$REPO_DIR/.git\" ]; then rm -rf \"$REPO_DIR\"; git clone \"$REPO_URL\" \"$REPO_DIR\"; fi; cd \"$REPO_DIR\"; git fetch --all --prune; git checkout test-token; git pull --ff-only; echo \"HEAD=$(git rev-parse --short HEAD)\"'`

### 5) Run (100 steps; full test eval every 50 steps)

- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-grpo-adv0think-balance-v6e-8-useast1d-spot-260212062815 --zone us-east1-d --project civil-rarity-482610-s5 --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; bash scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh --env-name mllm-jax --config projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_adv0_think_penalty_gapscale_pen0p1_batch16_roll8_v6e8.yaml'`

Monitor:
- `scripts/ssh_tpu_vm_root.sh --name ... --zone ... --command 'set -euo pipefail; cd /root/MLLM-JAX; grep -n \"^eval_full \" logs/nohup_grpo_gsm8k_qwen25_3b_adv0_think_penalty_gapscale_pen0p1_batch16_roll8_v6e8_latest.log | tail -n 5; cat logs/nohup_grpo_gsm8k_qwen25_3b_adv0_think_penalty_gapscale_pen0p1_batch16_roll8_v6e8_latest.exit'`

Observed:
- `eval_full step=49 ... questions=1319 ... accuracy=0.7885 ...`
- `eval_full step=99 ... questions=1319 ... accuracy=0.7824 ...`
- Exit code: `0`
- W&B run: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k-adv0-think-penalty-balance/runs/r2iawajc`

### 6) Cleanup TPU

- `scripts/delete_tpu_vm.sh --name mllm-jax-grpo-adv0think-balance-v6e-8-useast1d-spot-260212062815 --zone us-east1-d --project civil-rarity-482610-s5`

## Expected Result

- Training reaches `step=99` with no tracebacks and exit code `0`.
- Full eval sweep runs at `step=49` and `step=99` (for `eval_full_every_steps: 50`).
- W&B is `online` and run state is `finished`.

## Troubleshooting

- **W&B disabled**: ensure `WANDB_API_KEY` is present on TPU (`/root/.env`) and `wandb_mode: online` in YAML.
- **Spot preemption**: re-create TPU VM and re-run the YAML; prefer checkpointing if you extend steps.
- **Throughput slower than expected**: confirm you’re on `v6e-8` and not `v4-8`; check `rollout.max_length_sample` (decode dominates).

