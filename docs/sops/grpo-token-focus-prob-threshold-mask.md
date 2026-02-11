# SOP: GRPO token-focus mask (prob < threshold, first K tokens)

- **Title**: SOP: Enable GRPO token-focus mask (absolute prob threshold)
  **Goal**: Update only early low-probability completion tokens during the policy-gradient/PPO-style update.
  **Definition**:
  - Eligible token: `prob(token) < prob_threshold` (implemented as `logp < log(prob_threshold)`).
  - Scan left-to-right and keep only the first `K=max_tokens_per_sequence` eligible tokens per sequence.
  - Apply the mask only to **policy loss**; normalize by **selected token count** (avoid extra LR shrink).
  **Prereqs**: repo pushed to GitHub; TPU VM reachable; internet egress (HF + datasets).

## Files / Config

- Token-focus helper: `plugins/training/rl/token_focus.py`
- GRPO loss integration: `MLLM_JAX/train_modules/__init__.py` (`TrainGRPOModule`)
- PPO loss integration (value head path): `plugins/training/rl/ppo/module.py`
- Config schema: `plugins/training/rl/algorithms/config.py` (`algo.update.kwargs.token_focus`)
- GSM8K token-focus YAML: `projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_token_focus_v6e8.yaml`

## Steps (commands actually used)

### 1) Local: create a working branch, run tests, push

- `git switch -c test-token`
- `pytest -q`  # exit code 0 (JAX-dependent tests may skip)
- `git commit -m "feat: token-focus mask for GRPO policy loss"`
- `git push -u origin test-token`

### 2) Create TPU VM

Notes:
- v6e quota was `0` in `us-central2-b`, so we used `v4-8`.

- `scripts/create_tpu_vm.sh --type v4-8 --zone us-central2-b --name mllm-jax-grpo-tokenfocus-v4-8-260211083215`

### 3) Bootstrap conda env on TPU

- `scripts/bootstrap_miniconda_on_tpu_vm.sh --name mllm-jax-grpo-tokenfocus-v4-8-260211083215 --zone us-central2-b --project civil-rarity-482610-s5`

### 4) Git-sync repo on TPU and checkout the commit

- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-grpo-tokenfocus-v4-8-260211083215 --zone us-central2-b --project civil-rarity-482610-s5 --command 'set -euo pipefail; REPO_URL=https://github.com/demon2036/MLLM-JAX.git; REPO_DIR=/root/MLLM-JAX; if [ ! -d \"$REPO_DIR/.git\" ]; then rm -rf \"$REPO_DIR\"; git clone \"$REPO_URL\" \"$REPO_DIR\"; fi; cd \"$REPO_DIR\"; git fetch --all --prune; git checkout 5c66e79; git status -sb; echo \"HEAD=$(git rev-parse --short HEAD)\"'`

### 5) Install TPU deps (JAX + requirements)

- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-grpo-tokenfocus-v4-8-260211083215 --zone us-central2-b --project civil-rarity-482610-s5 --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; python -V; python -m pip install -U pip; python -m pip install -U \"jax[tpu]\" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; python -m pip install -U torch --index-url https://download.pytorch.org/whl/cpu; cd /root/MLLM-JAX; python -m pip install -U -r requirements-tpu.txt; python - <<\"PY\"\nimport sys\nimport jax, jaxlib\nprint(\"python\", sys.version.split()[0])\nprint(\"jax\", jax.__version__, \"jaxlib\", jaxlib.__version__)\nprint(\"backend\", jax.default_backend())\nprint(\"process\", jax.process_index(), \"/\", jax.process_count())\nprint(\"device_count\", jax.device_count(), \"local\", len(jax.local_devices()))\nPY'`

### 6) Run GRPO/GSM8K token-focus config (nohup)

- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-grpo-tokenfocus-v4-8-260211083215 --zone us-central2-b --project civil-rarity-482610-s5 --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; bash scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh --config projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_token_focus_v6e8.yaml'`

Monitor:
- `scripts/ssh_tpu_vm_root.sh --name ... --zone ... --command 'set -euo pipefail; cd /root/MLLM-JAX; tail -n 80 logs/nohup_grpo_gsm8k_qwen25_3b_token_focus_v6e8_latest.log'`

Observed:
- Config print includes `algo.update.kwargs.token_focus`.
- Training prints `step=0 ...` and `step=1 ...`.

## Expected Result

- TPU training log shows `token_focus` config.
- No Python tracebacks; training proceeds to `step=0` and beyond.

## Troubleshooting

- **W&B online fails**: if `WANDB_API_KEY` is missing, the runner prints:
  - `wandb disabled due to init error: No API key configured...`
  - Fix: create a `.env` containing `WANDB_API_KEY=...` and sync it:
    - `scripts/sync_env_to_tpu_vm.sh --name <TPU_NAME> --zone us-central2-b --src .env`
- **HF rate limits**: set `HF_TOKEN` or accept slower downloads.
- **v6e quota 0**: use `v4-8` / request quota / change zone.

