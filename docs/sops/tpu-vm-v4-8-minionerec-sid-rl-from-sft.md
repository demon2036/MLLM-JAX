# SOP: TPU v4-8 MiniOneRec SID RL (from saved SID SFT checkpoint)

- **Title**: SOP: Run MiniOneRec SID RL (GRPO-style) on TPU v4-8 starting from a saved SID SFT checkpoint and report HR@K/NDCG@K
- **Prereqs**: TPU VM already bootstrapped with conda env `mllm-jax`; `workdir/MiniOneRec` present; W&B API key available
- **Environment (verified)**:
  - Date: 2026-01-31
  - TPU VM: `v4-8` spot, zone `us-central2-b`
  - TPU_NAME: `plugins-refactor-sid-sft-muon-260131052355`
  - Python: `3.12.12` (conda env `mllm-jax`)
  - JAX: `0.9.0` + `jaxlib 0.9.0`, backend `tpu`, `device_count=4`
  - Repo branch/commit: `nano-gpt-sft` @ `fc6f76c`

## Notes

- This run initializes params from the **saved SID SFT last checkpoint**:
  `runs/sid_sft_jax_qwen25_1p5b_base_industrial_v4_8_e3_muon_refactor_20260131_train/sft_state_last.msgpack`.
- To avoid TPU compile OOM on v4-8, this config uses `train.grad_accum_steps: 16`.

## Steps (commands actually run)

### 1) Provide W&B key via `.env` (do NOT commit)

- Create local `.env`:
  - `WANDB_API_KEY=<your_key>`
- Sync to TPU:
  - `./scripts/sync_env_to_tpu_vm.sh --name plugins-refactor-sid-sft-muon-260131052355 --zone us-central2-b --worker all`

### 2) Update repo on TPU via Git (no SCP)

- `./scripts/ssh_tpu_vm_root.sh --name plugins-refactor-sid-sft-muon-260131052355 --zone us-central2-b --command 'set -euo pipefail; cd /root/MLLM-JAX; git fetch --all --prune; git checkout nano-gpt-sft; git pull; git rev-parse --short HEAD'`

### 3) Train+Eval (RL)

- Config:
  - `projects/minionerec_rl/configs/v4-8/minionerec_rl_jax_qwen25_1p5b_base_industrial_v4_8_steps100_from_sft_last_20260131.yaml`
- Command:
  - `./scripts/ssh_tpu_vm_root.sh --name plugins-refactor-sid-sft-muon-260131052355 --zone us-central2-b --command 'set -euo pipefail; export PYTHONUNBUFFERED=1; export HF_HUB_ENABLE_HF_TRANSFER=1; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; ./scripts/run_minionerec_rl.sh --config projects/minionerec_rl/configs/v4-8/minionerec_rl_jax_qwen25_1p5b_base_industrial_v4_8_steps100_from_sft_last_20260131.yaml --run-mode train_eval'`

## Expected result

- Output dir:
  - `runs/minionerec_rl_jax_qwen25_1p5b_base_industrial_v4_8_steps100_from_sft_last_20260131/`
- Artifacts:
  - `runs/minionerec_rl_jax_qwen25_1p5b_base_industrial_v4_8_steps100_from_sft_last_20260131/sft_state_rl_last.msgpack`
  - `runs/minionerec_rl_jax_qwen25_1p5b_base_industrial_v4_8_steps100_from_sft_last_20260131/run_summary.json`
  - `runs/minionerec_rl_jax_qwen25_1p5b_base_industrial_v4_8_steps100_from_sft_last_20260131/eval_predictions.json`
  - `runs/minionerec_rl_jax_qwen25_1p5b_base_industrial_v4_8_steps100_from_sft_last_20260131/eval_predictions.metrics.json`

## Verified results (2026-01-31)

- W&B run: `johntitordemon2036/minionerec-sid-rl/runs/p3qigxkv` (mode=online)
- Eval (samples=4533, invalid=0):
  - HR@K: 1=`0.08295`, 3=`0.10611`, 5=`0.12111`, 10=`0.14979`, 20=`0.18575`, 50=`0.23781`
  - NDCG@K: 1=`0.08295`, 3=`0.09629`, 5=`0.10237`, 10=`0.11154`, 20=`0.12067`, 50=`0.13097`

## Run 2: align with upstream `workdir/MiniOneRec/rl.sh` (reward logging + GRPO loss)

- Repo branch/commit: `nano-gpt-sft` @ `e28145d`
- Config:
  - `projects/minionerec_rl/configs/v4-8/minionerec_rl_jax_qwen25_1p5b_base_industrial_v4_8_steps100_align_minionerec_rl_sh_20260131.yaml`
- Command:
  - `./scripts/ssh_tpu_vm_root.sh --name plugins-refactor-sid-sft-muon-260131052355 --zone us-central2-b --command 'set -euo pipefail; export PYTHONUNBUFFERED=1; export HF_HUB_ENABLE_HF_TRANSFER=1; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; ./scripts/run_minionerec_rl.sh --config projects/minionerec_rl/configs/v4-8/minionerec_rl_jax_qwen25_1p5b_base_industrial_v4_8_steps100_align_minionerec_rl_sh_20260131.yaml --run-mode train_eval'`
- W&B run: `johntitordemon2036/minionerec-sid-rl/runs/sr8u7pgg` (mode=online)
  - Logs include per-reward means: `train/reward_rule_mean`, `train/reward_ndcg_mean`, `train/reward_mean`
- Output dir:
  - `runs/minionerec_rl_jax_qwen25_1p5b_base_industrial_v4_8_steps100_align_minionerec_rl_sh_20260131/`
- Eval (samples=4533, invalid=0):
  - HR@K: 1=`0.08317`, 3=`0.10611`, 5=`0.12199`, 10=`0.15067`, 20=`0.18619`, 50=`0.24311`
  - NDCG@K: 1=`0.08317`, 3=`0.09626`, 5=`0.10266`, 10=`0.11183`, 20=`0.12087`, 50=`0.13217`

## Cleanup

- Per task requirement, this run did **not** delete the TPU VM.
  When you want to stop billing:
  - `./scripts/delete_tpu_vm.sh --name plugins-refactor-sid-sft-muon-260131052355 --zone us-central2-b`
