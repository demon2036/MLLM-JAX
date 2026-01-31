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

## Run 3: pb32 batch semantics aligned with upstream (unique prompts per step)

- Repo branch/commit: `nano-gpt-sft` @ `af0b0f7`
- Config:
  - `projects/minionerec_rl/configs/v4-8/minionerec_rl_jax_qwen25_1p5b_base_industrial_v4_8_steps100_pb32_align_minionerec_rl_sh_20260131.yaml`
- W&B run: `johntitordemon2036/minionerec-sid-rl/runs/gkobyl1b` (mode=online)
  - Logs include per-reward means: `train/reward_rule_mean`, `train/reward_ndcg_mean`, `train/reward_mean`
- Output dir:
  - `runs/minionerec_rl_jax_qwen25_1p5b_base_industrial_v4_8_steps100_pb32_align_minionerec_rl_sh_20260131/`
- Eval (samples=4533, invalid=0):
  - HR@K: 1=`0.08317`, 3=`0.10523`, 5=`0.12067`, 10=`0.15067`, 20=`0.18575`, 50=`0.23980`
  - NDCG@K: 1=`0.08317`, 3=`0.09584`, 5=`0.10213`, 10=`0.11177`, 20=`0.12062`, 50=`0.13132`

## Run 4: save best-on-valid SFT checkpoint (then use it for RL init)

- Repo branch/commit: `nano-gpt-sft` @ `c2c3e2c`
  - Adds `train.save_best`/`train.save_best_metric` and `eval.split` to `projects/sid_sft`.
- Config:
  - `projects/sid_sft/configs/train/v4-8/sid_sft_jax_qwen25_1p5b_base_industrial_v4_8_e3_muon_refactor_20260131_train_bestval.yaml`
- Command:
  - `./scripts/ssh_tpu_vm_root.sh --name plugins-refactor-sid-sft-muon-260131052355 --zone us-central2-b --command 'set -euo pipefail; export PYTHONUNBUFFERED=1; export HF_HUB_ENABLE_HF_TRANSFER=1; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; ./scripts/run_sid_sft.sh --config projects/sid_sft/configs/train/v4-8/sid_sft_jax_qwen25_1p5b_base_industrial_v4_8_e3_muon_refactor_20260131_train_bestval.yaml --run-mode train_eval'`
- W&B run: `johntitordemon2036/minionerec-sid-sft/runs/jiykssr3` (mode=online)
- Output dir:
  - `runs/sid_sft_jax_qwen25_1p5b_base_industrial_v4_8_e3_muon_refactor_20260131_train_bestval/`
  - Best checkpoint:
    - `runs/sid_sft_jax_qwen25_1p5b_base_industrial_v4_8_e3_muon_refactor_20260131_train_bestval/sft_state_best.msgpack`
- Eval (valid split, beams=20, samples=4532, invalid=0):
  - HR@K: 1=`0.08694`, 3=`0.11584`, 5=`0.12842`, 10=`0.15181`, 20=`0.17829`
  - NDCG@K: 1=`0.08694`, 3=`0.10350`, 5=`0.10865`, 10=`0.11617`, 20=`0.12287`

## Run 5: RL pb32 init from SFT best-on-valid checkpoint

- Repo branch/commit: `nano-gpt-sft` @ `c2c3e2c`
- Config:
  - `projects/minionerec_rl/configs/v4-8/minionerec_rl_jax_qwen25_1p5b_base_industrial_v4_8_steps100_pb32_from_sft_bestval_20260131.yaml`
- Command:
  - `./scripts/ssh_tpu_vm_root.sh --name plugins-refactor-sid-sft-muon-260131052355 --zone us-central2-b --command 'set -euo pipefail; export PYTHONUNBUFFERED=1; export HF_HUB_ENABLE_HF_TRANSFER=1; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; ./scripts/run_minionerec_rl.sh --config projects/minionerec_rl/configs/v4-8/minionerec_rl_jax_qwen25_1p5b_base_industrial_v4_8_steps100_pb32_from_sft_bestval_20260131.yaml --run-mode train_eval'`
- W&B run: `johntitordemon2036/minionerec-sid-rl/runs/teo6pia7` (mode=online)
- Output dir:
  - `runs/minionerec_rl_jax_qwen25_1p5b_base_industrial_v4_8_steps100_pb32_from_sft_bestval_20260131/`
- Eval (test split, beams=50, samples=4533, invalid=0):
  - HR@K: 1=`0.08317`, 3=`0.10523`, 5=`0.12067`, 10=`0.15067`, 20=`0.18575`, 50=`0.23980`
  - NDCG@K: 1=`0.08317`, 3=`0.09584`, 5=`0.10213`, 10=`0.11177`, 20=`0.12062`, 50=`0.13132`

## Run 6: RL (valid split eval, beams=20) with TRL sync defaults + fast-attn patch

- Repo branch/commit: `nano-gpt-sft` @ `5c87b61`
  - Applies `patch_qwen2_attention_decode_fast()` for faster decode on TPU.
- Config:
  - `projects/minionerec_rl/configs/v4-8/minionerec_rl_jax_qwen25_1p5b_base_industrial_v4_8_steps640_pb32_sync512_a06_fastattn_from_sft_bestval_evalvalid_beam20_20260131.yaml`
- Command:
  - `./scripts/ssh_tpu_vm_root.sh --name plugins-refactor-sid-sft-muon-260131052355 --zone us-central2-b --command 'set -euo pipefail; export PYTHONUNBUFFERED=1; export HF_HUB_ENABLE_HF_TRANSFER=1; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; ./scripts/run_minionerec_rl.sh --config projects/minionerec_rl/configs/v4-8/minionerec_rl_jax_qwen25_1p5b_base_industrial_v4_8_steps640_pb32_sync512_a06_fastattn_from_sft_bestval_evalvalid_beam20_20260131.yaml --run-mode train_eval'`
- W&B run: `johntitordemon2036/minionerec-sid-rl/runs/ml2v3yqy` (mode=online)
  - Reward breakdown logs: `rewards/rule_reward`, `rewards/ndcg_rule_reward`, `reward`, `reward_std`
- Output dir:
  - `runs/minionerec_rl_jax_qwen25_1p5b_base_industrial_v4_8_steps640_pb32_sync512_a06_fastattn_from_sft_bestval_evalvalid_beam20_20260131/`
- Artifacts:
  - `runs/minionerec_rl_jax_qwen25_1p5b_base_industrial_v4_8_steps640_pb32_sync512_a06_fastattn_from_sft_bestval_evalvalid_beam20_20260131/sft_state_rl_last.msgpack`
  - `runs/minionerec_rl_jax_qwen25_1p5b_base_industrial_v4_8_steps640_pb32_sync512_a06_fastattn_from_sft_bestval_evalvalid_beam20_20260131/eval_predictions.json`
  - `runs/minionerec_rl_jax_qwen25_1p5b_base_industrial_v4_8_steps640_pb32_sync512_a06_fastattn_from_sft_bestval_evalvalid_beam20_20260131/eval_predictions.metrics.json`
- Eval (valid split, beams=20, samples=4532, invalid=0):
  - HR@K: 3=`0.11231`, 5=`0.12732`, 10=`0.14850`
  - NDCG@K: 3=`0.10218`, 5=`0.10827`, 10=`0.11506`

## Cleanup

- Per task requirement, this run did **not** delete the TPU VM.
  When you want to stop billing:
  - `./scripts/delete_tpu_vm.sh --name plugins-refactor-sid-sft-muon-260131052355 --zone us-central2-b`
