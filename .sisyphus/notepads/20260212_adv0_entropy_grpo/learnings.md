## 2026-02-12T15:56:58Z Task: adv0-entropy-grpo
- Entry point:  (typically launched via ).
- Dataflow: rollout (naive sampler) -> reward funcs (reward_correct/format/tag_count) -> GRPO advantages (group_id baseline) -> PPO-style clipped policy gradient update (TrainGRPOModule) -> W&B log.
- Advantage zeros:  returns exactly 0 when reward equals group mean (common for homogeneous reward groups like all-0 early training).
- Eval modes: periodic eval uses ; full-sweep eval is implemented once after training when .
## 2026-02-12T15:57:24Z Task: adv0-entropy-grpo (corrected)
- Entry point: `python -u projects/gsm8k_grpo/scripts/run_train.py --config <yaml>` (typically launched via `bash scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh --config <yaml>`).
- Key files traced:
  - `projects/gsm8k_grpo/scripts/run_train.py` (YAML -> GRPOGsm8kConfig)
  - `projects/gsm8k_grpo/jax/train.py` (rollout→reward→advantage→update loop + wandb)
  - `plugins/training/rl/advantage/grpo.py` (advantage=0 when centered reward==0)
  - `plugins/training/rl/update/ppo.py` + `plugins/training/core/step/train_step.py` (PPO epochs + grad-accum)
  - `training2.py:get_state` + `MLLM_JAX/train_modules/__init__.py:TrainGRPOModule` (loss implementation)
  - `plugins/sample/workflows/grpo_sync.py` + `plugins/sample/backends/mllm_jax_sampler.py` (naive sampler rollout)
- Advantage zeros: `compute_grpo_advantages_by_group_id` returns exactly 0 when reward equals group mean (common for homogeneous reward groups like all-0 early training).
- Eval modes: periodic eval uses `eval_every_steps` + `eval_batches_per_process` (subset); full-sweep eval is implemented once after training when `eval_full_sweep: true`.

## 2026-02-12T16:49:58Z Task: adv0-entropy-grpo (runner W&B keys)
- Added advantage sign bucket metrics (global allgather):
  - `train-reward/advantage/sign/zero`, `train-reward/advantage/sign/pos`, `train-reward/advantage/sign/neg`
  - `train-reward/advantage/sign/zero_frac`, `train-reward/advantage/sign/pos_frac`, `train-reward/advantage/sign/neg_frac`

## 2026-02-12T16:40:03Z Task: adv0-entropy-grpo (configs)
- Added v6e-8 configs (batch=16, rollout_n=8, steps=400, eval_every_steps=50, eval_split=test, eval_rollout_n=1, eval_full_sweep=true, wandb_project=test-rl, wandb_mode=online):
  - `projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_batch16_roll8_steps400_eval50full_v6e8.yaml` (baseline; `algo.update.kwargs: {}`)
  - `projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_batch16_roll8_steps400_eval50full_adv0entropy_v6e8.yaml` (`algo.update.kwargs.adv0_entropy_coef=0.01`, `adv0_entropy_eps=0.0`)

## 2026-02-12T16:42:40Z Task: adv0-entropy-grpo (adv0 entropy loss)
- Implemented in `MLLM_JAX/train_modules/__init__.py:TrainGRPOModule`:
  - `loss_pg = sum((-min(ratio*adv, clip(ratio)*adv)) * mask_loss) / total_valid_token_count`
  - `loss_entropy_adv0 = -adv0_entropy_coef * sum(token_entropy * mask_loss * 1[abs(adv)<=adv0_entropy_eps]) / total_valid_token_count`
  - `loss = loss_pg + loss_entropy_adv0`
- Metrics emitted: `loss_pg`, `loss_entropy_adv0`, `adv0_token_frac`, `entropy_adv0_mean` (with `entropy_adv0_mean=0` when `adv0_token_count==0`).

## 2026-02-12T17:11:52Z Task: adv0-entropy-grpo (config validation)
- Allowed `algo.update.kwargs.adv0_entropy_coef` and `adv0_entropy_eps` for update `policy_gradient` (incl aliases like `grpo`), defaulting to 0.0; both validated to be >= 0 and preserved in normalized config.

## 2026-02-12T17:18:48Z Commit
- b1cf0a2f4e82fde0bb0d81d4aaca1b8d7f34833f

## 2026-02-12T17:56:20Z Commit
- 05b529731fd6468f1112a795030ab5790ccfc963
- Added W&B `train_log` keys (guarded on `last_meta` presence):
  - `train-adv0_entropy/loss_pg`
  - `train-adv0_entropy/loss_entropy_adv0`
  - `train-adv0_entropy/adv0_token_frac`
  - `train-adv0_entropy/entropy_adv0_mean`
  - `train-adv0_entropy/coef`
  - `train-adv0_entropy/eps`

## 2026-02-12T18:29:28Z Task: adv0-entropy-grpo (tuning config)
- Added a smaller-coef config variant:
  - `projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_batch16_roll8_steps400_eval50full_adv0entropy0p001_v6e8.yaml`
    - `algo.update.kwargs.adv0_entropy_coef: 0.001`
    - `algo.update.kwargs.adv0_entropy_eps: 0.0`
