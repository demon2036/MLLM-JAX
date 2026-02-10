# SOP: GSM8K GRPO/MaxRL (batch_size=16, rollout.n=128) with eval_rollout_n=1 + full sweep

- **Title**: SOP: 使用 YAML 显式配置 train rollout 与 eval rollout 解耦（train n=128, eval n=1）并在 TPU 上运行
- **Prereqs**: TPU VM 可 SSH；`/root/MLLM-JAX` 已 Git 同步到目标分支；`/root/.env` 含 `WANDB_API_KEY`
- **Environment (verified)**:
  - Date: 2026-02-10
  - Branch: `max-rl`
  - Commit: `2ed1dad`

## 关键配置点

- 训练配置：
  - `rollout.batch_size: 16`
  - `rollout.n: 128`
- 评估配置（与训练解耦）：
  - `eval_rollout_n: 1`
  - `eval_full_sweep: true`
  - `eval_every_steps: 0`（仅末尾全量评估一次）

## 配置文件

- GRPO:
  - `projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_batch16_roll128_eval1full_v6e8.yaml`
- MaxRL:
  - `projects/gsm8k_grpo/configs/rl_gsm8k_qwen25_3b_batch16_roll128_eval1full_maxrl_v6e8.yaml`

## 启动脚本

- GRPO:
  - `bash scripts/tpu_vm_start_grpo_gsm8k_bs16_roll128_eval1full_nohup.sh --env-name mllm-jax`
- MaxRL:
  - `bash scripts/tpu_vm_start_maxrl_gsm8k_bs16_roll128_eval1full_nohup.sh --env-name mllm-jax`

## 验证命令

- 打印最终配置（不启动训练）：
  - `python3 projects/gsm8k_grpo/scripts/run_train.py --config <yaml> --print-config`
- 检查日志中 eval 字段：
  - `grep -n "eval_rollout_n\|eval_full_sweep" logs/nohup_*_latest.log`
- 检查 W&B run：
  - 日志中 `wandb: 🚀 View run at ...`

## Expected Result

- 训练按 `rollout.n=128` 运行。
- 评估按 `eval_rollout_n=1` 运行（不复用训练 n）。
- 训练结束后执行一次 `eval_full` 全量评估并上报 W&B。
