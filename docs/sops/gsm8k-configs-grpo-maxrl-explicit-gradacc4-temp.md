# SOP: 仅保留 GRPO/MaxRL 主配置，显式写全并固定 grad_accum_steps=4

- **Title**: SOP: 清理 `projects/gsm8k_grpo/configs`（主目录仅 GRPO/MaxRL，其他迁移 `temp/`）
- **Prereqs**: Python 3；可运行 `projects/gsm8k_grpo/scripts/run_train.py --print-config`
- **Environment (verified)**:
  - Date: 2026-02-10
  - Branch: `max-rl`
  - Python: `3.13`

## 目标

- 主目录只保留 GRPO/MaxRL 配置。
- GRPO/MaxRL 配置全部显式写全（不依赖隐藏默认值）。
- GRPO/MaxRL 统一 `train.grad_accum_steps: 4`。
- DAPO/PPO 配置移至 `projects/gsm8k_grpo/configs/temp/`。

## 步骤（本次实际执行）

- 迁移非 GRPO/MaxRL 配置到 `temp/`：
  - `mv projects/gsm8k_grpo/configs/dapo_standard.yaml projects/gsm8k_grpo/configs/temp/`
  - `mv projects/gsm8k_grpo/configs/dapo_dynamic_sampling_acc.yaml projects/gsm8k_grpo/configs/temp/`
  - `mv projects/gsm8k_grpo/configs/ppo_gae_standard.yaml projects/gsm8k_grpo/configs/temp/`
- 重写主目录 7 个 GRPO/MaxRL YAML：
  - 显式包含 `rollout.dynamic_sampling`、`train.max_length_total`、`train.grad_accum_steps`、`optimizer`、`algo`、`wandb/eval`。
- 逐文件校验：
  - `python3 projects/gsm8k_grpo/scripts/run_train.py --print-config --config <yaml>`
  - 检查输出含 `grad_accum_steps: 4`。

## Expected Result

- `projects/gsm8k_grpo/configs/` 根目录有且仅有 7 个 GRPO/MaxRL YAML。
- `projects/gsm8k_grpo/configs/temp/` 包含 3 个 DAPO/PPO YAML。
- 每个根目录 YAML `--print-config` 成功，且 `grad_accum_steps: 4`。

## References

- `projects/gsm8k_grpo/configs/README.md`
- `projects/gsm8k_grpo/configs/grpo_standard.yaml`
- `projects/gsm8k_grpo/configs/rl_gsm8k_qwen25_3b_batch128_roll8_literal_maxrl_v6e8.yaml`
- `projects/gsm8k_grpo/scripts/run_train.py`
