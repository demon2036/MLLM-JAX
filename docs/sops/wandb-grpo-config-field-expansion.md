# W&B GRPO/MaxRL 配置字段来源与 v3 显式化修复（含 rollout dynamic sampling）

- **Title**: SOP: 解释 W&B 中 GRPO/MaxRL 配置字段，并将日志配置改为仅显示当前激活分支（v3 + dynamic sampling）
- **Prereqs**: 可运行本地 CLI；若需查线上 run，需可用 W&B API key
- **Environment (verified)**:
  - Date: 2026-02-10
  - Branch: `max-rl`
  - Python: `3.13`

## 适用场景

- 你在 W&B 配置里看到一些“并非当前算法”的字段，怀疑 YAML 与 run 不一致。
- 你希望 run config 只展示当前激活算法所需字段，减少误解与面板噪音。

## 步骤（本次实际执行）

- 本地打印默认 GRPO 解析后配置（确认最终上报结构）：
  - `python3 projects/gsm8k_grpo/scripts/run_train.py --print-config`
- 本地打印 PPO+GAE 的解析后配置（确认 value head 字段仅在 PPO 分支出现）：
  - `python3 projects/gsm8k_grpo/scripts/run_train.py --print-config --config '' --set algo.estimator.name=gae --set algo.update.name=ppo`
- 本地打印 dynamic sampling 覆盖配置（确认字段进入 rollout 而非 algo.estimator）：
  - `python3 projects/gsm8k_grpo/scripts/run_train.py --print-config --config '' --set rollout.dynamic_sampling.enabled=true --set rollout.dynamic_sampling.metric=seq_reward`
- 本地验证旧平铺键被拒绝（防止旧 schema 混入 W&B）：
  - `python3 projects/gsm8k_grpo/scripts/run_train.py --print-config --config '' --set algo.update.value_coef=0.5`
- 对照代码链路：
  - `projects/gsm8k_grpo/config_schema.py`（`to_logging_dict`）
  - `plugins/training/rl/algorithms/config.py`（`sanitize_algo_config_for_logging`）
  - `projects/gsm8k_grpo/jax/train.py`（`maybe_init_wandb(cfg=cfg.to_logging_dict(), ...)`）

## 结论

- v3 改造后，W&B 上报使用 `cfg.to_logging_dict()`，并通过 `sanitize_algo_config_for_logging(...)` 仅保留当前激活的 estimator/update 分支。
- 默认 `grpo`/`maxrl` 不再展示 PPO value-head 字段；`value_coef/value_clip_range/entropy_coef` 仅在 `algo.update.name=ppo` 时出现。
- 旧平铺键（如 `algo.update.value_coef`、`algo.estimator.gae_gamma`）会被 CLI 直接拒绝，避免“历史字段混入新 run”。
- dynamic sampling 字段位于 `rollout.dynamic_sampling.*`；不会与 estimator 参数混合。

## Expected Result

- `--print-config` 的 `algo` 段始终是 plugin 分支结构：
  - `algo.estimator.{name, kwargs}`
  - `algo.update.{name, kwargs}`
- `--print-config` 的 `rollout` 段包含 `dynamic_sampling` 显式块。
- `grpo/maxrl` 配置中不出现 PPO 的 value-head 字段。
- 使用旧键启动时进程非零退出并报 `Detected deprecated RL config keys`。

## Troubleshooting

- 若 W&B 仍出现旧字段：
  - 先确认是否真的使用了新版入口 `projects/gsm8k_grpo/scripts/run_train.py`。
  - 检查是否通过 `--set` 注入了旧键。
  - 检查是否是旧 commit 启动（看 run 中的 `git.commit`）。

## References

- `projects/gsm8k_grpo/scripts/run_train.py`
- `projects/gsm8k_grpo/config_schema.py`
- `projects/gsm8k_grpo/jax/train.py`
- `plugins/training/rl/algorithms/config.py`
