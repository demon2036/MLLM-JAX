# SOP: Inspect AReaL config system (Hydra + OmegaConf + dataclass defaults)

- **Title**: SOP: 复盘 AReaL 配置系统（结构化默认值 + CLI 覆盖 + 落盘）并提炼可迁移实践
- **Prereqs**: 本仓库已存在 `workdir/areal` clone；可运行 `python3` 与 `rg`
- **Environment (verified)**:
  - Date: 2026-02-10
  - Repo: `workdir/areal`
  - Commit: `0d11df6`

## 目标

- 明确 AReaL 如何解析配置（YAML + CLI override）。
- 明确默认值如何补齐（structured merge）。
- 明确最终配置如何保存与上报（log_dir `config.yaml` + W&B config）。
- 对比本仓库现状，提炼可直接借鉴的最小方案。

## 步骤（本次实际执行命令）

1) 定位配置入口与解析链路

- `rg -n "def parse_cli_args|hydra|OmegaConf|to_structured_cfg|load_expr_config|save_config" workdir/areal/areal/api/cli_args.py`
- `sed -n '1991,2058p' workdir/areal/areal/api/cli_args.py`

2) 查看核心 dataclass（optimizer / actor / ppo / grpo）

- `sed -n '286,360p' workdir/areal/areal/api/cli_args.py`
- `sed -n '919,1015p' workdir/areal/areal/api/cli_args.py`
- `sed -n '1953,1989p' workdir/areal/areal/api/cli_args.py`

3) 查看真实算法配置样例（GRPO / DAPO dynamic / PPO）

- `sed -n '1,220p' workdir/areal/examples/math/gsm8k_grpo.yaml`
- `sed -n '1,230p' workdir/areal/examples/math/gsm8k_dapo_dynamic_bs.yaml`
- `sed -n '1,230p' workdir/areal/examples/math/gsm8k_ppo.yaml`

4) 查看配置如何进入 W&B 与本地日志

- `sed -n '1590,1688p' workdir/areal/areal/api/cli_args.py`
- `sed -n '1,120p' workdir/areal/areal/utils/stats_logger.py`

## 关键结论

- AReaL 使用 **Hydra + OmegaConf + dataclass**：
  - `parse_cli_args(...)` 读取 `--config` 与 override。
  - `to_structured_cfg(...)` 用 `OmegaConf.structured(config_cls)` + `OmegaConf.merge(...)` 补齐 Python dataclass 默认值。
  - `load_expr_config(...)` 将配置转为 typed object，并在 rank0 调用 `save_config(...)` 落盘。
- AReaL 的 optimizer 与 actor 参数是 **typed fixed fields**（例如 `optimizer.type/lr/beta1/...`，`actor.eps_clip/gae_lambda/...`），不是插件式 `name+kwargs`。
- AReaL 的 `GRPOConfig` 是 `PPOConfig` 的兼容壳，算法切换主要靠 `actor.*` 参数组合与 YAML 模板切换。
- DAPO 的动态采样在 AReaL 里通过 `dynamic_bs` 控制，默认是 `false`；仅 `gsm8k_dapo_dynamic_bs.yaml` 显式设为 `true`。
- W&B 配置上报为完整实验配置（`asdict(self.exp_config)`），且同时写本地 `config.yaml`，追溯性强。

## 对本仓库的直接建议

- 保留当前插件 schema（`name + kwargs`）以满足“可扩展且低侵入”。
- 借鉴 AReaL 的三点：
  - 启动时执行“结构化默认值补齐 + 严格类型/范围校验”；
  - 将最终解析配置完整落盘到 run 目录；
  - W&B 记录“最终解析后的有效配置”。
- 对 DAPO 增加显式开关（如 `algo.estimator.kwargs.dynamic_sampling: false`）时，默认必须 `false`，仅在 DAPO 专用 YAML 显式打开。

## Expected Result

- 能明确回答“某个参数默认从哪里来、何时生效、何时上报 W&B”。
- 能将“标准 GRPO”与“DAPO/GAE/PPO”行为在配置层严格分离，避免 run 配置混杂。

## References

- AReaL repo: `https://github.com/inclusionAI/AReaL`
- `workdir/areal/areal/api/cli_args.py`
- `workdir/areal/areal/utils/stats_logger.py`
- `workdir/areal/examples/math/gsm8k_grpo.yaml`
- `workdir/areal/examples/math/gsm8k_dapo_dynamic_bs.yaml`
- `workdir/areal/examples/math/gsm8k_ppo.yaml`
