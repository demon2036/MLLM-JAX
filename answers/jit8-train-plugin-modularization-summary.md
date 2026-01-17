# jit8 训练模块化实验总结（对齐 `test_jit8.py`）

本次实验在 `john` 分支上进行，目标是：在不侵入 `MLLM_JAX/` 与 `training2.py` 的前提下，沿用 `test_jit8.py` 的入口行为，同时把训练 pipeline 拆成更清晰的可复用模块（参考 AReaL 的分层与“契约优先”思路）。

## 1) 当前 `test_jit8.py` 入口（必须对齐的行为）

- `test_jit8.py` 现在是 CLI 包装器：调用 `plugins.jit8_train.run.cli_main()`。
- 因此“对齐 test_jit8”意味着：`plugins/jit8_train/*` 的 CLI 与训练路径必须保持兼容。

## 2) 本次新增/调整的模块（plugins-first）

### 2.1 新增：训练契约（schema/contracts）

- 新增 `plugins/training/api/batch_schema.py`
  - 提供 `validate_grpo_batch(batch, stage=...)`，用于校验 GRPO batch/trajectory 的 *结构不变量*（keys、rank、shape 对齐）。
  - 支持阶段化校验：`rollout` / `rewarded` / `advantaged` / `train_step` / `train_ready`。
- 新增 `plugins/training/api/interfaces.py`
  - 提供若干 `Protocol`（Sampler/Workflow/RewardFunction）的最小签名，便于后续继续拆分 runner/workflow/algo。

### 2.2 扩展：jit8 插件可选 schema 校验开关

- `plugins/jit8_train/config.py` 与 `plugins/jit8_train/configs/gsm8k_default.yaml` 增加：
  - `validate_schema: false`（默认关闭，避免额外开销；重构时可打开做断言）
- `plugins/jit8_train/run.py` 在关键阶段挂了校验（开启时失败会报 `step/stage` 上下文）：
  - rollout 结束后、写入 rewards 后、写入 advantages 后、进入 train_step 前、生成 old_logps 后（train_ready）

### 2.3 拆分：把“奖励计算”和“PPO/GRPO 更新”从大 loop 中抽出来

- 新增 `plugins/jit8_train/rewarding.py`
  - `compute_weighted_rewards(...)`：计算每个 reward function 的分量与加权总和（与旧逻辑保持一致：异常样本 reward=-1）。
- 新增 `plugins/jit8_train/update.py`
  - `ppo_update(...)`：封装 `ppo_steps × grad_accum_steps` 的 update 过程，并在第一个 PPO step 后生成 `old_per_token_logps`（用于后续 step）。
- `plugins/jit8_train/run.py` 相应改为显式 phases：
  - collect_rollout（已有 `sampling.py`）→ compute_rewards → compute_advantages → ppo_update → log

### 2.4 修复：`jnp.concat` → `jnp.concatenate`

- `plugins/jit8_train/run.py` / `plugins/jit8_train/update.py` 统一使用 `jnp.concatenate(..., axis=0)`，避免 `jax.numpy` 中不存在 `concat` 的潜在运行时问题。

## 3) 本地验证（本机无需 JAX 也能跑的部分）

已实际执行并通过：

- 语法检查：`python -m py_compile test_jit8.py plugins/jit8_train/run.py plugins/jit8_train/rewarding.py plugins/jit8_train/update.py plugins/training/api/batch_schema.py tests/test_jit8_schema_and_cli.py`
- CLI config smoke：
  - `python test_jit8.py --print-config`
  - `python test_jit8.py --print-config --set validate_schema=true`
- 轻量回归测试（不依赖 JAX）：`python tests/test_jit8_schema_and_cli.py`

说明：

- `python -m unittest -v` 在本机环境会因缺少 `flax` 导致全量 discovery 失败（`MLLM_JAX/train_modules` 导入报错）；本次用“单文件运行”方式验证新增测试脚本。

## 4) 如何开启/使用 schema 校验（推荐重构时开启）

- 仅验证配置合并（不跑训练）：`python test_jit8.py --print-config --set validate_schema=true`
- 真正训练时启用（TPU/GPU 环境需要 JAX + 依赖）：`python test_jit8.py --set validate_schema=true`

## 5) 相关 SOP 更新

- `docs/sops/grpo-gsm8k-jit8-yaml-config.md`：补充了 `validate_schema` 开关与本地测试命令
- `docs/sops/areal-rl-organization.md`：更新为 repo 内 `workdir/areal` 的 AReaL clone + 阅读命令
- `docs/sops/training-modularization-plan.md`：更新为“jit8 plugin-first”的实际路径与本地验证命令
- `docs/sops/git-main-copy-and-switch-to-john.md`：补充了 `main--copy` + 切换 `john` 的可复现步骤

## 6) 下一步建议（对齐 AReaL 的更完整分层）

1. **Workflow 进一步对象化**：把 `sampling.py + reward_funcs` 组合成可替换的 workflow（类似 AReaL `RolloutWorkflow`）。
2. **Algorithm 层收敛**：把 advantage estimator / clipping / KL / entropy 等作为 config-driven toggles（避免复制训练脚本）。
3. **异步 rollout 演进**：引入 AReaL 风格的 `submit/wait/prepare_batch` + staleness/version 控制（先把 `versions` 字段写进 schema）。

