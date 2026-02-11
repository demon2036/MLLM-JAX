# GRPO/MaxRL 默认配置与 estimator-specific 默认值（v3）

- **Title**: SOP: 统一 RL 配置为 plugin schema（v3），并按 estimator/update 类型应用默认值
- **Prereqs**: Python 3；仓库检出到 `max-rl` 分支
- **Environment (verified)**:
  - Date: 2026-02-10
  - Branch: `max-rl`
  - Python: `3.13`

## 目标

- 默认行为回到“标准 GRPO”：不引入 value model，不启用 DAPO/RLOO/GAE 专属逻辑。
- 配置结构可插拔且低侵入：新增算法块时不改训练主循环，只扩展 schema + factory。
- W&B 侧可追踪且不混乱：只展示当前激活分支。

## v3 schema 形状（核心约束）

- 估计器统一为：
  - `algo.estimator = {name, kwargs}`
- 更新器统一为：
  - `algo.update = {name, kwargs}`
- 优化器统一为：
  - `train.optimizer = {name, kwargs, lr_schedule:{name, kwargs}}`
- dynamic sampling 统一为 rollout 阶段：
  - `rollout.dynamic_sampling = {...}`
- 旧平铺键全部拒绝（breaking）：
  - 例如 `algo.update.value_coef`、`algo.estimator.gae_gamma`、`algo.estimator.dapo_alpha`

## 默认值策略（当前实现）

- 默认算法：`algo.name=grpo`
- 默认 estimator：`grpo`
- 默认 update：`policy_gradient`
- 默认 GRPO/MaxRL 不含 value-head 参数。
- 默认 dynamic sampling：`rollout.dynamic_sampling.enabled=false`
- 仅当 `algo.update.name=ppo` 时，才生效 `algo.update.ppo` 默认：
  - `value_coef: 0.5`
  - `value_clip_range: 0.2`
  - `entropy_coef: 0.0`
- 仅当 estimator 为对应类型时，才生效其默认块：
  - `dapo -> {alpha: 0.2}`
  - `rloo -> {whiten: true}`
  - `gae -> {gamma: 1.0, gae_lambda: 0.95, normalize: true}`

## Dynamic sampling 语义（rollout-stage）

- dynamic sampling 不属于 estimator 数学定义；它是 rollout 采样策略。
- 当前触发策略：按 prompt-group 同质性（默认 metric=`acc`）触发补采。
- 预算由 `max_extra_roll_rounds` 限制，避免无限采样。
- 预算耗尽时由 `fallback_policy` 决定保留或丢弃 unresolved groups。

## 步骤（本次实际执行）

- 语法与核心回归：
  - `python3 -m py_compile plugins/training/rl/algorithms/config.py plugins/training/rl/algorithms/factory.py plugins/training/rl/ppo/state.py projects/gsm8k_grpo/scripts/run_train.py projects/gsm8k_grpo/config_schema.py projects/gsm8k_grpo/jax/train.py`
- 关键测试：
  - `python3 -m pytest -q tests/test_rl_config_schema_v2.py tests/test_grpo_training_print_config_cli.py tests/test_rl_algorithm_factory_maxrl.py tests/test_advantage_estimators.py`
  - `python3 -m pytest -q tests/test_rollout_backend_factory.py tests/test_grpo_batching_inference.py`
- CLI 行为验证：
  - `python3 projects/gsm8k_grpo/scripts/run_train.py --print-config`
  - `python3 projects/gsm8k_grpo/scripts/run_train.py --print-config --config '' --set algo.name=ppo --set algo.estimator.name=gae --set algo.update.name=ppo --set algo.update.ppo={}`
  - `python3 projects/gsm8k_grpo/scripts/run_train.py --print-config --config '' --set algo.update.value_coef=0.5`

## Expected Result

- 默认 `--print-config` 的 `algo` 为：`grpo + policy_gradient`，不出现 value-head 字段。
- `ppo + gae` 时出现 `algo.update.ppo.value_coef/value_clip_range/entropy_coef`。
- 旧键注入时抛错：`Detected deprecated RL config keys`。

## References

- `plugins/training/rl/algorithms/config.py`
- `plugins/training/rl/algorithms/factory.py`
- `plugins/training/rl/config.py`
- `plugins/training/rl/ppo/state.py`
- `projects/gsm8k_grpo/scripts/run_train.py`
- `projects/gsm8k_grpo/config_schema.py`
- `projects/gsm8k_grpo/jax/train.py`
- `tests/test_rl_config_schema_v2.py`
- `tests/test_grpo_training_print_config_cli.py`
