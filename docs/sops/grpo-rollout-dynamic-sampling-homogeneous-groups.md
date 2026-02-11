# SOP: GRPO/DAPO rollout dynamic sampling（按组内同质性补采）

- **Title**: SOP: 在 GSM8K GRPO runner 中启用 rollout dynamic sampling（homogeneous-group trigger）
- **Prereqs**: Python 3；使用 `projects/gsm8k_grpo/scripts/run_train.py` 启动
- **Environment (verified)**:
  - Date: 2026-02-10
  - Branch: `max-rl`
  - Python: `3.13`

## 适用场景

- 你希望“组内 reward/acc 全同”时继续 rollout，而不是直接用该组更新。
- 你希望把该逻辑放在 rollout 阶段，而不是改 advantage estimator 数学。

## 配置结构（v3）

- 位置：`rollout.dynamic_sampling`
- 关键字段：
  - `enabled`: 是否启用
  - `metric`: `acc` 或 `seq_reward`
  - `homogeneity_threshold`: 同质判定阈值（1.0=全同）
  - `min_unique_reward_values`: 至少多少个唯一值才算“非同质”
  - `max_extra_roll_rounds`: 最多补采轮数
  - `target_valid_groups`: 目标有效组数（null => `rollout.batch_size`）
  - `fallback_policy`: `keep_last` 或 `drop_group`

## 本次验证命令（实际执行）

- 默认打印（dynamic 关闭）：
  - `python3 projects/gsm8k_grpo/scripts/run_train.py --print-config`
- 覆盖开启 dynamic：
  - `python3 projects/gsm8k_grpo/scripts/run_train.py --print-config --config '' --set rollout.dynamic_sampling.enabled=true --set rollout.dynamic_sampling.metric=seq_reward`
- 非法 metric 校验：
  - `python3 projects/gsm8k_grpo/scripts/run_train.py --print-config --config '' --set rollout.dynamic_sampling.metric=foo`

## Expected Result

- 默认配置出现 `rollout.dynamic_sampling.enabled: false`。
- 覆盖后出现 `enabled: true` 且 `metric: seq_reward`。
- 非法 metric 启动时抛错：`rollout.dynamic_sampling.metric must be one of: acc, seq_reward`。

## References

- `projects/gsm8k_grpo/config_schema.py`
- `projects/gsm8k_grpo/scripts/run_train.py`
- `projects/gsm8k_grpo/jax/train.py`
- `plugins/training/rl/rollout/dynamic_sampling.py`
- `projects/gsm8k_grpo/configs/dapo_dynamic_sampling_acc.yaml`

