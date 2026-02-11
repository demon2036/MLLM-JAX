# SOP: Inspect MaxRL vs GRPO implementation differences in this repo

- **Title**: SOP: 对比 `maxrl` 与 `grpo` 在统一 RL runner 中的实现差异
- **Prereqs**: 仓库已检出；可使用 `rg`/`sed`/`diff`
- **Environment (verified)**:
  - Date: 2026-02-11
  - Repo: `MLLM-JAX-max-rl`
  - Branch: `max-rl`

## Steps

1) 定位算法工厂与 estimator 入口

```bash
rg -n "create_algorithm|maxrl|grpo|estimator" plugins/training/rl projects/gsm8k_grpo
```

2) 查看 GRPO 与 MaxRL advantage 公式实现

```bash
sed -n '1,220p' plugins/training/rl/advantage/grpo.py
sed -n '140,220p' plugins/training/rl/advantage/estimators.py
```

3) 查看算法归一化默认值与工厂绑定

```bash
sed -n '1,260p' plugins/training/rl/algorithms/config.py
sed -n '1,140p' plugins/training/rl/algorithms/factory.py
```

4) 查看 runner 是否走同一 update 路径

```bash
sed -n '410,460p' projects/gsm8k_grpo/jax/train.py
sed -n '724,815p' projects/gsm8k_grpo/jax/train.py
sed -n '1,120p' plugins/training/rl/update/modules.py
sed -n '1,120p' plugins/training/rl/update/ppo.py
sed -n '150,260p' MLLM_JAX/train_modules/__init__.py
```

5) 对比 GRPO/MaxRL YAML

```bash
diff -u projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_batch16_roll128_eval1full_v6e8.yaml \
  projects/gsm8k_grpo/configs/rl_gsm8k_qwen25_3b_batch16_roll128_eval1full_maxrl_v6e8.yaml
```

## Expected Result

- 两者共享同一个 runner 主流程（rollout/reward/update）。
- 核心差异仅在 advantage estimator：
  - GRPO: 组内标准化 `(r - mean_group) / (std_group + eps)`
  - MaxRL: 组均值归一化 `(r - mean_group) / (mean_group + eps)`（对齐 upstream `tajwarfahim/maxrl`）
- 默认 `eps` 不同：GRPO 为 `1e-4`，MaxRL 为 `1e-6`。
- 两者默认都走 `policy_gradient` update，不启用 value head。

## Troubleshooting

- 若看到 `value_coef` 等 PPO 字段，请确认当前配置不是 `algo.update.name=ppo`。
- 若 YAML 使用 `max-rl` 别名但行为异常，检查 `normalize_estimator_name` 是否将其归一到 `maxrl`。

## References

- `plugins/training/rl/advantage/grpo.py`
- `plugins/training/rl/advantage/estimators.py`
- `plugins/training/rl/algorithms/config.py`
- `plugins/training/rl/algorithms/factory.py`
- `projects/gsm8k_grpo/jax/train.py`
- `projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_batch16_roll128_eval1full_v6e8.yaml`
- `projects/gsm8k_grpo/configs/rl_gsm8k_qwen25_3b_batch16_roll128_eval1full_maxrl_v6e8.yaml`
