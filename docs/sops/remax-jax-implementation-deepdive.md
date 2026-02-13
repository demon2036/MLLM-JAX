# SOP: Deep-dive `ReMax` (JAX) implementation in this repo

- **Title**: SOP: Audit and understand the JAX `ReMax` implementation (rollout → reward → advantage → update)
  **Prereqs**: None (code-reading only)
  **Scope**: `plugins/training/rl/remax/`, `plugins/training/rl/rollout/backends/remax_mixed_naive.py`, `plugins/training/rl/advantage/*`, `projects/gsm8k_grpo/jax/train.py`

## Goal

快速回答：

- 这个仓库里的 `ReMax` 在 JAX 里是如何落地的（模块拆分 / 数据流 / shape 语义）？
- 是否对齐官方 `liziniu/ReMax`（DeepSpeed-Chat）语义：greedy baseline + token-local KL shaping + 折扣传播 terminal advantage？
- 关键配置约束在哪里做（避免 silent misconfig）？

## Steps (commands used)

定位实现文件与 wiring：

```bash
rg -n --ignore-case "remax" docs memory plugins projects tests | head -n 200
rg -n --ignore-case "remax" -S . | head -n 200
```

看核心实现（loss / state / rollout / advantage）：

```bash
sed -n '1,220p' plugins/training/rl/remax/module.py
sed -n '1,220p' plugins/training/rl/remax/state.py
sed -n '1,240p' plugins/training/rl/rollout/backends/remax_mixed_naive.py
sed -n '150,280p' plugins/training/rl/advantage/estimators.py
```

看 runner 如何选择 ReMax state + rollout backend：

```bash
sed -n '420,580p' projects/gsm8k_grpo/jax/train.py
sed -n '430,520p' projects/gsm8k_grpo/scripts/run_train.py
```

## Expected result

你应该能定位到如下组件，并理解它们如何拼装成 ReMax：

1. **Rollout (ReMax mixed, baseline injected)**：
   - backend: `plugins/training/rl/rollout/backends/remax_mixed_naive.py`
   - 约定：caller 传入的 `prompts` 已经按 `group_size=rollout.n` 重复成 prompt-group
   - 每组里 `baseline_position` 对应的那一行用 **greedy decoding**；其余行用 stochastic sampler
   - baseline 行会被强制 `labels=0`（completion mask 为空），从而 **不产生梯度**

2. **Reward (host-side)**：
   - runner 里 reward funcs（GSM8K）：`training2.reward_correct/reward_format/tag_count_reward`
   - 组合：`plugins/training/rl/reward/modules.py` → `WeightedRewardModule`

3. **Advantage (ReMax greedy baseline)**：
   - `plugins/training/rl/advantage/estimators.py:compute_remax_advantages_by_group_id`
   - 对每个 `group_id`：`adv_i = reward_i - reward_baseline`（baseline 自身 adv=0）
   - baseline 的选择依赖 **batch 内同组样本的出现顺序** 与 `baseline_position` 一致

4. **Update / Loss (JAX)**：
   - `plugins/training/rl/remax/module.py:ReMaxPolicyGradientModule`
   - 输入 batch key（核心）：`input_ids, attention_mask, labels, advantages`
   - return（每 token）语义由 `algo.update.kwargs.returns_style` 控制：
     - `official`（默认；对齐 DeepSpeed-Chat ReMax）：
       - terminal term：把标量 `advantage` 以 `gamma**k` 折扣传播到 completion tokens（k=剩余 token 数，含当前 token）
       - KL shaping（token-local，不累积）：`-beta * (logp - logp_ref)`
     - `verl`（对齐 VERL）：
       - 把标量 outcome advantage 放到最后一个 completion token
       - token-level rewards 里做 `-beta * KL(token)`，再做 reverse-cumsum 得到 per-token returns（KL 会随时间累积）
   - actor loss：`-sum(returns * logp * completion_mask) / total_valid_token_count`

5. **Train state + ref params（KL shaping）**：
   - `plugins/training/rl/remax/state.py:get_remax_state`
   - 当 `train.beta != 0` 时：把 `ref_params` 存在 state 里，并在 `training_step` 里以 `params["ref_model"]=ref_params` 的方式喂给 loss module（ref logits stop-gradient）

6. **配置/约束（避免配置错了但还能跑）**：
   - `plugins/training/rl/algorithms/config.py` 强制 `algo.estimator.name=remax` 与 `algo.update.name=remax` 必须配套
   - `projects/gsm8k_grpo/scripts/run_train.py` 强制：
     - `rollout.backend` 必须是 `remax_mixed_naive`
     - `rollout.n >= 2`（1 baseline + >=1 sample）
     - `rollout.dynamic_sampling.enabled=false`
     - `train.ppo_epochs=1`

## Notes (semantic alignment & tradeoffs)

- **对齐官方 ReMax 折扣细节**：官方 `compute_returns()` 会在回溯时先 `cumulative_reward *= gamma`，导致最后一个 token 的 terminal term 是 `adv * gamma^1`，第一个 completion token 是 `adv * gamma^N`；本仓库在 `_discounted_terminal_returns()` 中复刻了相同的指数定义（见 `tests/test_remax_policy_gradient_module.py`）。
- **baseline 样本的计算开销**：默认实现把 greedy baseline 行也打包进训练 batch，但通过 `labels=0` 让它不反传梯度；这简化了数据管线，但会额外做一次 forward/backward（梯度为 0）。
  - 若希望更贴近 VERL（baseline 序列不参与更新），可设置 `algo.update.kwargs.drop_baseline_rows=true`（在 advantage 计算完成后，把 baseline 行从 update batch 里过滤掉）。

## References

- ReMax JAX core: `plugins/training/rl/remax/module.py`, `plugins/training/rl/remax/state.py`
- Rollout backend: `plugins/training/rl/rollout/backends/remax_mixed_naive.py`
- Advantage estimator: `plugins/training/rl/advantage/estimators.py` (`compute_remax_advantages_by_group_id`)
- Runner wiring: `projects/gsm8k_grpo/jax/train.py`, config checks `projects/gsm8k_grpo/scripts/run_train.py`
- Official reference deep-dive: `docs/sops/remax-official-implementation-deepdive.md`
