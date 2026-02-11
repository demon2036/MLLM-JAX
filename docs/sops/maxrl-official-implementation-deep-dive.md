# SOP: Deep dive official `tajwarfahim/maxrl` implementation

- **Title**: SOP: 追踪官方 MaxRL 实现链路（配置 -> trainer -> advantage -> actor loss -> reward）
- **Prereqs**: 可联网；安装 `git`；本仓库可读写
- **Environment (verified)**:
  - Date: 2026-02-11
  - Local repo: `MLLM-JAX-max-rl`
  - Upstream repo: `https://github.com/tajwarfahim/maxrl`
  - Upstream HEAD: `91118ea594d69be295d44437566f4aff86ceaae7` (2026-02-05)

## Steps (commands actually used)

1) 校验 upstream 最新 HEAD 并拉取

```bash
git ls-remote https://github.com/tajwarfahim/maxrl HEAD
git clone --depth 1 https://github.com/tajwarfahim/maxrl workdir/maxrl_official
git -C workdir/maxrl_official show -s --date=iso --format='%H%n%ad%n%s' HEAD
```

2) 定位 MaxRL estimator 的注册和实现

```bash
rg -n "AdvantageEstimator\.MAXRL|compute_maxrl_outcome_advantage|register_adv_est" workdir/maxrl_official/verl/trainer/ppo/core_algos.py
```

3) 定位 trainer 如何调用 MaxRL

```bash
rg -n "compute_advantage\(|adv_estimator in \[AdvantageEstimator\.MAXRL|use_critic" workdir/maxrl_official/verl/trainer/ppo/ray_trainer.py
```

4) 定位 policy update 目标（是否仍是 PPO 风格裁剪目标）

```bash
rg -n "compute_grpo_loss_objective|compute_policy_loss|ratio|cliprange" workdir/maxrl_official/verl/trainer/ppo/core_algos.py
```

5) 定位 reward 写入 token-level 的方式

```bash
rg -n "reward_tensor\[i, valid_resp_len - 1\]" workdir/maxrl_official/verl/workers/reward_manager/naive.py
```

6) 定位官方实验脚本如何启用 MaxRL

```bash
rg -n "ADVANTAGE_ESTIMATOR=maxrl|algorithm\.adv_estimator" \
  workdir/maxrl_official/qwen3_experiments/run_qwen3_training.sh \
  workdir/maxrl_official/smollm/smollm.sh \
  workdir/maxrl_official/maze/maze_17.sh
```

## Key Findings

- MaxRL 在官方代码中是一个 **advantage estimator 选项**，而不是单独的 trainer：
  - `AdvantageEstimator.MAXRL = "maxrl"`
  - `@register_adv_est(AdvantageEstimator.MAXRL)`
- MaxRL 核心公式（按 prompt-group）：
  - `adv = (score - group_mean) / (group_mean + epsilon)`
  - group size=1 时，`group_mean=0`
- 训练仍使用 PPO/GRPO 风格的策略更新目标：
  - 使用 `ratio = exp(log_prob - old_log_prob)`
  - 对 ratio 进行 clipped surrogate（含 dual-clip 变体）
- 对于 MaxRL，官方 trainer 默认 **不使用 critic**（`use_critic=False`），即不走 GAE/value-head 分支。
- reward manager 输出的 reward 被写入响应最后一个有效 token；然后 `token_level_rewards.sum(dim=-1)` 变成每条响应的 `score` 参与 MaxRL advantage。
- 官方提供的多套脚本（Qwen3/SmolLM/Maze）均通过 `algorithm.adv_estimator=maxrl` 切换，且常见配置是 `use_kl_in_reward=False`、`actor.use_kl_loss=False`。

## Expected Result

- 你能在 upstream 中复现完整调用链：
  - 配置设置 `adv_estimator=maxrl`
  - trainer 在 `compute_advantage()` 分发到 `compute_maxrl_outcome_advantage()`
  - actor update 依旧是 PPO clipped policy loss，只是 advantage 来自 MaxRL 公式
- 你能解释：MaxRL 与 GRPO 在该代码库中的关键差异位于 advantage 归一化分母（`mean` vs `std`）。

## Notes / Caveats

- 当前 HEAD 中 `compute_maxrl_outcome_advantage()` 里 `id2std` 与 `N` 未参与最终公式（保留变量）。
- `ray_trainer.py` 中存在 `AdvantageEstimator.P_NORMALIZATION_WITH_CLIPPING` 分支引用；该枚举成员不在当前 `AdvantageEstimator` 定义里，属于潜在不一致代码路径。

## References

- `workdir/maxrl_official/README.md`
- `workdir/maxrl_official/verl/trainer/ppo/core_algos.py`
- `workdir/maxrl_official/verl/trainer/ppo/ray_trainer.py`
- `workdir/maxrl_official/verl/trainer/config/ppo_trainer.yaml`
- `workdir/maxrl_official/verl/workers/reward_manager/naive.py`
- `workdir/maxrl_official/qwen3_experiments/run_qwen3_training.sh`
- `workdir/maxrl_official/smollm/smollm.sh`
- `workdir/maxrl_official/maze/maze_17.sh`
