# SOP: Align JAX ReMax returns semantics to VERL (reverse-cumsum + KL-in-reward)

- **Title**: SOP: Enable VERL-style ReMax returns in this repo (JAX)
  **Prereqs**: Local dev env with JAX/Flax for unit tests
  **Scope**: `plugins/training/rl/remax/*`, `plugins/training/rl/algorithms/config.py`, `projects/gsm8k_grpo/jax/train.py`, `projects/gsm8k_grpo/configs/*`

## Goal

在不破坏既有“官方 ReMax（DeepSpeed-Chat）对齐语义”默认行为的前提下，提供一个 **VERL 对齐** 的可配置变体：

- token-level returns = `reverse_cumsum(token_level_rewards * mask)`
- `token_level_rewards = token_level_scores - beta * KL(token)`（KL 惩罚会随时间累积到更早 token 的 returns 里）
- baseline completion **不进入 update batch**（可选；仍用于计算 baseline advantage）

## What changed (implementation knobs)

1) `algo.update.kwargs.returns_style`

- `official`（默认）：discounted terminal returns + token-local KL shaping
- `verl`：outcome advantage 放到最后一个 completion token，随后对 token-level rewards 做 reverse-cumsum

2) `algo.update.kwargs.drop_baseline_rows`

- `false`（默认）：baseline 行仍在 update batch 里，但 `labels=0` → 不产生梯度（有额外 forward/backward 开销）
- `true`：在 advantage 计算完成后，把 baseline 行从 update batch 里过滤掉（更贴近 VERL 的“baseline 序列不参与训练”）

## Code pointers

- ReMax loss: `plugins/training/rl/remax/module.py`
- ReMax state: `plugins/training/rl/remax/state.py`
- Strict config normalization: `plugins/training/rl/algorithms/config.py`
- Runner wiring + baseline-row drop: `projects/gsm8k_grpo/jax/train.py`
- Example YAML: `projects/gsm8k_grpo/configs/remax_verlstyle_gsm8k_qwen25_3b_batch64_roll2_steps400_evalfull50_kl0p04_mb2_ckptgcs_v6e8_testmonitor.yaml`

## Steps (commands used)

Repo-local verification:

```bash
python -m pytest -q
```

Expected output: all tests pass (exit=0).

## Expected result

- `tests/test_remax_policy_gradient_module.py` 覆盖两种 returns_style：
  - `official`：折扣指数与 token-local KL shaping
  - `verl`：reverse-cumsum 使 KL shaping 对早期 token 累积
- 训练时可通过 YAML 明确启用 VERL 语义：
  - `algo.update.kwargs.returns_style: verl`
  - （可选）`algo.update.kwargs.drop_baseline_rows: true`

## Troubleshooting

- `Unsupported algo.update.kwargs keys for update 'remax'`：确认 YAML 里使用的是
  - `returns_style`（值为 `official|verl`）
  - `drop_baseline_rows`（bool）

- `remax_drop_baseline_rows=1 requires rollout batch to include 'is_baseline'`：确认 `rollout.backend: remax_mixed_naive`（该 backend 会输出 `is_baseline`）。
