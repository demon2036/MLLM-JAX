# GRPO/GSM8K: adv==0 conditional entropy regularization

- **Title**: SOP: Enable entropy regularization only for `adv≈0` samples in GRPO/GSM8K
  **Prereqs**: None
  **Scope**: `plugins/training/rl/algorithms/config.py`, `training2.py`, `MLLM_JAX/train_modules/__init__.py`, `projects/gsm8k_grpo/jax/train.py`

## Motivation

In GRPO, advantages are normalized within each prompt-group. When a group’s rewards are homogeneous (`std==0`), the computed advantages become exactly `0`, so the policy-gradient term becomes `0` and those samples stop contributing updates.

This knob adds **entropy regularization only for that `adv≈0` subset**, encouraging exploration without changing behavior for `adv≠0` samples.

## How to enable (YAML)

Use `algo.update.name: policy_gradient` and set:

- `algo.update.kwargs.adv_zero_entropy_coef` (float, default `0.0`)
  - `0.0` disables the feature (baseline behavior).
  - `>0` enables the entropy bonus for `adv≈0` samples only.
- `algo.update.kwargs.adv_zero_epsilon` (float, default `0.0`)
  - `adv` is treated as “zero” if `abs(adv) <= adv_zero_epsilon`.

Example:

```yaml
algo:
  update:
    name: policy_gradient
    kwargs:
      adv_zero_entropy_coef: 0.01
      adv_zero_epsilon: 0.0
```

## What it does (loss-level)

Inside `TrainGRPOModule`:

- compute per-token entropy on completion tokens
- build a mask for sequences with `adv≈0`
- add an entropy term **only on those tokens**, scaled by `adv_zero_entropy_coef`

## Logged signals (W&B)

### Advantage sign distribution

Logged from the GRPO runner:

- `train-reward/advantage/sign/zero_fraction`
- `train-reward/advantage/sign/pos_fraction`
- `train-reward/advantage/sign/neg_fraction`

(`*_count` variants are also logged.)

### Adv-zero entropy regularizer metrics

Logged from the training module (via the runner):

- `train-loss/entropy_adv0_reg` (the loss contribution; negative when enabled)
- `train-adv_zero/seq_fraction`
- `train-adv_zero/token_fraction`
- `train-adv_zero/entropy_per_valid_token`
- `train-adv_zero/entropy_mean_per_adv0_token`

## References

- `plugins/training/rl/advantage/grpo.py` (why homogeneous groups yield `adv==0`)
- `MLLM_JAX/train_modules/__init__.py` (`TrainGRPOModule` entropy term)
- `projects/gsm8k_grpo/jax/train.py` (logging + periodic full eval)

