# GSM8K GRPO configs

This directory keeps **only GRPO/MaxRL production configs** at root.

Policy:
- Root `configs/` contains only GRPO/MaxRL YAMLs.
- Non-GRPO/MaxRL experimental templates are moved to `configs/temp/`.
- Root GRPO/MaxRL YAMLs are written as **full explicit configs** (no hidden defaults).
- Root GRPO/MaxRL YAMLs pin `train.grad_accum_steps: 4`.

Schema (v3):
- `train.optimizer = {name, kwargs, lr_schedule:{name, kwargs}}`
- `algo.estimator = {name, kwargs}`
- `algo.update = {name, kwargs}`
- `rollout.dynamic_sampling = {enabled, trigger, metric, homogeneity_threshold, min_unique_reward_values, max_extra_roll_rounds, target_valid_groups, fallback_policy}`

Root files:
- `grpo_standard.yaml`
- `grpo_gsm8k_qwen25_3b_batch128_roll8_literal_v6e8.yaml`
- `grpo_gsm8k_qwen25_3b_batch128_roll8_literal_v6e8_mbpd1.yaml`
- `grpo_gsm8k_qwen25_3b_batch128_roll8_literal_v6e8_mbpd2.yaml`
- `rl_gsm8k_qwen25_3b_batch128_roll8_literal_maxrl_v6e8.yaml`
- `rl_gsm8k_qwen25_3b_batch128_roll8_literal_maxrl_v6e8_mbpd1.yaml`
- `rl_gsm8k_qwen25_3b_batch128_roll8_literal_maxrl_v6e8_mbpd2.yaml`

Temp files:
- `temp/dapo_standard.yaml`
- `temp/dapo_dynamic_sampling_acc.yaml`
- `temp/ppo_gae_standard.yaml`
