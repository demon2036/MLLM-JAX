# GSM8K GRPO configs

This folder keeps explicit full-schema configs for the main TPU runs:

- `grpo_gsm8k_qwen25_3b_batch128_roll8_literal_v6e8.yaml`
- `rl_gsm8k_qwen25_3b_batch128_roll8_literal_maxrl_v6e8.yaml`
- `grpo_gsm8k_qwen25_3b_batch16_roll128_eval1full_v6e8.yaml`
- `rl_gsm8k_qwen25_3b_batch16_roll128_eval1full_maxrl_v6e8.yaml`
- `grpo_gsm8k_qwen25_3b_batch16_roll8_steps400_evalfull50_v6e8.yaml`
- `grpo_gsm8k_qwen25_3b_batch16_roll8_steps400_evalfull50_adv0entropy0p01_v6e8.yaml`
- `grpo_gsm8k_qwen25_3b_batch16_roll8_steps400_evalfull50_adv0entropy0p001_v6e8.yaml`
- `remax_gsm8k_qwen25_3b_batch16_roll8_steps400_evalfull50_kl0p04_ckptgcs_v6e8_testmonitor.yaml`
- `remax_gsm8k_qwen25_3b_batch16_roll8_steps400_evalfull50_kl0p04_mb2_ckptgcs_v6e8_testmonitor.yaml`
- `remax_gsm8k_qwen25_3b_batch64_roll2_steps400_evalfull50_kl0p04_mb2_ckptgcs_v6e8_testmonitor.yaml`

All configs are explicit (no hidden defaults). Common fixed values:

- `train.max_length_total: 1536` (`512 + 1024`)
- Most configs: `train.micro_batch_size_per_device: 4`

Notes:

- ReMax + token-local KL shaping (`train.beta > 0`) can require smaller train micro-batches on v6e-8.

For the `batch16_roll128_eval1full` pair:

- `rollout.batch_size: 16`
- `rollout.n: 128`
- `eval_rollout_n: 1`
- `eval_full_sweep: true`

These ensure evaluation is decoupled from train rollout count and runs full-split once at end.
