# GSM8K GRPO configs

This folder intentionally keeps only two root configs:

- `grpo_gsm8k_qwen25_3b_batch128_roll8_literal_v6e8.yaml`
- `rl_gsm8k_qwen25_3b_batch128_roll8_literal_maxrl_v6e8.yaml`

Both are explicit full-schema configs (no hidden defaults), and both pin:

- `train.micro_batch_size_per_device: 4`
- `train.grad_accum_steps: 4`
- `train.max_length_total: 1536` (`512 + 1024`)

All other historical/experimental configs were removed as requested.
