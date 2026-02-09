# GSM8K GRPO configs

These YAML files are GSM8K/GRPO project configs.

They are tracked in Git so W&B runs can be reproduced by referencing an exact committed config path.

- `grpo_gsm8k_qwen25_3b_batch128_roll8_literal_v6e8.yaml`: GRPO literal profile (`rollout.batch_size=128`, `rollout.n=8` => 1024 seq/step).
- `rl_gsm8k_qwen25_3b_batch128_roll8_literal_maxrl_v6e8.yaml`: MaxRL literal profile with the same batch semantics (`128x8`, 1024 seq/step).
