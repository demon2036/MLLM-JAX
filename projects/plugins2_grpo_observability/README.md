# plugins2 GRPO observability (TPU web demo)

This project provides a TPU-ready web backend/frontend for:

- system prompt + user prompt -> `k` sampled generations
- GSM8K-style label-based GRPO reward
- GRPO advantage + one backward/update step
- per-token observability: token id, decoded token text, probability, logprob, `dL/dlogprob`, token loss contribution

## Entrypoint

```bash
python -u projects/plugins2_grpo_observability/scripts/run_web.py \
  --config projects/plugins2_grpo_observability/configs/plugins2_grpo_observability_gsm8k_qwen25_0p5b_v4_8.yaml
```

## One-shot CLI debug

```bash
python -u projects/plugins2_grpo_observability/scripts/run_web.py \
  --config projects/plugins2_grpo_observability/configs/plugins2_grpo_observability_gsm8k_qwen25_0p5b_v4_8.yaml \
  --once \
  --user-prompt "If Alice has 3 apples and buys 5 more, how many apples does she have?" \
  --label "8" \
  --k 8
```
