# MiniOneRec SID SFT configs

This folder is organized by **purpose first** (train/eval/smoke/official_eval), then by TPU type.

## Layout

```
projects/minionerec/sft/configs/
  official_eval/         # Eval official MiniOneRec checkpoints (Office/Industrial)
  smoke/                 # Small smoke configs (quick compile / sanity)
  bench/                 # Microbench configs (e.g., step time)
  train/
    v4-8/
    v6e-8/
    v6e-16/
  eval/
    v4-8/
    v6e-8/
    v6e-16/
  legacy/                # Non-JAX / historical configs kept for reference
```

## Rules

- **No hyperparam overrides in scripts/env**: if you need new settings, create a **new YAML** config (so W&B captures it).
- Keep **project-oriented** configs here (MiniOneRec-only). Reusable training components live in `plugins/`.
- Naming convention (recommendation):
  - `{task}_{backend}_{model}_{category}_{tpu}_{variant}.yaml`
  - Examples: `sid_sft_jax_qwen25_1p5b_base_industrial_v6e8_full.yaml`

