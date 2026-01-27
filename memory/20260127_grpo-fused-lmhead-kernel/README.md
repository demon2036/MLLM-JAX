# GRPO fused LM-head kernel (no logits/dlogits materialization)

## Goal

- Implement a fused GRPO loss that consumes LM hidden states + `lm_head` weights and:
  - computes per-token logp + GRPO clipped objective,
  - avoids materializing `[B,T,V]` logits (and ideally avoids `[B,T,V]` dlogits),
  - matches the repo reference numerics (within tolerances),
  - improves TPU peak HBM vs pure-JAX baseline at shape `(B=1,T=8192,V=*)`.

## Completion criteria

- Unit tests pass (local syntax + TPU `pytest`).
- TPU microbench A/B run exists (W&B `online`):
  - baseline (pure JAX logits + reference loss)
  - fused kernel enabled
- Reported `mem/peak_bytes_in_use_max` is lower for fused kernel.
- SOP updated with exact commands actually run.

## Evidence (to be filled as executed)

### Code changes

- Files:
  - `<to fill>`

### TPU runs

- Fused microbench baseline (kernel off): `<to fill>`
- Fused microbench kernel on: `<to fill>`

### Key metrics

- `mem/peak_bytes_in_use_max`:
  - baseline: `<to fill>`
  - fused: `<to fill>`

- `bench/step_s_mean` (optional):
  - baseline: `<to fill>`
  - fused: `<to fill>`

