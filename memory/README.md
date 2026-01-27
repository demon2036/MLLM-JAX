# memory/

Task-scoped scratch + evidence logs (git-tracked).

## Active / recent tasks

- `memory/20260125_grpo-pallas-kernel/`: GRPO Pallas kernel (fwd+bwd) + TPU gradcheck vs reference.
- `memory/20260125_jax-splash-tiled-ce-kernel/`: Inspect JAX SplashAttention + design a tiled (streaming) cross-entropy/logprob kernel for LM head (no logits materialization).
- `memory/20260125_tiled-ce-pallas-kernel/`: Fix Pallas tiled CE/logprob kernel correctness + prep TPU gradcheck.
- `memory/20260126_grpo-pallas-kernel-multidevice/`: Make GRPO Pallas kernel shard_map multi-device compatible + run TPU 100-step train A/B with W&B.
- `memory/20260127_grpo-kernel-perf-fix/`: Make GRPO logits-level kernel faster+lower-HBM than pure JAX on TPU microbench (B=1,T=4096,V=151936).
