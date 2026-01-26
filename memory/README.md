# memory/

Task-scoped scratch + evidence logs (git-tracked).

## Active / recent tasks

- `memory/20260126_ce-kernel-len2048/`: TPU CE kernel gradcheck + speed/memory bench at `time=2048`.
- `memory/20260126_unsloth-liger-grpo-fused/`: Inspect Unsloth/Liger GRPO fused kernels (LM head fused?).
- `memory/20260126_unsloth-liger-fused-ce/`: Inspect Unsloth/Liger fused CE + whether LM head is fused.
- `memory/20260125_grpo-pallas-kernel/`: GRPO Pallas kernel (fwd+bwd) + TPU gradcheck vs reference.
- `memory/20260125_jax-splash-tiled-ce-kernel/`: Inspect JAX SplashAttention + design a tiled (streaming) cross-entropy/logprob kernel for LM head (no logits materialization).
- `memory/20260125_tiled-ce-pallas-kernel/`: Fix Pallas tiled CE/logprob kernel correctness + prep TPU gradcheck.
