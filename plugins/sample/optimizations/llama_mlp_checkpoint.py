from __future__ import annotations

from typing import Any, Callable


def patch_llama_mlp_checkpoint_gate_up() -> None:
    """Checkpoint the MLP gate/up activation to reduce training HBM.

    Motivation (v6e memory cliff at long seq / larger micro-batch):
    - The GRPO mb4/dev compile OOM is dominated by large per-layer MLP intermediates
      (e.g. bf16[mb, T, intermediate] ~ 215MB each for Qwen2.5-3B at T=2560).
    - We already fused the GRPO loss to remove the full-vocab entropy softmax in the
      legacy JAX path; remaining OOM margin is small and often driven by HLO-temp
      fragmentation.
    - This patch adds an extra checkpoint boundary around the SwiGLU gate/up
      activation (`silu(gate_proj(x)) * up_proj(x)`), allowing XLA to recompute it
      in backward instead of keeping it live.

    Notes:
    - This is opt-in via the GRPO runner config.
    - Safe to call multiple times.
    """

    from MLLM_JAX.language.llama import llama as llama_mod

    if getattr(llama_mod, "_mlp_checkpoint_gate_up_patched", False):
        return

    original_call: Callable[..., Any] = llama_mod.LlamaMLP.__call__

    def call_with_checkpoint(self, x):
        import flax.linen as nn
        import jax

        def gate_up(y):
            return nn.silu(self.gate_proj(y)) * self.up_proj(y)

        gate_up_ckpt = jax.checkpoint(gate_up, policy=jax.checkpoint_policies.nothing_saveable)
        x = gate_up_ckpt(x)

        if getattr(self, "jax_config", None) is not None:
            # Keep the same hook point as the original implementation.
            pass

        return self.down_proj(x)

    llama_mod.LlamaMLP.__call__ = call_with_checkpoint
    llama_mod._mlp_checkpoint_gate_up_patched = True
    llama_mod._mlp_checkpoint_gate_up_original_call = original_call


__all__ = ["patch_llama_mlp_checkpoint_gate_up"]

