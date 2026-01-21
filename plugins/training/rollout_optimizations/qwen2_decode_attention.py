from __future__ import annotations

from typing import Any


def patch_qwen2_attention_decode_fast() -> None:
    """Patch attention fallback path to avoid float32 matmuls during decode.

    After the attention refactor (see `MLLM_JAX/language/attention.py`),
    `LlamaAttention.__call__` (and thus Qwen2 attention) routes decode (q_len=1)
    through `_naive_sdpa`, which does float32 matmuls. On TPU this is expensive
    and dominates rollout time. This patch keeps the matmuls in model dtype
    (bf16) for decode, but keeps the softmax in float32 for stability.

    This is opt-in via env in the runner and is safe to call multiple times.
    """

    # Import lazily so local pytest does not require JAX/TPU deps.
    from MLLM_JAX.language import attention as attention_mod

    if getattr(attention_mod, "_fast_decode_attention_patched", False):
        return

    original_naive_sdpa = attention_mod._naive_sdpa

    def naive_sdpa_fast(
        query_states: Any,
        key_states: Any,
        value_states: Any,
        *,
        head_dim: int,
        attn_mask: Any,
        dtype: Any,
    ):
        q_len = int(query_states.shape[2])
        if q_len != 1:
            return original_naive_sdpa(
                query_states,
                key_states,
                value_states,
                head_dim=head_dim,
                attn_mask=attn_mask,
                dtype=dtype,
            )

        # Key change vs baseline: keep QK^T and (softmax @ V) matmuls in model dtype.
        attn_weights = query_states @ key_states.swapaxes(2, 3)
        scale = attention_mod.jnp.asarray(
            1.0 / attention_mod.math.sqrt(head_dim),
            dtype=attention_mod.jnp.float32,
        )
        attn_weights = attn_weights.astype(attention_mod.jnp.float32) * scale

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask.astype(attention_mod.jnp.float32)

        attn_probs = attention_mod.jax.nn.softmax(attn_weights, axis=-1).astype(attention_mod.jnp.float32)
        attn_output = (attn_probs.astype(dtype) @ value_states).astype(dtype)
        return attn_output

    attention_mod._naive_sdpa = naive_sdpa_fast
    attention_mod._fast_decode_attention_patched = True
    attention_mod._fast_decode_attention_original_naive_sdpa = original_naive_sdpa


__all__ = ["patch_qwen2_attention_decode_fast"]
