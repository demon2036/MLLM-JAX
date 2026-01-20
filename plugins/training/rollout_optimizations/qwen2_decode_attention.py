from __future__ import annotations

from typing import Any


def patch_qwen2_attention_decode_fast() -> None:
    """Patch Qwen2Attention fallback path to avoid float32 matmuls during decode.

    The baseline Qwen2 attention implementation casts Q,K,V to float32 for the
    non-splash path (used by decode where q_len=1). On TPU this is expensive and
    dominates rollout time. This patch keeps the matmuls in model dtype (bf16),
    but keeps the softmax in float32 for stability.

    This is opt-in via env in the runner and is safe to call multiple times.
    """

    # Import lazily so local pytest does not require JAX/TPU deps.
    from MLLM_JAX.language.qwen2 import modular_qwen2

    Qwen2Attention = modular_qwen2.Qwen2Attention
    if getattr(Qwen2Attention, "_fast_decode_attention_patched", False):
        return

    original_call = Qwen2Attention.__call__

    def call_fast(
        self,
        x: Any,
        input_ids: Any,
        cache: Any,
        attn_mask: Any,
        position_embeddings: Any,
    ):
        bsz, q_len, _ = x.shape

        query_states = self.q_proj(x)
        key_states = self.k_proj(x)
        value_states = self.v_proj(x)

        query_states = modular_qwen2.einops.rearrange(query_states, "b n (h d)->b h n  d ", d=self.head_dim)
        key_states = modular_qwen2.einops.rearrange(key_states, "b n (h d)->b h n  d ", d=self.head_dim)
        value_states = modular_qwen2.einops.rearrange(value_states, "b n (h d)->b h n  d ", d=self.head_dim)

        dtype = x.dtype
        cos, sin = position_embeddings
        query_states = query_states.astype(modular_qwen2.jnp.float32)
        key_states = key_states.astype(dtype)
        query_states, key_states = modular_qwen2.apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states = query_states.astype(dtype)
        key_states = key_states.astype(dtype)

        if cache is not None:
            end_index = cache["end_index"][0]
            slice_indices = (0, 0, end_index, 0)
            value_states = modular_qwen2.jax.lax.dynamic_update_slice(
                cache["v"],
                value_states.astype(cache["v"].dtype),
                slice_indices,
            )
            key_states = modular_qwen2.jax.lax.dynamic_update_slice(
                cache["k"], key_states.astype(cache["k"].dtype), slice_indices
            )
            new_cache = {
                "v": value_states,
                "k": key_states,
                "end_index": cache["end_index"] + q_len,
            }
            value_states = value_states.astype(dtype)
            key_states = key_states.astype(dtype)
        else:
            new_cache = None

        value_states = modular_qwen2.repeat_kv(value_states, self.num_key_value_groups)
        key_states = modular_qwen2.repeat_kv(key_states, self.num_key_value_groups)

        if q_len % 128 == 0 and value_states.shape[-1] % 128 == 0:

            @modular_qwen2.functools.partial(
                modular_qwen2.shard_map,
                mesh=self.jax_config.mesh,
                in_specs=modular_qwen2.P(["dp", "fsdp"], "tp", None, None),
                out_specs=modular_qwen2.P(["dp", "fsdp"], "tp", None, None),
                check_rep=False,
            )
            def wrap_splash_attention(query_states, key_states, value_states):
                mask = modular_qwen2.splash_attention_mask.CausalMask(shape=(key_states.shape[2], key_states.shape[2]))
                multi_head_mask = modular_qwen2.splash_attention_mask.MultiHeadMask(
                    masks=(mask,) * value_states.shape[1]
                )

                block_sizes = modular_qwen2.splash_attention_kernel.BlockSizes(
                    block_q=min(512, query_states.shape[2]),
                    block_kv_compute=min(512, key_states.shape[2]),
                    block_kv=min(512, key_states.shape[2]),
                    block_q_dkv=min(512, query_states.shape[2]),
                    block_kv_dkv=min(512, key_states.shape[2]),
                    block_kv_dkv_compute=min(512, query_states.shape[2]),
                    block_q_dq=min(512, query_states.shape[2]),
                    block_kv_dq=min(512, query_states.shape[2]),
                )
                del block_sizes

                splash_kernel = modular_qwen2.splash_attention_kernel.make_splash_mha(
                    mask=multi_head_mask,
                    head_shards=1,
                    q_seq_shards=1,
                    # block_sizes=block_sizes,
                )

                attn_output = modular_qwen2.jax.vmap(splash_kernel)(query_states, key_states, value_states)
                return attn_output

            attn_output = wrap_splash_attention(
                query_states / modular_qwen2.math.sqrt(self.head_dim), key_states, value_states
            ).astype(modular_qwen2.jnp.bfloat16)
        else:
            # Key change vs baseline: keep QK^T and (softmax @ V) matmuls in model dtype.
            attn_weights = (query_states @ key_states.swapaxes(2, 3)) / modular_qwen2.math.sqrt(self.head_dim)
            attn_weights = attn_weights.astype(modular_qwen2.jnp.float32)
            if attn_mask is not None:
                attn_weights = attn_weights + attn_mask

            attn_weights = modular_qwen2.nn.softmax(attn_weights, axis=-1)
            attn_output = (attn_weights.astype(dtype) @ value_states).astype(dtype)

        attn_output = modular_qwen2.einops.rearrange(attn_output, "b h n d-> b n (h d)")
        attn_output = self.o_proj(attn_output)
        return new_cache, attn_output.astype(dtype)

    Qwen2Attention.__call__ = call_fast
    Qwen2Attention._fast_decode_attention_patched = True
    Qwen2Attention._fast_decode_attention_original_call = original_call


__all__ = ["patch_qwen2_attention_decode_fast"]

