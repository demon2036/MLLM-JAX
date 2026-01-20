from __future__ import annotations

import functools
import math
import os
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

try:
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel, splash_attention_mask
except Exception:  # pragma: no cover - depends on jaxlib build
    splash_attention_kernel = None
    splash_attention_mask = None

try:
    from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention  # type: ignore
except Exception:  # pragma: no cover - depends on jaxlib build
    flash_attention = None


AttentionBackend = Literal["auto", "dot", "splash", "flash"]


def _normalize_backend(name: str | None) -> AttentionBackend:
    raw = (name or "").strip().lower()
    if raw in {"", "auto"}:
        return "auto"
    if raw in {"dot", "dot_product", "softmax"}:
        return "dot"
    if raw in {"splash", "splash_attention"}:
        return "splash"
    if raw in {"flash", "flash_attention"}:
        return "flash"
    return "auto"


def _env_backend() -> AttentionBackend:
    return _normalize_backend(os.environ.get("MLLM_JAX_ATTENTION") or os.environ.get("MLLM_JAX_ATTENTION_IMPL"))


def _dot_attention(
    query_states: jax.Array,
    key_states: jax.Array,
    value_states: jax.Array,
    *,
    attn_bias: jax.Array | None,
    head_dim: int,
    out_dtype: Any,
) -> jax.Array:
    q = query_states.astype(jnp.float32)
    k = key_states.astype(jnp.float32)
    v = value_states.astype(jnp.float32)

    attn_weights = (q @ k.swapaxes(2, 3)) / math.sqrt(float(head_dim))
    if attn_bias is not None:
        attn_weights = attn_weights + attn_bias.astype(jnp.float32)

    attn_probs = jax.nn.softmax(attn_weights, axis=-1)
    attn_output = attn_probs @ v
    return attn_output.astype(out_dtype)


def _splash_attention(
    query_states: jax.Array,
    key_states: jax.Array,
    value_states: jax.Array,
    *,
    mesh: Any,
    head_dim: int,
    out_dtype: Any,
) -> jax.Array:
    if splash_attention_kernel is None or splash_attention_mask is None:
        raise RuntimeError("Splash attention is not available in this JAX build.")

    def _impl(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
        mask = splash_attention_mask.CausalMask(shape=(k.shape[2], k.shape[2]))
        multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(mask,) * v.shape[1])

        block_sizes = splash_attention_kernel.BlockSizes(
            block_q=min(512, q.shape[2]),
            block_kv_compute=min(512, k.shape[2]),
            block_kv=min(512, k.shape[2]),
            block_q_dkv=min(512, q.shape[2]),
            block_kv_dkv=min(512, k.shape[2]),
            block_kv_dkv_compute=min(512, q.shape[2]),
            block_q_dq=min(512, q.shape[2]),
            block_kv_dq=min(512, q.shape[2]),
        )

        splash_kernel = splash_attention_kernel.make_splash_mha(
            mask=multi_head_mask,
            head_shards=1,
            q_seq_shards=1,
            mask_value=-1e17,
            block_sizes=block_sizes,
        )
        return jax.vmap(splash_kernel)(q, k, v)

    q_scaled = (query_states / math.sqrt(float(head_dim))).astype(query_states.dtype)

    if mesh is None:
        return _impl(q_scaled, key_states, value_states).astype(out_dtype)

    @functools.partial(
        shard_map,
        mesh=mesh,
        in_specs=P(["dp", "fsdp"], "tp", None, None),
        out_specs=P(["dp", "fsdp"], "tp", None, None),
        check_rep=False,
    )
    def _sharded(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
        return _impl(q, k, v)

    return _sharded(q_scaled, key_states, value_states).astype(out_dtype)


def _flash_attention(
    query_states: jax.Array,
    key_states: jax.Array,
    value_states: jax.Array,
    *,
    mesh: Any,
    out_dtype: Any,
) -> jax.Array:
    if flash_attention is None:
        raise RuntimeError("Flash attention is not available in this JAX build.")

    if mesh is None:
        return flash_attention(query_states, key_states, value_states, causal=True).astype(out_dtype)

    @functools.partial(
        shard_map,
        mesh=mesh,
        in_specs=P(["dp", "fsdp"], "tp", None, None),
        out_specs=P(["dp", "fsdp"], "tp", None, None),
        check_rep=False,
    )
    def _sharded(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
        return flash_attention(q, k, v, causal=True)

    return _sharded(query_states, key_states, value_states).astype(out_dtype)


def apply_attention(
    query_states: jax.Array,
    key_states: jax.Array,
    value_states: jax.Array,
    *,
    attn_bias: jax.Array | None,
    head_dim: int,
    mesh: Any | None,
    backend: AttentionBackend | str = "auto",
    out_dtype: Any | None = None,
) -> jax.Array:
    """Apply attention with a swappable backend.

    Shapes:
    - query_states: [B, H, Q, D]
    - key_states:   [B, H, K, D]
    - value_states: [B, H, K, D]

    attn_bias is expected to be additive (0 for valid, negative large for masked),
    broadcastable to [B, H, Q, K].
    """
    if out_dtype is None:
        out_dtype = query_states.dtype

    backend = _normalize_backend(str(backend))
    if backend == "auto":
        backend = _env_backend()

    q_len = int(query_states.shape[2])
    kv_len = int(key_states.shape[2])
    d = int(query_states.shape[-1])

    can_use_splash = (
        splash_attention_kernel is not None
        and splash_attention_mask is not None
        and mesh is not None
        and q_len == kv_len
        and q_len % 128 == 0
        and d % 128 == 0
    )

    if backend == "splash":
        if can_use_splash:
            return _splash_attention(
                query_states,
                key_states,
                value_states,
                mesh=mesh,
                head_dim=head_dim,
                out_dtype=out_dtype,
            )
        # Fall back quietly (useful on CPU/GPU or odd shapes).
        return _dot_attention(
            query_states,
            key_states,
            value_states,
            attn_bias=attn_bias,
            head_dim=head_dim,
            out_dtype=out_dtype,
        )

    if backend == "flash":
        try:
            return _flash_attention(query_states, key_states, value_states, mesh=mesh, out_dtype=out_dtype)
        except Exception:
            return _dot_attention(
                query_states,
                key_states,
                value_states,
                attn_bias=attn_bias,
                head_dim=head_dim,
                out_dtype=out_dtype,
            )

    if backend == "dot":
        return _dot_attention(
            query_states,
            key_states,
            value_states,
            attn_bias=attn_bias,
            head_dim=head_dim,
            out_dtype=out_dtype,
        )

    # backend resolved to "auto" but env var was empty/invalid: select best effort.
    if can_use_splash:
        return _splash_attention(
            query_states,
            key_states,
            value_states,
            mesh=mesh,
            head_dim=head_dim,
            out_dtype=out_dtype,
        )
    return _dot_attention(
        query_states,
        key_states,
        value_states,
        attn_bias=attn_bias,
        head_dim=head_dim,
        out_dtype=out_dtype,
    )

