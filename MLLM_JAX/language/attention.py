from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

try:
    from jax.experimental.shard_map import shard_map
except Exception:  # pragma: no cover - older JAX
    shard_map = None  # type: ignore[assignment]

try:
    from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention as tpu_flash_attention
except Exception:  # pragma: no cover - not available on all JAX builds
    tpu_flash_attention = None  # type: ignore[assignment]

try:
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel, splash_attention_mask
except Exception:  # pragma: no cover - not available on all JAX builds
    splash_attention_kernel = None  # type: ignore[assignment]
    splash_attention_mask = None  # type: ignore[assignment]

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)

AttentionBackend = Literal["auto", "splash", "flash", "dot"]


@dataclass(frozen=True)
class AttentionSpec:
    """Sharding/spec knobs for attention kernels.

    Shapes are assumed to be `[B, H, S, D]` for Q/K/V.
    """

    batch_axes: tuple[str, ...] = ("dp", "fsdp")
    head_axis: str | None = "tp"
    q_seq_axis: str | None = None


def _dedupe_keep_order(items: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out)


def _normalize_spec(spec: AttentionSpec) -> AttentionSpec:
    batch_axes = _dedupe_keep_order(tuple(spec.batch_axes))
    head_axis = spec.head_axis
    q_seq_axis = spec.q_seq_axis
    if head_axis is not None and q_seq_axis is not None and head_axis == q_seq_axis:
        raise ValueError(f"Invalid AttentionSpec: head_axis == q_seq_axis == {head_axis!r}")
    if q_seq_axis is not None and q_seq_axis in batch_axes:
        batch_axes = tuple(ax for ax in batch_axes if ax != q_seq_axis)
    if head_axis is not None and head_axis in batch_axes:
        batch_axes = tuple(ax for ax in batch_axes if ax != head_axis)
    return AttentionSpec(batch_axes=batch_axes, head_axis=head_axis, q_seq_axis=q_seq_axis)


def _parse_env_bool(value: str) -> bool:
    value = value.strip().lower()
    if value in ("1", "true", "t", "yes", "y", "on"):
        return True
    if value in ("0", "false", "f", "no", "n", "off"):
        return False
    raise ValueError(f"Invalid bool env value: {value!r}")


def _parse_env_axes(value: str) -> tuple[str, ...]:
    parts = [p.strip() for p in value.replace("|", ",").split(",")]
    return tuple(p for p in parts if p)


def _parse_env_axis(value: str) -> str | None:
    value = value.strip()
    if value == "":
        return None
    lowered = value.lower()
    if lowered in ("none", "null"):
        return None
    return value


def _normalize_backend(value: str) -> AttentionBackend:
    raw = value.strip().lower()
    if raw == "":
        return "auto"
    if raw in ("auto", "splash", "flash", "dot"):
        return raw  # type: ignore[return-value]
    if raw in ("splash_attention",):
        return "splash"
    if raw in ("flash_attention",):
        return "flash"
    if raw in ("dot_product", "softmax"):
        return "dot"
    raise ValueError(f"Unknown attention backend: {value!r}")


def _maybe_override_backend(backend: AttentionBackend) -> AttentionBackend:
    env = (
        os.environ.get("MLLM_JAX_ATTENTION_BACKEND")
        or os.environ.get("MLLM_JAX_ATTENTION")
        or os.environ.get("MLLM_JAX_ATTENTION_IMPL")
    )
    if env:
        return _normalize_backend(env)
    return backend


def _maybe_override_spec(spec: AttentionSpec) -> AttentionSpec:
    batch_axes = os.environ.get("MLLM_JAX_ATTENTION_BATCH_AXES")
    head_axis = os.environ.get("MLLM_JAX_ATTENTION_HEAD_AXIS")
    q_seq_axis = os.environ.get("MLLM_JAX_ATTENTION_Q_SEQ_AXIS")
    if batch_axes is None and head_axis is None and q_seq_axis is None:
        return spec
    return AttentionSpec(
        batch_axes=_parse_env_axes(batch_axes) if batch_axes is not None else spec.batch_axes,
        head_axis=_parse_env_axis(head_axis) if head_axis is not None else spec.head_axis,
        q_seq_axis=_parse_env_axis(q_seq_axis) if q_seq_axis is not None else spec.q_seq_axis,
    )


def _naive_sdpa(
    query_states: jax.Array,
    key_states: jax.Array,
    value_states: jax.Array,
    *,
    head_dim: int,
    attn_mask: jax.Array | None,
    dtype: Any,
) -> jax.Array:
    attn_weights = (
        query_states.astype(jnp.float32) @ key_states.swapaxes(2, 3).astype(jnp.float32)
    ) / math.sqrt(head_dim)
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.astype(jnp.float32)
    attn_probs = jax.nn.softmax(attn_weights, axis=-1).astype(jnp.float32)
    attn_output = attn_probs @ value_states.astype(jnp.float32)
    return attn_output.astype(dtype)


def _maybe_pad_batch_for_shard_map(
    query_states: jax.Array,
    key_states: jax.Array,
    value_states: jax.Array,
    *,
    mesh: Any,
    spec: AttentionSpec,
) -> tuple[jax.Array, jax.Array, jax.Array, int]:
    """Pad batch dim so shard_map can shard it across spec.batch_axes."""
    spec = _normalize_spec(spec)
    batch_shards = 1
    for axis in spec.batch_axes:
        batch_shards *= int(mesh.shape[axis])
    if batch_shards <= 1:
        return query_states, key_states, value_states, 0

    batch = int(query_states.shape[0])
    remainder = batch % batch_shards
    if remainder == 0:
        return query_states, key_states, value_states, 0

    pad = batch_shards - remainder
    pad_cfg = ((0, pad), (0, 0), (0, 0), (0, 0))
    query_states = jnp.pad(query_states, pad_cfg)
    key_states = jnp.pad(key_states, pad_cfg)
    value_states = jnp.pad(value_states, pad_cfg)
    return query_states, key_states, value_states, pad


def _tpu_flash_sdpa(
    query_states: jax.Array,
    key_states: jax.Array,
    value_states: jax.Array,
    *,
    mesh: Any,
    spec: AttentionSpec,
    causal: bool,
    sm_scale: float,
) -> jax.Array:
    if tpu_flash_attention is None:
        raise RuntimeError("TPU flash_attention backend is not available in this JAX build.")
    if shard_map is None:
        raise RuntimeError("jax.experimental.shard_map is not available in this JAX build.")

    spec = _normalize_spec(spec)
    if spec.q_seq_axis is not None:
        raise ValueError("flash_attention path does not support sharded q_seq_axis yet (needs causal offset handling).")

    batch_spec: Any = spec.batch_axes
    head_spec: Any = spec.head_axis
    in_specs = P(batch_spec, head_spec, None, None)
    out_specs = P(batch_spec, head_spec, None, None)

    @functools.partial(
        shard_map,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_rep=False,
    )
    def _wrap(q, k, v):
        return tpu_flash_attention(q, k, v, causal=causal, sm_scale=sm_scale)

    return _wrap(query_states, key_states, value_states)


def _tpu_splash_sdpa(
    query_states: jax.Array,
    key_states: jax.Array,
    value_states: jax.Array,
    *,
    mesh: Any,
    spec: AttentionSpec,
    causal: bool,
    mask_value: float,
    block_size: int,
    use_block_sizes: bool,
) -> jax.Array:
    if splash_attention_kernel is None or splash_attention_mask is None:
        raise RuntimeError("TPU splash_attention backend is not available in this JAX build.")
    if shard_map is None:
        raise RuntimeError("jax.experimental.shard_map is not available in this JAX build.")
    if not causal:
        raise ValueError("splash_attention path currently supports only causal=True for this repo.")

    spec = _normalize_spec(spec)

    q_len = int(query_states.shape[2])
    kv_len = int(key_states.shape[2])
    num_heads = int(value_states.shape[1])

    head_shards = int(mesh.shape[spec.head_axis]) if spec.head_axis is not None else 1
    q_seq_shards = int(mesh.shape[spec.q_seq_axis]) if spec.q_seq_axis is not None else 1

    if q_seq_shards > 1 and q_len % (q_seq_shards * q_seq_shards) != 0:
        raise ValueError(
            "splash_attention requires q_len divisible by (q_seq_shards^2): "
            f"{q_len=} {q_seq_shards=}"
        )

    mask = splash_attention_mask.CausalMask(shape=(q_len, kv_len), shard_count=q_seq_shards)
    multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(mask,) * num_heads)

    block_q = min(block_size, q_len)
    block_kv = min(block_size, kv_len)
    block_sizes = splash_attention_kernel.BlockSizes(
        block_q=block_q,
        block_kv_compute=block_kv,
        block_kv=block_kv,
        block_q_dkv=block_q,
        block_kv_dkv=block_kv,
        block_kv_dkv_compute=block_kv,
        block_q_dq=block_q,
        block_kv_dq=block_q,
    )

    if use_block_sizes:
        splash_kernel = splash_attention_kernel.make_splash_mha(
            mask=multi_head_mask,
            head_shards=head_shards,
            q_seq_shards=q_seq_shards,
            mask_value=mask_value,
            block_sizes=block_sizes,
            residual_checkpoint_name="context",
        )
    else:
        splash_kernel = splash_attention_kernel.make_splash_mha(
            mask=multi_head_mask,
            head_shards=head_shards,
            q_seq_shards=q_seq_shards,
            residual_checkpoint_name="context",
        )

    kernel_axis_names = P(spec.head_axis, spec.q_seq_axis)
    kernel_in_specs = splash_kernel.manual_sharding_spec(NamedSharding(mesh, kernel_axis_names))

    batch_spec: Any = spec.batch_axes
    head_spec: Any = spec.head_axis
    q_seq_spec: Any = spec.q_seq_axis
    q_in_specs = P(batch_spec, head_spec, q_seq_spec, None)
    kv_in_specs = P(batch_spec, head_spec, None, None)
    out_specs = q_in_specs

    @functools.partial(
        shard_map,
        mesh=mesh,
        in_specs=(q_in_specs, kv_in_specs, kv_in_specs, kernel_in_specs),
        out_specs=out_specs,
        check_rep=False,
    )
    def _wrap(q, k, v, kernel):
        return jax.vmap(kernel)(q, k, v)

    return _wrap(query_states, key_states, value_states, splash_kernel)


def scaled_dot_product_attention(
    query_states: jax.Array,
    key_states: jax.Array,
    value_states: jax.Array,
    *,
    head_dim: int,
    attn_mask: jax.Array | None,
    dtype: Any,
    mesh: Any | None,
    backend: AttentionBackend = "auto",
    spec: AttentionSpec = AttentionSpec(),
    causal: bool = True,
    mask_value: float = DEFAULT_MASK_VALUE,
    block_size: int = 512,
    use_block_sizes: bool = False,
) -> jax.Array:
    """Scaled dot-product attention with optional TPU fused kernels."""
    backend = _maybe_override_backend(backend)
    spec = _maybe_override_spec(spec)

    env_block_size = os.environ.get("MLLM_JAX_ATTENTION_BLOCK_SIZE")
    if env_block_size is not None:
        block_size = int(env_block_size)
    env_use_block_sizes = os.environ.get("MLLM_JAX_ATTENTION_USE_BLOCK_SIZES")
    if env_use_block_sizes is not None:
        use_block_sizes = _parse_env_bool(env_use_block_sizes)

    if backend not in ("auto", "splash", "flash", "dot"):
        raise ValueError(f"Unknown attention backend: {backend}")

    q_len = int(query_states.shape[2])
    head_dim_runtime = int(value_states.shape[-1])
    if int(head_dim) != head_dim_runtime:
        raise ValueError(f"head_dim={head_dim} does not match value_states.shape[-1]={head_dim_runtime}")
    head_dim_ok = head_dim_runtime <= 128 or (head_dim_runtime % 128 == 0)
    use_fused_shape_ok = (q_len % 128 == 0) and head_dim_ok

    sm_scale = float(1.0 / math.sqrt(head_dim))

    if backend in ("auto", "splash") and mesh is not None and use_fused_shape_ok:
        query_states_pad, key_states_pad, value_states_pad, batch_pad = _maybe_pad_batch_for_shard_map(
            query_states,
            key_states,
            value_states,
            mesh=mesh,
            spec=spec,
        )
        scale = jnp.asarray(sm_scale, dtype=jnp.float32)
        query_scaled = (query_states_pad * scale).astype(query_states_pad.dtype)
        try:
            out = _tpu_splash_sdpa(
                query_scaled,
                key_states_pad,
                value_states_pad,
                mesh=mesh,
                spec=spec,
                causal=causal,
                mask_value=mask_value,
                block_size=block_size,
                use_block_sizes=use_block_sizes,
            ).astype(dtype)
            if batch_pad:
                out = out[: int(query_states.shape[0])]
            return out
        except Exception:
            if backend == "splash":
                raise

    spec_norm = _normalize_spec(spec)
    if spec_norm.q_seq_axis is not None:
        if backend == "flash":
            raise ValueError("flash_attention backend does not support q_seq sharding; use backend='splash' instead.")
        if backend == "auto":
            return _naive_sdpa(
                query_states,
                key_states,
                value_states,
                head_dim=head_dim,
                attn_mask=attn_mask,
                dtype=dtype,
            )

    if backend in ("auto", "flash") and mesh is not None and use_fused_shape_ok and tpu_flash_attention is not None:
        query_states_pad, key_states_pad, value_states_pad, batch_pad = _maybe_pad_batch_for_shard_map(
            query_states,
            key_states,
            value_states,
            mesh=mesh,
            spec=spec_norm,
        )
        try:
            out = _tpu_flash_sdpa(
                query_states_pad,
                key_states_pad,
                value_states_pad,
                mesh=mesh,
                spec=spec_norm,
                causal=causal,
                sm_scale=sm_scale,
            ).astype(dtype)
            if batch_pad:
                out = out[: int(query_states.shape[0])]
            return out
        except Exception:
            if backend == "flash":
                raise

    return _naive_sdpa(
        query_states,
        key_states,
        value_states,
        head_dim=head_dim,
        attn_mask=attn_mask,
        dtype=dtype,
    )


def apply_attention(
    query_states: jax.Array,
    key_states: jax.Array,
    value_states: jax.Array,
    *,
    head_dim: int,
    attn_bias: jax.Array | None,
    out_dtype: Any,
    mesh: Any | None,
    backend: AttentionBackend | str = "auto",
) -> jax.Array:
    """Compatibility wrapper for older callsites."""
    return scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        head_dim=head_dim,
        attn_mask=attn_bias,
        dtype=out_dtype,
        mesh=mesh,
        backend=_normalize_backend(str(backend)),
        spec=AttentionSpec(),
        causal=True,
    )


__all__ = [
    "AttentionBackend",
    "AttentionSpec",
    "DEFAULT_MASK_VALUE",
    "apply_attention",
    "scaled_dot_product_attention",
]
