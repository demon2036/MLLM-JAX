"""Qwen2/Qwen2.5 parameter adaptation helpers (MLLM-JAX -> sglang-jax).

Keep this integration logic under `plugins/` to avoid modifying upstream
`sglang-jax` while enabling weight hot-swap experiments.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def lm_head_kernel_to_embedding(kernel):
    """Convert MLLM-JAX Dense kernel -> sglang-jax embedding layout.

    - MLLM-JAX uses `nn.Dense(vocab)` so params store: `lm_head.kernel` with
      shape [hidden_size, vocab_size].
    - sglang-jax uses `ParallelLMHead` with: `lm_head.embedding` shape
      [vocab_size, hidden_size].
    """

    import numpy as np

    if kernel is None:
        raise ValueError("kernel is None")
    if getattr(kernel, "ndim", None) != 2:
        raise ValueError(f"Expected 2D kernel, got ndim={getattr(kernel, 'ndim', None)}")
    return np.transpose(kernel, (1, 0))


def kv_head_expand(
    arr,
    *,
    head_dim: int,
    original_kv_heads: int,
    target_kv_heads: int,
    axis: int,
    strategy: str,
):
    """Expand KV heads to match sglang-jax tensor-parallel KV replication.

    This mirrors the behavior in `sglang-jax` (see `WeightLoader._apply_kv_head_padding`)
    with two supported strategies:
    - "replicate": repeat each original KV head `n` times (GQA semantics).
    - "zero": pad zeros to the target size (MHA semantics).
    """

    import numpy as np

    if head_dim <= 0:
        raise ValueError(f"head_dim must be > 0, got {head_dim}")
    if original_kv_heads <= 0:
        raise ValueError(f"original_kv_heads must be > 0, got {original_kv_heads}")
    if target_kv_heads <= 0:
        raise ValueError(f"target_kv_heads must be > 0, got {target_kv_heads}")
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    if original_kv_heads == target_kv_heads:
        return arr

    if strategy not in ("replicate", "zero"):
        raise ValueError(f"Unsupported strategy={strategy!r} (expected replicate/zero).")

    expected = original_kv_heads * head_dim
    got = int(arr.shape[axis])
    if got != expected:
        raise ValueError(
            f"KV tensor dim mismatch (axis={axis}): expected {expected}, got {got}. "
            f"(original_kv_heads={original_kv_heads}, head_dim={head_dim})"
        )

    target = target_kv_heads * head_dim

    if strategy == "zero":
        if got == target:
            return arr
        if got > target:
            slicer = [slice(None)] * arr.ndim
            slicer[axis] = slice(0, target)
            return arr[tuple(slicer)]
        pad = target - got
        pad_width = [(0, 0)] * arr.ndim
        pad_width[axis] = (0, pad)
        return np.pad(arr, pad_width, mode="constant", constant_values=0)

    # strategy == "replicate"
    # If target_kv_heads isn't divisible (rare), replicate and slice.
    if target_kv_heads % original_kv_heads == 0:
        num_replicas = target_kv_heads // original_kv_heads
    else:
        num_replicas = (target_kv_heads + original_kv_heads - 1) // original_kv_heads

    parts = []
    for original_head_id in range(original_kv_heads):
        start = original_head_id * head_dim
        end = (original_head_id + 1) * head_dim
        slicer = [slice(None)] * arr.ndim
        slicer[axis] = slice(start, end)
        one = arr[tuple(slicer)]
        for _ in range(num_replicas):
            parts.append(one)

    expanded = np.concatenate(parts, axis=axis)
    if int(expanded.shape[axis]) == target:
        return expanded

    # Trim to exact size required by the configured model.
    slicer = [slice(None)] * expanded.ndim
    slicer[axis] = slice(0, target)
    return expanded[tuple(slicer)]


def build_sglang_qwen2_param_dict_from_mllm_params(
    *,
    mllm_params,
    model_config,
    mesh,
    param_dtype: str = "bfloat16",
) -> dict[str, "jax.Array"]:
    """Build an in-memory param dict for sglang-jax Qwen2/Qwen2.5 models.

    Returns a mapping compatible with `swap_engine_weights_from_param_dict`:
      nnx_target_path -> sharded jax.Array

    Notes:
    - This function assumes `model_config.configure_for_tensor_parallel(tp)` has
      already been applied (sglang-jax does this in ModelRunner.load_model()).
    - KV head replication is applied if `tp_size > original_num_kv_heads`.
    """

    import numpy as np

    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    try:
        import ml_dtypes
    except Exception:  # pragma: no cover
        ml_dtypes = None

    tp_size = int(getattr(mesh, "shape", {}).get("tensor", 1))
    if tp_size <= 0:
        tp_size = 1

    dtype_raw = str(param_dtype).strip().lower()
    if dtype_raw in {"bfloat16", "bf16"}:
        jax_dtype = jnp.bfloat16
        np_dtype = getattr(ml_dtypes, "bfloat16", None) if ml_dtypes is not None else None
        if np_dtype is None:
            # Fallback: keep float16 host buffers; cast on device when assigning.
            np_dtype = np.float16
    elif dtype_raw in {"float16", "f16"}:
        jax_dtype = jnp.float16
        np_dtype = np.float16
    elif dtype_raw in {"float32", "f32"}:
        jax_dtype = jnp.float32
        np_dtype = np.float32
    else:
        raise ValueError(f"Unsupported param_dtype={param_dtype!r} (expected bf16/f16/f32).")

    def _get(path: str):
        cur = mllm_params
        for key in path.split("."):
            if key not in cur:
                raise KeyError(path)
            cur = cur[key]
        return cur

    def _has(path: str) -> bool:
        cur = mllm_params
        for key in path.split("."):
            if key not in cur:
                return False
            cur = cur[key]
        return True

    def _as_numpy(x):
        if isinstance(x, np.ndarray):
            arr = x
        else:
            arr = np.array(x)
        if arr.dtype != np_dtype:
            arr = arr.astype(np_dtype)
        return arr

    def _shard(arr: np.ndarray, sharding_spec: tuple) -> jax.Array:
        return jax.device_put(arr, NamedSharding(mesh, P(*sharding_spec))).astype(jax_dtype)

    def _spec_for_target_path(target_path: str) -> tuple:
        # Match sglang-jax Qwen2 weight mappings.
        if target_path in ("model.embed_tokens.embedding", "lm_head.embedding"):
            return ("tensor", None)
        if target_path.endswith(".scale") or target_path.endswith(".bias"):
            return (None,)
        if target_path.endswith(".weight"):
            if any(
                part in target_path
                for part in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
            ):
                return (None, "tensor")
            if any(part in target_path for part in ("o_proj", "down_proj")):
                return ("tensor", None)
        raise ValueError(f"Cannot infer sharding spec for target_path={target_path!r}")

    original_kv_heads = int(model_config.get_total_num_kv_heads())
    target_kv_heads = int(getattr(model_config, "num_key_value_heads", original_kv_heads))
    head_dim = int(getattr(model_config, "head_dim", 0) or 0)
    if head_dim <= 0:
        # Fallback; should not happen for Qwen2-family configs.
        head_dim = int(model_config.hidden_size // model_config.num_attention_heads)

    kv_strategy = "replicate"
    try:
        kv_strategy = str(model_config.get_kv_padding_strategy())
    except Exception:
        kv_strategy = "replicate"

    needs_replication = False
    try:
        needs_replication = bool(model_config.needs_kv_head_replication(tp_size))
    except Exception:
        needs_replication = target_kv_heads > original_kv_heads

    tie_word_embeddings = bool(
        getattr(getattr(model_config, "hf_text_config", model_config), "tie_word_embeddings", False)
    )

    out: dict[str, jax.Array] = {}

    # Embedding + final norm
    out["model.embed_tokens.embedding"] = _shard(
        _as_numpy(_get("model.embed_tokens.embedding")),
        _spec_for_target_path("model.embed_tokens.embedding"),
    )
    out["model.norm.scale"] = _shard(
        _as_numpy(_get("model.norm.scale")), _spec_for_target_path("model.norm.scale")
    )

    # LM head (if untied)
    if not tie_word_embeddings:
        head_kernel = _as_numpy(_get("lm_head.kernel"))
        head_embedding = lm_head_kernel_to_embedding(head_kernel)
        out["lm_head.embedding"] = _shard(
            head_embedding, _spec_for_target_path("lm_head.embedding")
        )

    # Transformer blocks
    num_layers = int(getattr(model_config, "num_hidden_layers", 0))
    if num_layers <= 0:
        raise ValueError(f"Invalid num_hidden_layers={num_layers}")

    for layer_idx in range(num_layers):
        src_prefix = f"model.layers_{layer_idx}"
        dst_prefix = f"model.layers.{layer_idx}"

        def _copy(dst_suffix: str, src_suffix: str, sharding_spec: tuple | None = None):
            dst = f"{dst_prefix}.{dst_suffix}"
            src = f"{src_prefix}.{src_suffix}"
            spec = sharding_spec if sharding_spec is not None else _spec_for_target_path(dst)
            out[dst] = _shard(_as_numpy(_get(src)), spec)

        _copy("input_layernorm.scale", "input_layernorm.scale")
        _copy("post_attention_layernorm.scale", "post_attention_layernorm.scale")

        # Attention projections
        _copy("self_attn.q_proj.weight", "self_attn.q_proj.kernel")
        if _has(f"{src_prefix}.self_attn.q_proj.bias"):
            _copy("self_attn.q_proj.bias", "self_attn.q_proj.bias")
        else:
            out[f"{dst_prefix}.self_attn.q_proj.bias"] = _shard(
                np.zeros((model_config.num_attention_heads * head_dim,), dtype=np_dtype),
                _spec_for_target_path(f"{dst_prefix}.self_attn.q_proj.bias"),
            )

        k_weight = _as_numpy(_get(f"{src_prefix}.self_attn.k_proj.kernel"))
        v_weight = _as_numpy(_get(f"{src_prefix}.self_attn.v_proj.kernel"))

        if _has(f"{src_prefix}.self_attn.k_proj.bias"):
            k_bias = _as_numpy(_get(f"{src_prefix}.self_attn.k_proj.bias"))
        else:
            k_bias = np.zeros((original_kv_heads * head_dim,), dtype=np_dtype)
        if _has(f"{src_prefix}.self_attn.v_proj.bias"):
            v_bias = _as_numpy(_get(f"{src_prefix}.self_attn.v_proj.bias"))
        else:
            v_bias = np.zeros((original_kv_heads * head_dim,), dtype=np_dtype)

        if needs_replication and target_kv_heads != original_kv_heads:
            k_weight = kv_head_expand(
                k_weight,
                head_dim=head_dim,
                original_kv_heads=original_kv_heads,
                target_kv_heads=target_kv_heads,
                axis=1,
                strategy=kv_strategy,
            )
            k_bias = kv_head_expand(
                k_bias,
                head_dim=head_dim,
                original_kv_heads=original_kv_heads,
                target_kv_heads=target_kv_heads,
                axis=0,
                strategy=kv_strategy,
            )
            v_weight = kv_head_expand(
                v_weight,
                head_dim=head_dim,
                original_kv_heads=original_kv_heads,
                target_kv_heads=target_kv_heads,
                axis=1,
                strategy=kv_strategy,
            )
            v_bias = kv_head_expand(
                v_bias,
                head_dim=head_dim,
                original_kv_heads=original_kv_heads,
                target_kv_heads=target_kv_heads,
                axis=0,
                strategy=kv_strategy,
            )

        out[f"{dst_prefix}.self_attn.k_proj.weight"] = _shard(
            k_weight, _spec_for_target_path(f"{dst_prefix}.self_attn.k_proj.weight")
        )
        out[f"{dst_prefix}.self_attn.k_proj.bias"] = _shard(
            k_bias, _spec_for_target_path(f"{dst_prefix}.self_attn.k_proj.bias")
        )
        out[f"{dst_prefix}.self_attn.v_proj.weight"] = _shard(
            v_weight, _spec_for_target_path(f"{dst_prefix}.self_attn.v_proj.weight")
        )
        out[f"{dst_prefix}.self_attn.v_proj.bias"] = _shard(
            v_bias, _spec_for_target_path(f"{dst_prefix}.self_attn.v_proj.bias")
        )

        _copy("self_attn.o_proj.weight", "self_attn.o_proj.kernel")

        # MLP
        _copy("mlp.gate_proj.weight", "mlp.gate_proj.kernel")
        _copy("mlp.up_proj.weight", "mlp.up_proj.kernel")
        _copy("mlp.down_proj.weight", "mlp.down_proj.kernel")

    return out


__all__ = [
    "lm_head_kernel_to_embedding",
    "kv_head_expand",
    "build_sglang_qwen2_param_dict_from_mllm_params",
]
