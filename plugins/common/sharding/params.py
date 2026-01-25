from __future__ import annotations

from typing import Any

import numpy as np


def _parse_dtype(dtype: Any):
    if not isinstance(dtype, str):
        return dtype
    import jax.numpy as jnp

    name = str(dtype).strip().lower()
    if name in {"bf16", "bfloat16"}:
        return jnp.bfloat16
    if name in {"fp16", "float16"}:
        return jnp.float16
    if name in {"fp32", "float32"}:
        return jnp.float32
    raise ValueError(f"Unsupported dtype string: {dtype!r}")


def place_tree(
    *,
    mesh: Any,
    tree: Any,
    partitions: Any,
    dtype: Any,
    already_numpy: bool = False,
) -> tuple[Any, Any]:
    """Place a pytree of arrays onto a JAX mesh with NamedSharding."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding

    dtype = _parse_dtype(dtype)

    if not already_numpy:
        tree = jax.tree_util.tree_map(lambda x: np.asarray(x, dtype=np.dtype(dtype)), tree)
    shardings = jax.tree_util.tree_map(lambda spec: NamedSharding(mesh, spec), partitions)
    placed = jax.tree_util.tree_map(lambda x, sh: jax.device_put(jnp.asarray(x, dtype=dtype), sh), tree, shardings)
    return placed, shardings


def place_params_llama(*, mesh: Any, params: Any, dtype: Any) -> tuple[Any, Any]:
    """Place llama-like params on mesh using `MLLM_JAX.utils.get_partition_rules_llama()`."""
    import jax

    from MLLM_JAX.utils import get_partition_rules_llama, match_partition_rules

    dtype = _parse_dtype(dtype)
    params_np = jax.tree_util.tree_map(lambda x: np.asarray(x, dtype=np.dtype(dtype)), params)
    shapes = jax.eval_shape(lambda x: x, params_np)
    partitions = match_partition_rules(get_partition_rules_llama(), shapes)
    return place_tree(mesh=mesh, tree=params_np, partitions=partitions, dtype=dtype, already_numpy=True)


__all__ = ["place_params_llama", "place_tree"]
