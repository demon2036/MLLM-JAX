"""Utilities to drop and rebuild sglang-jax KV cache buffers at runtime.

This module is intentionally dependency-light at import time so it can live in
this repo without forcing local installs of `sglang-jax`/`jax`.

Notes
-----
- This is intended for `enable_single_process=True` Engine usage where we can
  directly access in-process scheduler -> tp_worker -> model_runner.
- "Drop" here means best-effort releasing the device buffers backing the fused
  KV cache (HBM). Whether `bytes_in_use` decreases depends on the backend
  allocator; on TPU it should typically drop when buffers are deleted.
- After dropping, you MUST rebuild before calling `generate()`.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _get_model_runner_from_engine(engine):
    scheduler_info = getattr(engine, "scheduler_info", None)
    if not isinstance(scheduler_info, dict):
        raise TypeError("Engine does not expose `scheduler_info` (dict).")

    scheduler = scheduler_info.get("scheduler")
    if scheduler is None:
        raise RuntimeError(
            "Engine scheduler object is not accessible. "
            "Run with `enable_single_process=True` so the scheduler lives in-process."
        )

    tp_worker = getattr(scheduler, "tp_worker", None)
    worker = getattr(tp_worker, "worker", None)
    model_runner = getattr(worker, "model_runner", None)
    if model_runner is None:
        raise RuntimeError("Cannot locate `model_runner` from Engine internals.")
    return model_runner


def _flush_engine_cache(engine) -> tuple[bool, str]:
    out = engine.flush_cache()
    success = bool(getattr(out, "success", False))
    msg = str(getattr(out, "error_msg", "")) if not success else ""
    return success, msg


def _is_mha_kv_pool(token_to_kv_pool: Any) -> bool:
    return hasattr(token_to_kv_pool, "kv_buffer") and hasattr(token_to_kv_pool, "layer_num")


def _delete_jax_array_buffers(arr: Any) -> bool:
    delete_fn = getattr(arr, "delete", None)
    if callable(delete_fn):
        delete_fn()
        return True
    return False


def drop_engine_kv_cache(
    *,
    engine,
    flush_cache: bool = True,
    clear_jax_caches: bool = False,
) -> dict[str, Any]:
    """Drop (best-effort) KV cache device buffers from a live Engine.

    This only supports `enable_single_process=True` engines.

    Args:
        engine: `sgl_jax.srt.entrypoints.engine.Engine`.
        flush_cache: If True, calls `engine.flush_cache()` first (recommended).
        clear_jax_caches: If True, calls `jax.clear_caches()` after buffer deletion.
            This may help free some memory but can force re-compilation later.
    """

    if flush_cache:
        ok, msg = _flush_engine_cache(engine)
        if not ok:
            raise RuntimeError(f"engine.flush_cache() failed: {msg}")

    model_runner = _get_model_runner_from_engine(engine)
    token_to_kv_pool = getattr(model_runner, "token_to_kv_pool", None)
    if token_to_kv_pool is None:
        raise RuntimeError("model_runner.token_to_kv_pool is missing.")

    if not _is_mha_kv_pool(token_to_kv_pool):
        raise NotImplementedError(
            f"Unsupported KV pool type: {type(token_to_kv_pool)!r}. "
            "Only MHATokenToKVPool-like pools are supported."
        )

    kv_buffer = getattr(token_to_kv_pool, "kv_buffer", None)
    if not isinstance(kv_buffer, list) or not kv_buffer:
        raise RuntimeError("token_to_kv_pool.kv_buffer is empty; nothing to drop.")

    deleted = 0
    missing_delete = 0
    for idx, buf in enumerate(list(kv_buffer)):
        if buf is None:
            continue
        if _delete_jax_array_buffers(buf):
            deleted += 1
        else:
            missing_delete += 1
            kv_buffer[idx] = None

    # Mark as dropped for debugging.
    setattr(token_to_kv_pool, "_kv_cache_buffers_dropped", True)

    if clear_jax_caches:
        import jax

        jax.clear_caches()

    info: dict[str, Any] = {
        "kv_pool_type": type(token_to_kv_pool).__name__,
        "layer_num": int(getattr(token_to_kv_pool, "layer_num")),
        "deleted_buffers": int(deleted),
        "missing_delete_method": int(missing_delete),
        "clear_jax_caches": bool(clear_jax_caches),
    }
    logger.info("Dropped KV cache buffers: %s", info)
    return info


def rebuild_engine_kv_cache(*, engine) -> dict[str, Any]:
    """Rebuild KV cache buffers for a live Engine after `drop_engine_kv_cache()`."""

    import jax
    import jax.numpy as jnp

    model_runner = _get_model_runner_from_engine(engine)
    token_to_kv_pool = getattr(model_runner, "token_to_kv_pool", None)
    if token_to_kv_pool is None:
        raise RuntimeError("model_runner.token_to_kv_pool is missing.")

    if not _is_mha_kv_pool(token_to_kv_pool):
        raise NotImplementedError(
            f"Unsupported KV pool type: {type(token_to_kv_pool)!r}. "
            "Only MHATokenToKVPool-like pools are supported."
        )

    mesh = getattr(token_to_kv_pool, "mesh", None)
    kv_sharding = getattr(token_to_kv_pool, "kv_sharding", None)
    if mesh is None or kv_sharding is None:
        raise RuntimeError("KV pool is missing mesh/kv_sharding; cannot rebuild safely.")

    fused_buffer_shape = (
        int(getattr(token_to_kv_pool, "size")) + int(getattr(token_to_kv_pool, "page_size")),
        int(getattr(token_to_kv_pool, "head_num")) * 2,
        int(getattr(token_to_kv_pool, "head_dim")),
    )
    dtype = getattr(token_to_kv_pool, "dtype")

    make_zero = jax.jit(
        lambda: jnp.zeros(shape=fused_buffer_shape, dtype=dtype),
        out_shardings=kv_sharding,
    )

    layer_num = int(getattr(token_to_kv_pool, "layer_num"))
    with mesh:
        token_to_kv_pool.kv_buffer = [make_zero() for _ in range(layer_num)]

    # Refresh mem accounting logs (best-effort).
    calc = getattr(token_to_kv_pool, "_calculate_memory_usage", None)
    if callable(calc):
        calc()

    setattr(token_to_kv_pool, "_kv_cache_buffers_dropped", False)

    info: dict[str, Any] = {
        "kv_pool_type": type(token_to_kv_pool).__name__,
        "layer_num": layer_num,
        "fused_buffer_shape": tuple(int(x) for x in fused_buffer_shape),
        "dtype": str(jnp.dtype(dtype)),
    }
    logger.info("Rebuilt KV cache buffers: %s", info)
    return info

