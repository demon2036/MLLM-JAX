"""Utilities to swap sglang-jax Engine weights at runtime.

This module is intentionally dependency-light at import time so it can live in
this repo without forcing local installs of `sglang-jax`.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
import glob
import logging
from collections.abc import Sequence

logger = logging.getLogger(__name__)


def download_hf_snapshot(
    model_id_or_path: str,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    allow_patterns: Sequence[str] | None = None,
    ignore_patterns: Sequence[str] | None = ("original/**/*",),
) -> str:
    """Return a local directory containing a HF snapshot for `model_id_or_path`.

    If `model_id_or_path` is already a local directory, returns its absolute path.
    """

    if os.path.isdir(model_id_or_path):
        return os.path.abspath(model_id_or_path)

    from huggingface_hub import snapshot_download

    kwargs: dict[str, object] = {}
    if revision is not None:
        kwargs["revision"] = revision
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    if allow_patterns is not None:
        kwargs["allow_patterns"] = list(allow_patterns)
    if ignore_patterns is not None:
        kwargs["ignore_patterns"] = list(ignore_patterns)

    return snapshot_download(model_id_or_path, **kwargs)


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


def swap_engine_weights_from_snapshot_dir(*, engine, snapshot_dir: str) -> int:
    """Load weights from a local HF snapshot dir, then update Engine runtime leaves."""

    if not os.path.isdir(snapshot_dir):
        raise FileNotFoundError(snapshot_dir)
    if not glob.glob(os.path.join(snapshot_dir, "*.safetensors")):
        raise FileNotFoundError(f"No *.safetensors found under: {snapshot_dir}")

    import jax
    from flax import nnx

    model_runner = _get_model_runner_from_engine(engine)
    model_config = model_runner.model_config
    old_len = None
    if hasattr(model_runner, "model_state_leaves"):
        try:
            old_len = len(model_runner.model_state_leaves)
        except Exception:
            old_len = None

    # If the Engine was built with `load_format=dummy`, clear the dummy marker.
    if hasattr(model_config, "_dummy_mode"):
        delattr(model_config, "_dummy_mode")

    model_config.model_path = snapshot_dir
    logger.info("Swapping weights from snapshot_dir=%s", snapshot_dir)

    with jax.set_mesh(model_runner.mesh):
        model_runner.model.load_weights(model_config)
        model_state = nnx.split(model_runner.model)[1]
        model_runner.model_state_leaves, _ = jax.tree_util.tree_flatten(model_state)

    new_len = len(model_runner.model_state_leaves)
    if old_len is not None and new_len != old_len:
        raise RuntimeError(
            f"Model state structure changed after swap (old_len={old_len}, new_len={new_len})."
        )
    logger.info("Swap complete (num_leaves=%s).", new_len)
    return new_len


def swap_engine_weights_from_hf(
    *,
    engine,
    model_id_or_path: str,
    revision: str | None = None,
    cache_dir: str | None = None,
    allow_patterns: Sequence[str] | None = None,
    ignore_patterns: Sequence[str] | None = ("original/**/*",),
) -> tuple[str, int]:
    """Download (if needed) and swap Engine weights. Returns (snapshot_dir, num_leaves)."""

    snapshot_dir = download_hf_snapshot(
        model_id_or_path,
        revision=revision,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )
    num_leaves = swap_engine_weights_from_snapshot_dir(engine=engine, snapshot_dir=snapshot_dir)
    return snapshot_dir, num_leaves


def _get_nnx_state_leaf(params_state, path: str):
    # Copied (with small edits) from sglang-jax WeightLoader._get_param
    keys = path.split(".")
    current_level = params_state

    for key in keys:
        if key.isdigit():
            current_level = current_level[int(key)]
        else:
            if hasattr(current_level, "__contains__") and key in current_level:
                current_level = current_level[key]
            elif hasattr(current_level, key):
                current_level = getattr(current_level, key)
            else:
                raise ValueError(f"{path} is not a valid param path")

    return current_level


def swap_engine_weights_from_param_dict(
    *,
    engine,
    param_dict: Mapping[str, "jax.Array"],
    strict: bool = True,
) -> int:
    """Swap Engine weights from an in-memory param dict (target_path -> jax.Array).

    This is the core primitive used for "weight hot swap" experiments:
    - Build Engine with `load_format=dummy` (so it compiles with stable shapes).
    - Inject real weights by assigning nnx Params, then refresh `model_state_leaves`.

    Args:
        engine: sglang-jax Engine (must be in-process; `enable_single_process=True`).
        param_dict: Mapping from nnx param path (e.g. "model.layers.0.self_attn.q_proj.weight")
            to a sharded `jax.Array` with matching shape/dtype.
        strict: If True, raise on missing/invalid paths. If False, skip unknown paths.

    Returns:
        num_leaves: length of `model_runner.model_state_leaves` after refresh.
    """

    import jax
    from flax import nnx

    model_runner = _get_model_runner_from_engine(engine)
    model_config = model_runner.model_config
    old_len = None
    if hasattr(model_runner, "model_state_leaves"):
        try:
            old_len = len(model_runner.model_state_leaves)
        except Exception:
            old_len = None

    # If the Engine was built with `load_format=dummy`, clear the dummy marker.
    if hasattr(model_config, "_dummy_mode"):
        delattr(model_config, "_dummy_mode")

    params_state = nnx.state(model_runner.model)

    with jax.set_mesh(model_runner.mesh):
        for target_path, value in param_dict.items():
            try:
                model_param = _get_nnx_state_leaf(params_state, str(target_path))
            except Exception:
                if strict:
                    raise
                logger.warning("Skipping unknown param path: %s", target_path)
                continue

            if not hasattr(model_param, "value"):
                if strict:
                    raise TypeError(f"Target path is not an nnx Param: {target_path}")
                logger.warning("Skipping non-Param target path: %s", target_path)
                continue

            # Keep dtype consistent with the compiled model state.
            try:
                target_dtype = getattr(model_param.value, "dtype", None)
                if target_dtype is not None and getattr(value, "dtype", None) != target_dtype:
                    value = value.astype(target_dtype)
            except Exception:
                pass

            model_param.value = value

        model_state = nnx.split(model_runner.model)[1]
        model_runner.model_state_leaves, _ = jax.tree_util.tree_flatten(model_state)

    new_len = len(model_runner.model_state_leaves)
    if old_len is not None and new_len != old_len:
        raise RuntimeError(
            f"Model state structure changed after swap (old_len={old_len}, new_len={new_len})."
        )
    logger.info("Param-dict swap complete (num_leaves=%s).", new_len)
    return new_len


__all__ = [
    "download_hf_snapshot",
    "swap_engine_weights_from_hf",
    "swap_engine_weights_from_snapshot_dir",
    "swap_engine_weights_from_param_dict",
]
