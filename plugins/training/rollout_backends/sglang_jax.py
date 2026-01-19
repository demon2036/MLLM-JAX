from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from plugins.training.api import RolloutResult
from plugins.training.grpo.sampling import build_chat_prompts


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_env_int(name: str, default: int | None) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def _get_env_float(name: str, default: float | None) -> float | None:
    value = os.environ.get(name)
    if value is None:
        return default
    return float(value)


def _get_env_int_list(name: str) -> list[int] | None:
    value = os.environ.get(name)
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",")]
    ints = [int(p) for p in parts if p]
    return ints or None


def _normalize_kv_cache_dtype(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"", "auto"}:
        return "auto"
    if normalized in {"bf16", "bfloat16"}:
        # `sgl_jax` currently expects the token "bf16" (not "bfloat16").
        return "bf16"
    return value


def _require_tokenizer_pad_id(tokenizer: Any) -> int:
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is not None:
        return int(pad)
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        return int(eos)
    return 0


@dataclass
class SglangJaxRolloutBackend:
    """Rollout backend backed by an in-process `sglang-jax` Engine.

    Notes
    -----
    - The Engine is initialized lazily on the first `rollout()` call so that unit
      tests can run without the `sgl_jax` dependency installed.
    """

    tokenizer: Any
    model_path: str

    _engine: Any | None = field(default=None, init=False, repr=False)
    _last_synced_params: Any | None = field(default=None, init=False, repr=False)
    _base_model_state_leaves: tuple[Any, ...] | None = field(default=None, init=False, repr=False)
    _warned_release_memory_occupation: bool = field(default=False, init=False, repr=False)

    def initialize(self) -> None:
        """Eagerly initialize the underlying Engine.

        This is useful when co-locating rollout + training on the same TPU: it
        lets the Engine size its KV cache before the training state (FSDP +
        optimizer) consumes most of the device memory.
        """

        self._ensure_engine()

    def sync_weights(self, params: Any) -> None:
        """Sync (hot-swap) the latest training params into the Engine.

        Implementation: follow Tunix's approach and replace `model_state_leaves`
        in the underlying `ModelRunner` so the next forward uses the latest
        weights.

        This method is intentionally best-effort and depends on:
        - `enable_single_process=True` (so we can access the scheduler/model runner)
        - compatible param trees (Qwen2-style weights)
        """

        if self._last_synced_params is params:
            return
        self._last_synced_params = params

        engine = self._ensure_engine()

        try:
            import jax
            from flax import nnx
        except Exception as e:  # pragma: no cover - dependency/runtime specific
            raise RuntimeError("sglang_jax weight sync requires `jax` and `flax`.") from e

        scheduler = engine.scheduler_info.get("scheduler") if hasattr(engine, "scheduler_info") else None
        if scheduler is None:
            raise RuntimeError(
                "sglang-jax Engine does not expose a scheduler object (need `enable_single_process=True`)."
            )

        model_runner = scheduler.tp_worker.worker.model_runner
        # NOTE: Do NOT capture the Engine's initial weights by default.
        #
        # Keeping a reference to the initial (dummy) weights pins a full
        # parameter-sized buffer on device and can easily OOM on v4-8 when
        # co-locating training + rollout. If you want to enable restoring the
        # initial weights after each rollout, opt in explicitly via:
        #   SGLANG_JAX_CAPTURE_BASE_WEIGHTS=1
        if self._base_model_state_leaves is None and _get_env_bool("SGLANG_JAX_CAPTURE_BASE_WEIGHTS", False):
            try:  # pragma: no cover - depends on external engine internals
                self._base_model_state_leaves = tuple(model_runner.model_state_leaves)
            except Exception:
                self._base_model_state_leaves = None
        transformer_state = nnx.split(model_runner.model)[1]

        train_flat = _flatten_tree_to_dotted_dict(params)

        updated_keys = 0
        missing_keys = 0
        reused_keys = 0
        device_put_keys = 0
        device_put_bytes = 0
        debug_sync = _get_env_bool("SGLANG_JAX_WEIGHT_SYNC_DEBUG", False)
        device_put_records: list[tuple[int, str]] = []
        pending_transposes: list[tuple[str, Any, Any, Any, tuple[int, ...]]] = []
        for tgt_key, tgt_param in transformer_state.flat_state():
            tgt_path = _nnx_key_to_dotted_path(tgt_key)
            train_key, transpose = _engine_key_to_train_key(tgt_path)
            if train_key is None:
                continue

            if hasattr(tgt_param, "value"):
                tgt_value = tgt_param.value
            else:
                tgt_value = tgt_param
            tgt_sharding = tgt_value.sharding
            tgt_dtype = tgt_value.dtype
            tgt_shape = tuple(int(x) for x in tgt_value.shape)

            if train_key not in train_flat:
                # Some checkpoints omit attention bias vectors. The Engine always
                # instantiates them; use zeros so forward stays compatible.
                if train_key.endswith(".bias"):
                    import jax.numpy as jnp

                    val = jnp.zeros(tgt_shape, dtype=tgt_dtype)
                    val = jax.device_put(val, tgt_sharding)
                    device_put_keys += 1
                    size_bytes = int(np.prod(tgt_shape)) * int(np.dtype(tgt_dtype).itemsize)
                    device_put_bytes += size_bytes
                    if debug_sync:
                        device_put_records.append((size_bytes, train_key))
                    if hasattr(tgt_param, "value"):
                        tgt_param.value = val
                    else:
                        tgt_param = val
                    updated_keys += 1
                    continue
                missing_keys += 1
                continue

            val = train_flat[train_key]
            if transpose:
                pending_transposes.append((train_key, tgt_param, tgt_sharding, tgt_dtype, tgt_shape))
                continue

            val = _align_shape_and_dtype(val, tgt_shape=tgt_shape, tgt_dtype=tgt_dtype, key=train_key)
            maybe = _maybe_rewrap_global_array(val, tgt_sharding)
            if maybe is not None:
                val = maybe
                reused_keys += 1
            else:
                val = jax.device_put(val, tgt_sharding, may_alias=True)
                device_put_keys += 1
                size_bytes = int(np.prod(tgt_shape)) * int(np.dtype(tgt_dtype).itemsize)
                device_put_bytes += size_bytes
                if debug_sync:
                    device_put_records.append((size_bytes, train_key))

            if hasattr(tgt_param, "value"):
                tgt_param.value = val
            else:
                tgt_param = val
            updated_keys += 1

        if updated_keys == 0:
            raise RuntimeError(
                "sglang-jax weight sync made no updates. Likely mismatch between the training param tree and "
                "sglang-jax model state keys."
            )
        if missing_keys:
            raise RuntimeError(
                "sglang-jax weight sync is missing some mapped keys from the training param tree. "
                f"missing_keys={missing_keys} (first fix mapping/key formatting)."
            )

        # Apply non-transpose updates first so the Engine can drop references to
        # its own large weight buffers before we run any potentially expensive
        # transpose operations (notably lm_head).
        new_model_state_leaves, _ = jax.tree_util.tree_flatten(transformer_state)
        model_runner.model_state_leaves = new_model_state_leaves

        for train_key, tgt_param, tgt_sharding, tgt_dtype, tgt_shape in pending_transposes:
            # Engine expects lm_head.embedding [vocab, hidden] while this repo
            # stores lm_head.kernel [hidden, vocab] (Flax Dense).
            val = train_flat[train_key]
            try:
                transposed = val.T
            except Exception as e:  # pragma: no cover - depends on runtime memory/layout
                raise RuntimeError("Failed to transpose lm_head weights for sglang-jax sync.") from e
            transposed = _align_shape_and_dtype(
                transposed,
                tgt_shape=tgt_shape,
                tgt_dtype=tgt_dtype,
                key=f"{train_key}.T",
            )
            maybe = _maybe_rewrap_global_array(transposed, tgt_sharding)
            if maybe is not None:
                transposed = maybe
                reused_keys += 1
            else:
                transposed = jax.device_put(transposed, tgt_sharding, may_alias=True)
                device_put_keys += 1
                size_bytes = int(np.prod(tgt_shape)) * int(np.dtype(tgt_dtype).itemsize)
                device_put_bytes += size_bytes
                if debug_sync:
                    device_put_records.append((size_bytes, f"{train_key}.T"))
            if hasattr(tgt_param, "value"):
                tgt_param.value = transposed
            updated_keys += 1

        if pending_transposes:
            new_model_state_leaves, _ = jax.tree_util.tree_flatten(transformer_state)
            model_runner.model_state_leaves = new_model_state_leaves

        # Ensure weight sync time is correctly attributed in the runner's timers.
        jax.block_until_ready(model_runner.model_state_leaves)

        if jax.process_index() == 0:
            msg = f"sglang_jax_weight_sync=ok updated_keys={updated_keys} missing_keys={missing_keys}"
            if debug_sync:
                gb = device_put_bytes / (1024**3)
                msg += f" reused_keys={reused_keys} device_put_keys={device_put_keys} device_put_gb={gb:.3f}"
            print(msg)
            if debug_sync and device_put_records:
                topn = int(_get_env_int("SGLANG_JAX_WEIGHT_SYNC_DEBUG_TOPN", 10) or 10)
                top = sorted(device_put_records, key=lambda x: x[0], reverse=True)[: max(1, topn)]
                parts = []
                for size_bytes, key in top:
                    parts.append(f"{key}={size_bytes / (1024**2):.1f}MiB")
                print("sglang_jax_weight_sync_device_put_top=" + ", ".join(parts))

    def _ensure_engine(self):
        if self._engine is not None:
            return self._engine

        try:
            import jax
            from sgl_jax.srt.entrypoints.engine import Engine
        except Exception as e:  # pragma: no cover - depends on external install
            raise RuntimeError(
                "rollout.backend='sglang_jax' requires the `sglang-jax` Python package.\n"
                "On TPU VM you can install it via:\n"
                "  pip install --no-deps 'git+https://github.com/sgl-project/sglang-jax.git#subdirectory=python'\n"
                "Then ensure any missing deps are installed (fastapi/uvloop/pyzmq/etc)."
            ) from e

        local_devices = list(jax.local_devices())
        tp_size = int(_get_env_int("SGLANG_JAX_TP_SIZE", len(local_devices)) or len(local_devices))
        if tp_size <= 0:
            raise ValueError("SGLANG_JAX_TP_SIZE must be > 0")
        if tp_size > len(local_devices):
            raise ValueError(f"SGLANG_JAX_TP_SIZE={tp_size} exceeds local device count={len(local_devices)}")

        device_indexes = [int(d.id) for d in local_devices[:tp_size]]

        engine_kwargs: dict[str, Any] = {
            "model_path": str(self.model_path),
            "tp_size": tp_size,
            "device_indexes": device_indexes,
            # Critical for TPU + in-process integration (avoid subprocess JAX init/fork issues).
            "enable_single_process": True,
            # Avoid double-loading checkpoints when we will hot-swap training weights anyway.
            "load_format": os.environ.get("SGLANG_JAX_LOAD_FORMAT", "dummy"),
            # Keep Engine logs quiet by default.
            "log_level": os.environ.get("SGLANG_JAX_LOG_LEVEL", "error"),
            # Co-located rollout should default to bf16 to reduce HBM pressure.
            # (Training stays fp32; weight sync casts as needed.)
            "dtype": os.environ.get("SGLANG_JAX_DTYPE", "bfloat16"),
            # Keep KV dtype in sync with model dtype by default; override via env if needed.
            "kv_cache_dtype": _normalize_kv_cache_dtype(os.environ.get("SGLANG_JAX_KV_CACHE_DTYPE", "auto")),
            # Conserve HBM because training already holds model/optimizer state.
            "mem_fraction_static": float(_get_env_float("SGLANG_JAX_MEM_FRACTION_STATIC", 0.25) or 0.25),
            "disable_radix_cache": _get_env_bool("SGLANG_JAX_DISABLE_RADIX_CACHE", True),
            "disable_precompile": _get_env_bool("SGLANG_JAX_DISABLE_PRECOMPILE", True),
        }

        # Optional knobs (env driven) to help fit/benchmark on TPU.
        if (v := _get_env_int("SGLANG_JAX_CONTEXT_LENGTH", None)) is not None:
            engine_kwargs["context_length"] = int(v)
        if (v := _get_env_int("SGLANG_JAX_MAX_TOTAL_TOKENS", None)) is not None:
            engine_kwargs["max_total_tokens"] = int(v)
        if (v := _get_env_int("SGLANG_JAX_MAX_RUNNING_REQUESTS", None)) is not None:
            engine_kwargs["max_running_requests"] = int(v)
        if (v := _get_env_int("SGLANG_JAX_CHUNKED_PREFILL_SIZE", None)) is not None:
            engine_kwargs["chunked_prefill_size"] = int(v)
        if (v := _get_env_int("SGLANG_JAX_PAGE_SIZE", None)) is not None:
            engine_kwargs["page_size"] = int(v)
        if (v := _get_env_int_list("SGLANG_JAX_PRECOMPILE_TOKEN_PADDINGS")) is not None:
            engine_kwargs["precompile_token_paddings"] = v
        if (v := _get_env_int_list("SGLANG_JAX_PRECOMPILE_BS_PADDINGS")) is not None:
            engine_kwargs["precompile_bs_paddings"] = v

        if jax.process_index() == 0:
            print(
                "sglang_jax_engine_init="
                + str(
                    dict(
                        model_path=str(self.model_path),
                        tp_size=tp_size,
                        device_indexes=device_indexes,
                        dtype=engine_kwargs["dtype"],
                        kv_cache_dtype=engine_kwargs["kv_cache_dtype"],
                        mem_fraction_static=engine_kwargs["mem_fraction_static"],
                        disable_radix_cache=engine_kwargs["disable_radix_cache"],
                        disable_precompile=engine_kwargs["disable_precompile"],
                    )
                )
            )

        self._engine = Engine(**engine_kwargs)
        return self._engine

    def flush_cache(self) -> None:
        engine = self._engine
        if engine is None:
            return
        try:
            engine.flush_cache()
        except Exception as e:  # pragma: no cover - depends on external engine behavior
            print(f"WARNING: sglang-jax flush_cache failed: {e}")
        # Release KV/memory pool reservation so training can use the full HBM.
        # Keep this best-effort: some revisions may not implement the API.
        # NOTE: Disabled by default because sglang-jax 0.0.2 has a broken
        # `release_memory_occupation()` implementation on TPU.
        release_mem = _get_env_bool("SGLANG_JAX_RELEASE_MEMORY_OCCUPATION", False) or _get_env_bool(
            "SGLANG_JAX_TRY_RELEASE_MEMORY_OCCUPATION", False
        )
        if not release_mem:
            return
        if not hasattr(engine, "release_memory_occupation"):
            return
        try:  # pragma: no cover - external API surface is version-dependent
            engine.release_memory_occupation()
        except Exception as e:
            if not self._warned_release_memory_occupation:
                print(f"WARNING: sglang-jax release_memory_occupation failed: {e}")
                self._warned_release_memory_occupation = True

    def release_weights(self) -> None:
        """Drop Engine weight references so training can safely donate buffers.

        When co-locating rollout + training in one process, the training update
        step donates the `TrainState` buffers. If the rollout Engine still holds
        references/aliases to those buffers, donation becomes unsafe and can
        also increase peak memory.
        """

        self._last_synced_params = None
        engine = self._engine
        base_leaves = self._base_model_state_leaves
        if engine is None or base_leaves is None:
            return

        try:
            scheduler = engine.scheduler_info.get("scheduler") if hasattr(engine, "scheduler_info") else None
            if scheduler is None:
                return
            model_runner = scheduler.tp_worker.worker.model_runner
            model_runner.model_state_leaves = list(base_leaves)
        except Exception as e:  # pragma: no cover - external engine behavior
            print(f"WARNING: sglang-jax release_weights failed: {e}")

    def rollout(
        self,
        *,
        prompts: Sequence[str],
        params: Any,
        system_prompt: str,
        global_length: int,
        max_length_sample: int,
    ) -> RolloutResult:
        engine = self._ensure_engine()
        # Ensure on-policy weights (runner may call `sync_weights()` explicitly too).
        self.sync_weights(params)
        tokenizer = self.tokenizer

        chat_prompts = build_chat_prompts(tokenizer, list(prompts), system_prompt)
        tokenized = tokenizer(chat_prompts, padding=False, return_attention_mask=False)
        input_ids_list = tokenized["input_ids"]

        # IMPORTANT: Keep the returned training batch shape stable across steps.
        #
        # The JAX training step is `jit`-compiled; if we return variable-length
        # `[B, L]` arrays (based on the max completion length for this batch), the
        # update step will recompile repeatedly and become unusably slow.
        #
        # We follow the existing naive sampler strategy and always return
        # `[B, global_length + max_length_sample]` buffers, packing
        # (prompt + completion) contiguously at the front and leaving the tail as
        # padding.
        base_prompt_len = int(global_length)
        target_completion_len = int(max_length_sample)
        if base_prompt_len <= 0:
            raise ValueError(f"global_length must be > 0, got {global_length!r}")
        if target_completion_len <= 0:
            raise ValueError(f"max_length_sample must be > 0, got {max_length_sample!r}")

        prompt_lens = [len(x) for x in input_ids_list]
        max_prompt_len = max(prompt_lens) if prompt_lens else 0
        target_prompt_len = _ceil_to_bucket(max(base_prompt_len, max_prompt_len))
        if target_prompt_len != base_prompt_len:
            print(
                f"WARNING: padding sglang-jax training batch prompt bucket {base_prompt_len} -> {target_prompt_len} "
                f"(max_prompt_len={max_prompt_len}). Consider increasing rollout.global_length to avoid recompiles."
            )

        target_len = target_prompt_len + _ceil_to_bucket(target_completion_len)

        # Build per-request sampling params (sglang-jax expects a list for batch requests).
        sampling = engine.get_default_sampling_params()
        sampling.max_new_tokens = target_completion_len
        sampling.temperature = float(_get_env_float("SGLANG_JAX_TEMPERATURE", 1.0) or 1.0)
        sampling.top_k = int(_get_env_int("SGLANG_JAX_TOP_K", 50) or 50)
        sampling.top_p = float(_get_env_float("SGLANG_JAX_TOP_P", 1.0) or 1.0)

        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None:
            sampling.stop_token_ids = {int(eos_id)}

        sampling_params = [sampling.convert_to_dict() for _ in input_ids_list]
        outputs = engine.generate(input_ids=input_ids_list, sampling_params=sampling_params, stream=False)

        if isinstance(outputs, dict):
            outputs = [outputs]

        completion_ids_list = [list(o.get("output_ids") or []) for o in outputs]
        answers = tokenizer.batch_decode(completion_ids_list, skip_special_tokens=True)

        pad_id = _require_tokenizer_pad_id(tokenizer)
        batch_size = len(input_ids_list)
        train_input_ids = np.full((batch_size, target_len), pad_id, dtype=np.int32)
        train_attention_mask = np.zeros((batch_size, target_len), dtype=np.int32)
        train_labels = np.zeros((batch_size, target_len), dtype=np.int32)

        for i, (prompt_ids, completion_ids) in enumerate(zip(input_ids_list, completion_ids_list)):
            p_len = len(prompt_ids)
            if p_len:
                train_input_ids[i, :p_len] = np.asarray(prompt_ids, dtype=np.int32)
                train_attention_mask[i, :p_len] = 1

            # Pack the completion immediately after the *true* prompt length (not
            # after `global_length`) to avoid introducing an artificial gap.
            start = p_len
            end = min(start + len(completion_ids), target_len)
            if end > start:
                train_input_ids[i, start:end] = np.asarray(completion_ids[: end - start], dtype=np.int32)
                train_attention_mask[i, start:end] = 1
                train_labels[i, start:end] = 1

        return RolloutResult(
            chat_prompts=chat_prompts,
            answers=answers,
            batch={
                "input_ids": train_input_ids,
                "attention_mask": train_attention_mask,
                "labels": train_labels,
            },
        )


__all__ = ["SglangJaxRolloutBackend"]


def _device_put_or_reuse(val: Any, tgt_sharding: Any) -> Any:
    """Put `val` onto `tgt_sharding`, reusing device buffers when possible.

    On TPU co-located training+rollout, a full `jax.device_put` can temporarily
    triple memory usage (train weights + engine old weights + engine new weights)
    and trigger OOM. When the source array is already sharded across the same
    devices with compatible per-device shapes, we can re-wrap the existing
    buffers into a new global array with the target sharding without copying.
    """

    import jax

    maybe = _maybe_rewrap_global_array(val, tgt_sharding)
    if maybe is not None:
        return maybe
    return jax.device_put(val, tgt_sharding)


def _maybe_rewrap_global_array(val: Any, tgt_sharding: Any) -> Any | None:
    try:
        import jax
    except Exception:  # pragma: no cover - jax is required at runtime
        return None

    array_cls = getattr(jax, "Array", None)
    if array_cls is None or not isinstance(val, array_cls):
        return None

    # Only safe/possible when all shards are addressable locally.
    if int(jax.process_count()) != 1:
        return None

    mesh = getattr(tgt_sharding, "mesh", None)
    if mesh is None:
        return None

    tgt_devices = list(getattr(mesh, "local_devices", ()) or ())
    if not tgt_devices:
        return None

    device_to_buf: dict[Any, Any] = {}
    for shard in val.addressable_shards:
        if shard.device in device_to_buf:
            # Multiple shards per device (unexpected for our use case).
            return None
        device_to_buf[shard.device] = shard.data

    if any(dev not in device_to_buf for dev in tgt_devices):
        return None

    try:
        bufs = [device_to_buf[dev] for dev in tgt_devices]
        return jax.make_array_from_single_device_arrays(tuple(int(x) for x in val.shape), tgt_sharding, bufs)
    except Exception:  # pragma: no cover - depends on sharding compatibility
        return None


def _nnx_key_to_dotted_path(key: Any) -> str:
    """Convert an NNX `flat_state()` key to a stable dotted string.

    NNX may encode list elements as nested tuples like `('layers', 0)`; flatten
    them so downstream key mapping can treat it as `layers.0`.
    """

    parts: list[str] = []
    for item in key:
        if isinstance(item, (tuple, list)):
            parts.extend(str(x) for x in item)
        else:
            parts.append(str(item))
    return ".".join(parts)


def _flatten_tree_to_dotted_dict(tree: Any) -> dict[str, Any]:
    """Flatten a (nested) mapping-like pytree into `{'a.b.c': leaf}` form."""

    out: dict[str, Any] = {}

    def rec(prefix: str, node: Any) -> None:
        if isinstance(node, Mapping):
            for k, v in node.items():
                key = str(k)
                rec(f"{prefix}.{key}" if prefix else key, v)
            return
        out[prefix] = node

    rec("", tree)
    return out


_LAYER_PREFIX = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def _engine_key_to_train_key(engine_key: str) -> tuple[str | None, bool]:
    """Map an sglang-jax NNX flat key to this repo's training param flat key."""

    if engine_key == "lm_head.embedding":
        return "lm_head.kernel", True

    if engine_key in {"model.embed_tokens.embedding", "model.norm.scale"}:
        return engine_key, False

    m = _LAYER_PREFIX.match(engine_key)
    if m:
        layer_id = m.group(1)
        rest = m.group(2)
        engine_key = f"model.layers_{layer_id}.{rest}"
    elif engine_key.startswith("model.layers."):
        # Unexpected layer key format (best-effort skip).
        return None, False

    # LinearBase uses `.weight` for the kernel, Flax uses `.kernel`.
    if engine_key.endswith(".weight"):
        return engine_key[: -len(".weight")] + ".kernel", False
    if engine_key.endswith((".bias", ".scale")):
        return engine_key, False

    return None, False


def _align_shape_and_dtype(val: Any, *, tgt_shape: tuple[int, ...] | None, tgt_dtype: Any, key: str) -> Any:
    """Best-effort align training weights to the Engine leaf (shape + dtype)."""

    import jax.numpy as jnp

    if hasattr(val, "dtype") and tgt_dtype is not None and val.dtype != tgt_dtype:
        val = val.astype(tgt_dtype)

    if tgt_shape is None:
        return val

    src_shape = tuple(getattr(val, "shape", ()))
    if src_shape == tuple(tgt_shape):
        return val
    if len(src_shape) != len(tgt_shape):
        raise RuntimeError(f"Shape rank mismatch for {key}: {src_shape} vs {tgt_shape}")
    if any(int(s) > int(t) for s, t in zip(src_shape, tgt_shape)):
        raise RuntimeError(f"Shape mismatch (cannot shrink) for {key}: {src_shape} vs {tgt_shape}")

    pad_width = [(0, int(t) - int(s)) for s, t in zip(src_shape, tgt_shape)]
    return jnp.pad(val, pad_width)


_DEFAULT_SEQ_BUCKETS = (128, 256, 512, 1024, 2048, 4096, 8192)


def _ceil_to_bucket(length: int, buckets: Sequence[int] = _DEFAULT_SEQ_BUCKETS) -> int:
    value = int(length)
    for b in buckets:
        if int(b) >= value:
            return int(b)
    return value
