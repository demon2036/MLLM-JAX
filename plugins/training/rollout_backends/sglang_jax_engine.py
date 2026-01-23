from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from plugins.training.api import RolloutResult, RolloutSampler
from plugins.training.grpo.sampling import build_chat_prompts


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    raw = str(raw).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    return float(raw)


def _best_effort_extract_text(result: object) -> str | None:
    if isinstance(result, dict):
        text = result.get("text")
        if isinstance(text, str):
            return text
        choices = result.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                maybe = first.get("text")
                if isinstance(maybe, str):
                    return maybe
    return None


def _best_effort_extract_output_ids(result: object) -> list[int] | None:
    if isinstance(result, dict):
        ids = result.get("output_ids")
        if isinstance(ids, list) and all(isinstance(x, int) for x in ids):
            return [int(x) for x in ids]
        meta = result.get("meta_info")
        if isinstance(meta, dict):
            ids = meta.get("output_ids")
            if isinstance(ids, list) and all(isinstance(x, int) for x in ids):
                return [int(x) for x in ids]
    return None


def _get_nnx_state_leaf(params_state: object, path: str):
    keys = path.split(".")
    current_level = params_state

    for key in keys:
        if key.isdigit():
            current_level = current_level[int(key)]
            continue
        if hasattr(current_level, "__contains__") and key in current_level:
            current_level = current_level[key]
        elif hasattr(current_level, key):
            current_level = getattr(current_level, key)
        else:
            raise ValueError(f"{path} is not a valid param path")

    return current_level


def _iter_qwen2_target_param_paths(*, model_config: Any) -> list[str]:
    tie_word_embeddings = bool(
        getattr(getattr(model_config, "hf_text_config", model_config), "tie_word_embeddings", False)
    )
    num_layers = int(getattr(model_config, "num_hidden_layers", 0))
    if num_layers <= 0:
        raise ValueError(f"Invalid num_hidden_layers={num_layers}")

    paths: list[str] = [
        "model.embed_tokens.embedding",
        "model.norm.scale",
    ]
    if not tie_word_embeddings:
        paths.append("lm_head.embedding")

    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}"
        paths.extend(
            [
                f"{prefix}.input_layernorm.scale",
                f"{prefix}.post_attention_layernorm.scale",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.q_proj.bias",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.k_proj.bias",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.v_proj.bias",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            ]
        )
    return paths


def _qwen2_spec_for_target_path(target_path: str) -> tuple:
    if target_path in ("model.embed_tokens.embedding", "lm_head.embedding"):
        return ("tensor", None)
    if target_path.endswith(".scale"):
        return (None,)
    if target_path.endswith(".bias"):
        # NOTE: biases are small; sharding mismatches are typically tolerable.
        return (None,)
    if target_path.endswith(".weight"):
        if any(part in target_path for part in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")):
            return (None, "tensor")
        if any(part in target_path for part in ("o_proj", "down_proj")):
            return ("tensor", None)
    raise ValueError(f"Cannot infer sharding spec for target_path={target_path!r}")


def _alias_to_engine_mesh(arr: Any, *, mesh: Any, spec: tuple) -> Any:
    import jax
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    if not hasattr(arr, "addressable_shards"):
        raise TypeError(f"Expected a sharded jax.Array, got: {type(arr).__name__}")
    sharding = NamedSharding(mesh, P(*spec))

    shard_by_device: dict[Any, Any] = {}
    for shard in arr.addressable_shards:
        shard_by_device[shard.device] = shard.data

    try:
        local_bufs = [shard_by_device[d] for d in mesh.local_devices]
    except KeyError as e:
        raise RuntimeError(
            "Cannot alias array to engine mesh: missing local shard for some engine device. "
            "Ensure the rollout engine and training params use the same local devices."
        ) from e

    try:
        return jax.make_array_from_single_device_arrays(arr.shape, sharding, local_bufs)
    except Exception:
        # Fallback: allow a real reshard/copy when the training sharding does not
        # match the rollout engine's expected layout.
        return jax.device_put(arr, sharding)


def _get_by_path(tree: Any, path: str) -> Any:
    cur = tree
    for key in path.split("."):
        if hasattr(cur, "__contains__") and key in cur:  # supports dict/FrozenDict
            cur = cur[key]
        else:
            raise KeyError(path)
    return cur


def _build_sglang_qwen2_param_dict_from_train_params(
    *,
    train_params: Any,
    model_config: Any,
    mesh: Any,
    expected_shapes: Mapping[str, tuple[int, ...]] | None = None,
) -> dict[str, Any]:
    """Build an nnx target_path -> jax.Array mapping from training params (no host copy)."""

    import jax.numpy as jnp

    def _validate_shape(target_path: str, value: Any) -> None:
        if expected_shapes is None:
            return
        expected = expected_shapes.get(str(target_path))
        if expected is None:
            return
        got = tuple(int(x) for x in getattr(value, "shape", ()))
        if got != tuple(int(x) for x in expected):
            raise ValueError(f"Shape mismatch for {target_path}: expected={tuple(expected)}, got={got}")

    head_dim = int(getattr(model_config, "head_dim", 0) or 0)
    if head_dim <= 0:
        hidden_size = int(getattr(model_config, "hidden_size", 0) or 0)
        num_heads = int(getattr(model_config, "num_attention_heads", 0) or 0)
        if hidden_size <= 0 or num_heads <= 0:
            raise ValueError("Cannot infer head_dim from model_config.")
        head_dim = int(hidden_size // num_heads)
        if head_dim <= 0:
            raise ValueError(f"Invalid inferred head_dim={head_dim}.")

    def _kv_expand_2d(arr: Any, *, target: int) -> Any:
        got = int(arr.shape[1])
        target = int(target)
        if got == target:
            return arr
        if got > target:
            return arr[:, :target]
        if got % head_dim != 0 or target % head_dim != 0:
            raise ValueError(f"KV expand requires dims divisible by head_dim={head_dim}, got={got}, target={target}.")
        original_heads = int(got // head_dim)
        target_heads = int(target // head_dim)
        if original_heads <= 0 or target_heads <= 0:
            raise ValueError(f"Invalid KV heads (original={original_heads}, target={target_heads}).")
        num_replicas = (target_heads + original_heads - 1) // original_heads
        reshaped = arr.reshape(arr.shape[0], original_heads, head_dim)
        expanded = jnp.repeat(reshaped, repeats=num_replicas, axis=1)[:, :target_heads, :]
        return expanded.reshape(arr.shape[0], target_heads * head_dim)

    def _kv_expand_1d(arr: Any, *, target: int) -> Any:
        got = int(arr.shape[0])
        target = int(target)
        if got == target:
            return arr
        if got > target:
            return arr[:target]
        if got % head_dim != 0 or target % head_dim != 0:
            raise ValueError(f"KV expand requires dims divisible by head_dim={head_dim}, got={got}, target={target}.")
        original_heads = int(got // head_dim)
        target_heads = int(target // head_dim)
        if original_heads <= 0 or target_heads <= 0:
            raise ValueError(f"Invalid KV heads (original={original_heads}, target={target_heads}).")
        num_replicas = (target_heads + original_heads - 1) // original_heads
        reshaped = arr.reshape(original_heads, head_dim)
        expanded = jnp.repeat(reshaped, repeats=num_replicas, axis=0)[:target_heads, :]
        return expanded.reshape(target_heads * head_dim)

    out: dict[str, Any] = {}

    out["model.embed_tokens.embedding"] = _alias_to_engine_mesh(
        _get_by_path(train_params, "model.embed_tokens.embedding"),
        mesh=mesh,
        spec=_qwen2_spec_for_target_path("model.embed_tokens.embedding"),
    )
    _validate_shape("model.embed_tokens.embedding", out["model.embed_tokens.embedding"])
    out["model.norm.scale"] = _alias_to_engine_mesh(
        _get_by_path(train_params, "model.norm.scale"),
        mesh=mesh,
        spec=_qwen2_spec_for_target_path("model.norm.scale"),
    )
    _validate_shape("model.norm.scale", out["model.norm.scale"])

    tie_word_embeddings = bool(
        getattr(getattr(model_config, "hf_text_config", model_config), "tie_word_embeddings", False)
    )
    if not tie_word_embeddings:
        raise NotImplementedError("Untied lm_head is not supported in the sglang rollout backend yet.")

    num_layers = int(getattr(model_config, "num_hidden_layers", 0))
    if num_layers <= 0:
        raise ValueError(f"Invalid num_hidden_layers={num_layers}")

    for layer_idx in range(num_layers):
        src_prefix = f"model.layers_{layer_idx}"
        dst_prefix = f"model.layers.{layer_idx}"

        def _copy(dst_suffix: str, src_suffix: str):
            dst = f"{dst_prefix}.{dst_suffix}"
            src = f"{src_prefix}.{src_suffix}"
            src_value = _get_by_path(train_params, src)
            if expected_shapes is not None:
                expected = expected_shapes.get(dst)
                if expected is not None:
                    if dst.endswith((".self_attn.k_proj.weight", ".self_attn.v_proj.weight")):
                        src_value = _kv_expand_2d(src_value, target=int(expected[1]))
                    elif dst.endswith((".self_attn.k_proj.bias", ".self_attn.v_proj.bias")):
                        src_value = _kv_expand_1d(src_value, target=int(expected[0]))

            value = _alias_to_engine_mesh(src_value, mesh=mesh, spec=_qwen2_spec_for_target_path(dst))
            _validate_shape(dst, value)
            out[dst] = value

        _copy("input_layernorm.scale", "input_layernorm.scale")
        _copy("post_attention_layernorm.scale", "post_attention_layernorm.scale")

        _copy("self_attn.q_proj.weight", "self_attn.q_proj.kernel")
        _copy("self_attn.q_proj.bias", "self_attn.q_proj.bias")
        _copy("self_attn.k_proj.weight", "self_attn.k_proj.kernel")
        _copy("self_attn.k_proj.bias", "self_attn.k_proj.bias")
        _copy("self_attn.v_proj.weight", "self_attn.v_proj.kernel")
        _copy("self_attn.v_proj.bias", "self_attn.v_proj.bias")
        _copy("self_attn.o_proj.weight", "self_attn.o_proj.kernel")

        _copy("mlp.gate_proj.weight", "mlp.gate_proj.kernel")
        _copy("mlp.up_proj.weight", "mlp.up_proj.kernel")
        _copy("mlp.down_proj.weight", "mlp.down_proj.kernel")

    return out


@dataclass
class SglangJaxRolloutBackend:
    """Rollout backend backed by an in-process sglang-jax Engine.

    Design goals:
    - TPU-first (sglang-jax runs on TPU via JAX).
    - Weight sharing: swap Engine weights from the *existing* training params
      (no host-side reload) to avoid OOM from duplicate weight copies.
    - KV cache lifecycle: optionally drop/rebuild KV buffers between rollout and
      update to free HBM for the training step.

    Notes:
    - Requires `sglang-jax` importable (install editable on TPU, or set PYTHONPATH).
    - Currently supports Qwen2/Qwen2.5 models with tied embeddings.
    """

    sampler: RolloutSampler
    model_path: str

    _engine: Any | None = None
    _model_runner: Any | None = None
    _dummy_param_dict: dict[str, Any] | None = None
    _kv_dropped: bool = False

    # Env-configured knobs (read once per backend instance).
    _debug: bool = field(default_factory=lambda: _env_flag("SGLANG_ROLLOUT_DEBUG", False))
    _engine_log_level: str = field(default_factory=lambda: str(os.environ.get("SGLANG_ROLLOUT_LOG_LEVEL", "error")))
    _download_dir: str = field(default_factory=lambda: str(os.environ.get("SGLANG_ROLLOUT_DOWNLOAD_DIR", "")))

    _max_total_tokens: int = field(default_factory=lambda: _env_int("SGLANG_ROLLOUT_MAX_TOTAL_TOKENS", 262144))
    _max_prefill_tokens: int = field(default_factory=lambda: _env_int("SGLANG_ROLLOUT_MAX_PREFILL_TOKENS", 1024))
    _max_running_requests: int = field(default_factory=lambda: _env_int("SGLANG_ROLLOUT_MAX_RUNNING_REQUESTS", 256))
    _mem_fraction_static: float = field(default_factory=lambda: _env_float("SGLANG_ROLLOUT_MEM_FRACTION_STATIC", 0.80))

    _drop_kv_on_release: bool = field(default_factory=lambda: _env_flag("SGLANG_ROLLOUT_DROP_KV_ON_RELEASE", True))
    _clear_jax_caches_on_kv_drop: bool = field(
        default_factory=lambda: _env_flag("SGLANG_ROLLOUT_CLEAR_JAX_CACHES_ON_KV_DROP", False)
    )

    _temperature: float = field(default_factory=lambda: _env_float("SGLANG_ROLLOUT_TEMPERATURE", 1.0))
    _top_k: int = field(default_factory=lambda: _env_int("SGLANG_ROLLOUT_TOP_K", 50))
    _top_p: float = field(default_factory=lambda: _env_float("SGLANG_ROLLOUT_TOP_P", 1.0))

    def _ensure_engine(self) -> None:
        if self._engine is not None:
            return

        import jax

        if int(jax.process_count()) != 1:
            raise NotImplementedError("SglangJaxRolloutBackend currently supports single-process TPU runs only.")

        tp_size = int(jax.device_count())
        dp_size = 1

        from sgl_jax.srt.entrypoints.engine import Engine
        from flax import nnx

        engine_kwargs: dict[str, Any] = {
            "model_path": str(self.model_path),
            "tokenizer_path": str(self.model_path),
            "trust_remote_code": True,
            "device": "tpu",
            "tp_size": tp_size,
            "dp_size": dp_size,
            "enable_single_process": True,
            "load_format": "dummy",
            "dtype": "bfloat16",
            "mem_fraction_static": float(self._mem_fraction_static),
            "max_prefill_tokens": int(self._max_prefill_tokens),
            "max_total_tokens": int(self._max_total_tokens),
            "max_running_requests": int(self._max_running_requests),
            "disable_precompile": True,
            "skip_server_warmup": True,
            "log_level": str(self._engine_log_level),
        }
        if str(self._download_dir).strip() != "":
            engine_kwargs["download_dir"] = str(self._download_dir)

        engine = Engine(**engine_kwargs)
        self._engine = engine
        self._model_runner = engine.scheduler_info["scheduler"].tp_worker.worker.model_runner

        # Cache the dummy weights so we can swap back before the donated update step.
        model_runner = self._model_runner
        params_state = nnx.state(model_runner.model)
        dummy: dict[str, Any] = {}
        for path in _iter_qwen2_target_param_paths(model_config=model_runner.model_config):
            leaf = _get_nnx_state_leaf(params_state, str(path))
            if not hasattr(leaf, "value"):
                raise TypeError(f"Engine leaf is not an nnx Param: {path}")
            dummy[str(path)] = leaf.value
        self._dummy_param_dict = dummy

        if self._debug:
            print(
                f"[sglang_backend] engine_ready tp_size={tp_size} dp_size={dp_size} "
                f"max_total_tokens={int(self._max_total_tokens)}"
            )

    def _maybe_rebuild_kv(self) -> None:
        if not self._kv_dropped:
            return
        if self._engine is None:
            return
        from plugins.sglang_jax_inference.kv_cache_lifecycle import rebuild_engine_kv_cache

        rebuild_engine_kv_cache(engine=self._engine)
        self._kv_dropped = False

    def sync_weights(self, params: Any) -> None:
        self._ensure_engine()
        self._maybe_rebuild_kv()

        if self._engine is None or self._model_runner is None:
            raise RuntimeError("Engine is not initialized.")

        expected_shapes: dict[str, tuple[int, ...]] | None = None
        if self._dummy_param_dict is not None:
            expected_shapes = {
                str(k): tuple(int(x) for x in v.shape)
                for k, v in self._dummy_param_dict.items()
                if hasattr(v, "shape")
            }

        # Swap Engine weights from the *training* params (no host-side reload).
        param_dict = _build_sglang_qwen2_param_dict_from_train_params(
            train_params=params,
            model_config=self._model_runner.model_config,
            mesh=self._model_runner.mesh,
            expected_shapes=expected_shapes,
        )
        from plugins.sglang_jax_inference.engine_weight_swap import swap_engine_weights_from_param_dict

        swap_engine_weights_from_param_dict(engine=self._engine, param_dict=param_dict)

        # Drop the dict ASAP; the Engine owns the references post-swap.
        del param_dict

    def flush_cache(self) -> None:
        if self._engine is None:
            return
        out = self._engine.flush_cache()
        ok = bool(getattr(out, "success", False))
        if not ok:
            msg = str(getattr(out, "error_msg", "")) if not ok else ""
            raise RuntimeError(f"engine.flush_cache() failed: {msg}")

    def release_weights(self) -> None:
        if self._engine is None:
            return
        if self._dummy_param_dict is None:
            raise RuntimeError("Dummy weights were not captured; cannot safely release.")

        # Detach the Engine from training params before the donated update step.
        from plugins.sglang_jax_inference.engine_weight_swap import swap_engine_weights_from_param_dict

        swap_engine_weights_from_param_dict(engine=self._engine, param_dict=self._dummy_param_dict)

        if self._drop_kv_on_release:
            from plugins.sglang_jax_inference.kv_cache_lifecycle import drop_engine_kv_cache

            drop_engine_kv_cache(
                engine=self._engine,
                flush_cache=True,
                clear_jax_caches=bool(self._clear_jax_caches_on_kv_drop),
            )
            self._kv_dropped = True

    def rollout(
        self,
        *,
        prompts: Sequence[str],
        params: Any,
        system_prompt: str,
        global_length: int,
        max_length_sample: int,
    ) -> RolloutResult:
        self._ensure_engine()
        self._maybe_rebuild_kv()

        if self._engine is None:
            raise RuntimeError("Engine is not initialized.")

        tokenizer = self.sampler.tokenizer
        chat_prompts = build_chat_prompts(tokenizer, list(prompts), system_prompt)

        sampling_params = {
            "temperature": float(self._temperature),
            # sglang-jax expects top_k == -1 to disable; many configs use 0.
            "top_k": -1 if int(self._top_k) == 0 else int(self._top_k),
            "top_p": float(self._top_p),
            "max_new_tokens": int(max_length_sample),
        }
        results = self._engine.generate(prompt=chat_prompts, sampling_params=sampling_params)
        if isinstance(results, dict):
            results = [results]
        if not isinstance(results, list):
            raise TypeError(f"Unexpected engine.generate output type: {type(results).__name__}")
        if len(results) != len(chat_prompts):
            raise RuntimeError(f"Expected {len(chat_prompts)} outputs, got {len(results)}")

        output_ids_per_sample: list[list[int]] = []
        for item in results:
            out_ids = _best_effort_extract_output_ids(item)
            if out_ids is None:
                raise RuntimeError("sglang output missing output_ids; cannot build training batch.")
            output_ids_per_sample.append(out_ids)

        # Decode from the *token ids* we will feed into training to keep reward
        # scoring consistent with the actual sampled tokens.
        trimmed_output_ids = [ids[: int(max_length_sample)] for ids in output_ids_per_sample]
        try:
            decoded = tokenizer.batch_decode(trimmed_output_ids, skip_special_tokens=True)
            answers = [str(x) for x in decoded]
        except Exception:
            answers = []
            for item, out_ids in zip(results, trimmed_output_ids):
                text = _best_effort_extract_text(item)
                if text is None:
                    try:
                        text = tokenizer.decode(out_ids, skip_special_tokens=True)
                    except Exception:
                        text = ""
                answers.append(text)
        output_ids_per_sample = trimmed_output_ids

        # Tokenize prompts for prompt lengths + prompt token ids.
        enc = tokenizer(chat_prompts, return_tensors="np", padding=True, padding_side="right")
        # Important: keep masks/int token buffers in int32 to avoid TPU-slow int64
        # arithmetic in the training step (naive sampler also uses int32 buffers).
        prompt_input_ids = np.asarray(enc["input_ids"], dtype=np.int32)
        prompt_attention_mask = np.asarray(enc["attention_mask"], dtype=np.int32)
        true_prompt_lens = prompt_attention_mask.sum(axis=1).astype(np.int32)

        desired_length = max(int(global_length), int(prompt_input_ids.shape[1]))
        prefill_length = self.sampler.find_ceil(int(desired_length))
        if prefill_length is None:
            raise ValueError(f"No prefill bucket found for desired_length={desired_length}")
        prefill_length = int(prefill_length)

        seq_len = prefill_length + int(max_length_sample)
        pad_token_id = int(getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0) or 0)

        train_input_ids = np.full((len(chat_prompts), seq_len), fill_value=pad_token_id, dtype=np.int32)
        train_attention_mask = np.zeros((len(chat_prompts), seq_len), dtype=np.int32)
        train_labels = np.zeros((len(chat_prompts), seq_len), dtype=np.int32)

        for i in range(len(chat_prompts)):
            prompt_len = int(true_prompt_lens[i])
            if prompt_len <= 0:
                continue
            if prompt_len > seq_len:
                raise ValueError(f"Prompt length {prompt_len} exceeds seq_len={seq_len}")

            train_input_ids[i, :prompt_len] = prompt_input_ids[i, :prompt_len]
            train_attention_mask[i, :prompt_len] = 1

            completion = output_ids_per_sample[i][: int(max_length_sample)]
            completion_len = int(len(completion))
            if completion_len == 0:
                continue

            end = prompt_len + completion_len
            if end > seq_len:
                completion = completion[: seq_len - prompt_len]
                completion_len = int(len(completion))
                end = prompt_len + completion_len
            train_input_ids[i, prompt_len:end] = np.asarray(completion, dtype=np.int32)
            train_attention_mask[i, prompt_len:end] = 1
            train_labels[i, prompt_len:end] = 1

        return RolloutResult(
            chat_prompts=chat_prompts,
            answers=answers,
            batch={
                "input_ids": train_input_ids,
                "attention_mask": train_attention_mask,
                "labels": train_labels,
            },
        )
