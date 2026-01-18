from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Sequence

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
    - MVP intentionally does not implement on-policy weight syncing. The `params`
      argument is currently ignored.
    - The Engine is initialized lazily on the first `rollout()` call so that unit
      tests can run without the `sgl_jax` dependency installed.
    """

    tokenizer: Any
    model_path: str

    _engine: Any | None = field(default=None, init=False, repr=False)

    def initialize(self) -> None:
        """Eagerly initialize the underlying Engine.

        This is useful when co-locating rollout + training on the same TPU: it
        lets the Engine size its KV cache before the training state (FSDP +
        optimizer) consumes most of the device memory.
        """

        self._ensure_engine()

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
            # Keep Engine logs quiet by default.
            "log_level": os.environ.get("SGLANG_JAX_LOG_LEVEL", "error"),
            # Conserve HBM because training already holds model/optimizer state.
            "mem_fraction_static": float(_get_env_float("SGLANG_JAX_MEM_FRACTION_STATIC", 0.1) or 0.1),
            "disable_radix_cache": _get_env_bool("SGLANG_JAX_DISABLE_RADIX_CACHE", True),
            "disable_precompile": _get_env_bool("SGLANG_JAX_DISABLE_PRECOMPILE", False),
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
                        mem_fraction_static=engine_kwargs["mem_fraction_static"],
                        disable_radix_cache=engine_kwargs["disable_radix_cache"],
                        disable_precompile=engine_kwargs["disable_precompile"],
                    )
                )
            )

        self._engine = Engine(**engine_kwargs)
        return self._engine

    def rollout(
        self,
        *,
        prompts: Sequence[str],
        params: Any,
        system_prompt: str,
        global_length: int,
        max_length_sample: int,
    ) -> RolloutResult:
        del params, global_length  # MVP: no weight sync and no bucketed prefill needed.

        engine = self._ensure_engine()
        tokenizer = self.tokenizer

        chat_prompts = build_chat_prompts(tokenizer, list(prompts), system_prompt)
        tokenized = tokenizer(chat_prompts, padding=False, return_attention_mask=False)
        input_ids_list = tokenized["input_ids"]

        # Build per-request sampling params (sglang-jax expects a list for batch requests).
        sampling = engine.get_default_sampling_params()
        sampling.max_new_tokens = int(max_length_sample)
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

        prompt_lens = [len(x) for x in input_ids_list]
        completion_lens = [len(x) for x in completion_ids_list]
        max_len = max((p + c) for p, c in zip(prompt_lens, completion_lens)) if prompt_lens else 0
        if max_len <= 1:
            raise RuntimeError(f"sglang_jax produced an invalid sequence length: {max_len}")

        pad_id = _require_tokenizer_pad_id(tokenizer)
        batch_size = len(input_ids_list)
        train_input_ids = np.full((batch_size, max_len), pad_id, dtype=np.int32)
        train_attention_mask = np.zeros((batch_size, max_len), dtype=np.int32)
        train_labels = np.zeros((batch_size, max_len), dtype=np.int32)

        for i, (prompt_ids, completion_ids) in enumerate(zip(input_ids_list, completion_ids_list)):
            p_len = len(prompt_ids)
            c_len = len(completion_ids)
            end = p_len + c_len
            train_input_ids[i, :p_len] = np.asarray(prompt_ids, dtype=np.int32)
            train_attention_mask[i, :p_len] = 1
            if c_len:
                train_input_ids[i, p_len:end] = np.asarray(completion_ids, dtype=np.int32)
                train_attention_mask[i, p_len:end] = 1
                train_labels[i, p_len:end] = 1

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
