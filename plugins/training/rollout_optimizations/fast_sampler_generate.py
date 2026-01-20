from __future__ import annotations

import types
from typing import Any


def patch_sampler_generate_fast(sampler: Any) -> None:
    """Patch a Sampler-like object with a JAX `while_loop` decode.

    This avoids per-token Python loops and avoids `collect_process_data(cache)` +
    `pad_cache_right(...)` host round-trips used by the baseline implementation.
    """

    if getattr(sampler, "_fast_generate_patched", False):
        return

    original_generate = getattr(sampler, "generate", None)
    if original_generate is None:
        raise AttributeError("sampler has no generate() to patch")

    def _get_decode_fn(self, *, prefill_length: int, max_length: int):
        cache: dict[tuple[int, int], Any] = getattr(self, "_fast_decode_fn_cache", None)
        if cache is None:
            cache = {}
            setattr(self, "_fast_decode_fn_cache", cache)

        key = (int(prefill_length), int(max_length))
        fn = cache.get(key)
        if fn is not None:
            return fn

        import jax
        import jax.numpy as jnp

        prefill_length_i = int(prefill_length)
        max_length_i = int(max_length)
        max_decoding_step = prefill_length_i + max_length_i - 1

        def _decode(sample_state, params):
            def cond_fn(s):
                return jnp.logical_and(
                    s.decoding_step < max_decoding_step,
                    jnp.logical_not(jnp.all(s.dones)),
                )

            def body_fn(s):
                return self.infer(s, params)

            return jax.lax.while_loop(cond_fn, body_fn, sample_state)

        fn = jax.jit(_decode, donate_argnums=(0,))
        cache[key] = fn
        return fn

    def generate_fast(
        self,
        input_ids_pad,
        pad_attention,
        position_ids,
        prefill_length,
        max_length=8192,
        params=None,
    ):
        import numpy as np

        import jax
        import jax.numpy as jnp

        from MLLM_JAX.language.qwen2.configuration_qwen2 import init_cache, pad_cache_right
        from MLLM_JAX.sample.sample_state_right_padding2 import SampleState
        from MLLM_JAX.utils import collect_process_data

        if params is None:
            raise ValueError("sampler.generate(..., params=...) is required for fast generate")

        prefill_length_i = int(prefill_length)
        max_length_i = int(max_length)
        if max_length_i <= 0:
            raise ValueError(f"max_length must be > 0, got {max_length_i}")

        max_length_ceiled = self.find_ceil(max_length_i)
        if max_length_ceiled is None:
            raise ValueError(f"No decode bucket found for max_length={max_length_i}")
        max_length_ceiled = int(max_length_ceiled)

        batch_size = int(np.shape(input_ids_pad)[0])
        eos_token_id = int(self.tokenizer.eos_token_id)

        input_ids_host = np.asarray(input_ids_pad)
        attention_host = np.asarray(pad_attention)
        position_host = np.asarray(position_ids)

        input_ids_global = self.global_collect_method(input_ids_host)
        attention_global = self.global_collect_method(attention_host)
        position_global = self.global_collect_method(position_host)

        token_buffer_host = np.pad(
            input_ids_host,
            ((0, 0), (0, max_length_ceiled)),
            constant_values=eos_token_id,
        )
        attention_full_host = np.pad(
            attention_host,
            ((0, 0), (0, max_length_ceiled)),
            constant_values=0,
        )

        token_buffer = self.global_collect_method(token_buffer_host)
        attention_full = self.global_collect_method(attention_full_host)

        cache = init_cache(
            self.model.config,
            batch_size,
            max_cache_length=prefill_length_i,
            dtype=self.dtype,
            shard_method=self.global_collect_method,
        )

        logits, cache = self.jit_infer_prefill(
            {"params": params},
            input_ids=input_ids_global,
            position_ids=position_global,
            attention_mask=attention_global,
            cache=cache,
        )
        cache = pad_cache_right(cache, prefill_length_i, max_length_ceiled)

        start_positions = jnp.max(position_global * attention_global, axis=1).reshape((-1, 1)) + 1
        next_token_logits = jnp.take_along_axis(logits, start_positions[..., None] - 1, axis=1)[:, -1]
        first_token = self.sample_fn(self.key, next_token_logits)

        token_buffer = token_buffer.at[:, prefill_length_i].set(first_token)
        attention_full = attention_full.at[:, prefill_length_i].set(1)

        sample_state = SampleState(
            decoding_step=jnp.asarray(prefill_length_i, dtype=jnp.int32),
            num_input_tokens=jnp.asarray(prefill_length_i, dtype=jnp.int32),
            token_buffer=token_buffer,
            positions=start_positions,
            cache=cache,
            attention_mask=attention_full,
            next_token_buffer=first_token,
            key=self.key,
            dones=jnp.zeros((batch_size,), dtype=jnp.bool_),
            sample_steps=jnp.zeros((batch_size,), dtype=jnp.int32),
        )

        decode_fn = _get_decode_fn(self, prefill_length=prefill_length_i, max_length=max_length_i)
        sample_state = decode_fn(sample_state, params)

        local_sample_step = collect_process_data(sample_state.sample_steps)
        local_token_buffer = collect_process_data(sample_state.token_buffer)
        local_attention_mask = collect_process_data(sample_state.attention_mask)

        self.key = sample_state.key
        return {
            "local_token_buffer": local_token_buffer,
            "local_sample_step": local_sample_step,
            "local_attention_mask": local_attention_mask,
        }

    sampler.generate = types.MethodType(generate_fast, sampler)
    sampler._fast_generate_patched = True
    sampler._fast_generate_original = original_generate
