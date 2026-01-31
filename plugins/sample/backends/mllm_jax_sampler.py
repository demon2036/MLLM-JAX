from __future__ import annotations

import os
import random
from functools import partial
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
from transformers import AutoConfig, AutoTokenizer

from plugins.training.core.io.hf_safetensors import load_hf_safetensors_state_dict
from plugins.training.core.io.hf_config import ensure_rope_theta

from MLLM_JAX.language.llama.llama import LlamaForCausalLM, LlamaJaxConfig, convert_torch_to_flax_llama
from MLLM_JAX.language.qwen2.configuration_qwen2 import init_cache, pad_cache_right
from MLLM_JAX.language.qwen2.modular_qwen2 import Qwen2ForCausalLM
from MLLM_JAX.utils import collect_process_data, get_partition_rules_llama, match_partition_rules, _form_global_array


def _resolve_param_dtype() -> jnp.dtype:
    param_dtype_raw = os.environ.get("MLLM_JAX_PARAM_DTYPE", "float32").strip().lower()
    if param_dtype_raw in {"float32", "f32"}:
        return jnp.float32
    if param_dtype_raw in {"bfloat16", "bf16"}:
        return jnp.bfloat16
    if param_dtype_raw in {"float16", "f16"}:
        return jnp.float16
    raise ValueError(
        f"Unsupported MLLM_JAX_PARAM_DTYPE={param_dtype_raw!r} (expected float32/bfloat16/float16)."
    )


def _top_k_sampling_batched(rng, logits, k: int = 50, t: float = 0.9):
    logits = logits / float(t)

    def sample_single(rng_single, logits_single):
        top_logits, top_indices = jax.lax.top_k(logits_single, int(k))
        sampled_relative_idx = jax.random.categorical(rng_single, top_logits)
        return top_indices[sampled_relative_idx]

    rngs = jax.random.split(rng, int(logits.shape[0]))
    return jax.vmap(sample_single)(rngs, logits)


def get_params(model_path: str, *, allow_torch_fallback: bool = True):
    """Load HF weights and convert to the MLLM_JAX Flax param tree.

    Prefers `*.safetensors` (via `plugins/training/core/io/hf_safetensors.py`). When a model
    repo only provides PyTorch `pytorch_model.bin`, `allow_torch_fallback=True`
    falls back to `transformers.AutoModelForCausalLM` + `state_dict()`.
    """
    try:
        state_dict = load_hf_safetensors_state_dict(model_path)
    except Exception as safetensors_error:
        if not allow_torch_fallback:
            raise
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except Exception as torch_import_error:
            raise RuntimeError(
                f"Failed to load safetensors for {model_path!r} and torch fallback is unavailable. "
                "Either install torch or provide a model with safetensors weights."
            ) from torch_import_error

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        state_dict = model.state_dict()
        del model

        # Keep the original exception around for debugging.
        _ = safetensors_error

    # Some HF repos tie `lm_head` to `embed_tokens` and omit `lm_head.weight`.
    # Our Flax param tree expects an `lm_head.kernel`, so synthesize the missing
    # key before conversion (the conversion function will transpose it).
    if "lm_head.weight" not in state_dict:
        tied = state_dict.get("model.embed_tokens.weight")
        if tied is not None:
            state_dict = dict(state_dict)
            state_dict["lm_head.weight"] = tied

    params = convert_torch_to_flax_llama(state_dict)
    return jax.tree_util.tree_map(lambda x: np.asarray(x), params)


def get_model(mesh: Any, model_path: str = "Qwen/Qwen2.5-14B", *, only_model: bool = False):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    ensure_rope_theta(config)

    jax_config = LlamaJaxConfig(mesh=mesh)
    model_type = str(getattr(config, "model_type", "") or "")
    if model_type in {"qwen2"}:
        model = Qwen2ForCausalLM(config, jax_config)
    elif model_type in {"llama"}:
        model = LlamaForCausalLM(config, jax_config=jax_config)
    else:
        raise ValueError(f"Unsupported model_type={model_type!r} for Sampler (expected qwen2/llama)")

    if only_model:
        return model

    params = get_params(model_path)

    def init_fn(p):
        return p

    state_shapes = jax.eval_shape(init_fn, params)
    train_state_partition = match_partition_rules(get_partition_rules_llama(), state_shapes)
    train_state_sharding = jax.tree_util.tree_map(lambda x: jax.sharding.NamedSharding(mesh, x), train_state_partition)

    param_dtype = _resolve_param_dtype()
    params = jax.tree_util.tree_map(lambda x, d: jnp.asarray(x, dtype=param_dtype, device=d), params, train_state_sharding)
    params = jax.jit(init_fn, out_shardings=train_state_sharding)(params)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, params, tokenizer


@chex.dataclass
class SampleState:
    decoding_step: jnp.int32
    num_input_tokens: jnp.ndarray
    token_buffer: jnp.ndarray
    positions: jnp.ndarray
    cache: Any
    attention_mask: jnp.ndarray
    next_token_buffer: jnp.ndarray
    key: jnp.ndarray
    dones: jnp.ndarray
    sample_steps: jnp.ndarray


def create_sample_state(
    input_ids_pad,
    position_ids,
    cache,
    pad_attention,
    true_length,
    *,
    decoding_step: int = 0,
    key=None,
):
    if key is None:
        key = jax.random.PRNGKey(int(random.randint(0, 2**31 - 1)))

    return SampleState(
        decoding_step=jnp.asarray(int(decoding_step), dtype=jnp.int32),
        num_input_tokens=true_length,
        token_buffer=input_ids_pad,
        positions=position_ids,
        cache=cache,
        attention_mask=pad_attention,
        next_token_buffer=jnp.zeros((pad_attention.shape[0],), dtype=jnp.int32),
        key=key,
        dones=jnp.zeros((pad_attention.shape[0],), dtype=jnp.bool_),
        sample_steps=jnp.zeros((pad_attention.shape[0],), dtype=jnp.int32),
    )


class Sampler:
    def __init__(self, model, tokenizer, mesh=None, *args, **kwargs):
        del args, kwargs
        self.model = model
        self.tokenizer = tokenizer
        self.dtype = jnp.bfloat16
        self.mesh = mesh

        self.key = jax.random.PRNGKey(2036)

        def warp_sample_fn(rng, logits):
            rngs = jax.random.split(rng, jax.device_count())

            def sample_inner(rng_local, logits_local):
                return _top_k_sampling_batched(rng_local[0], logits_local, t=1.0)

            sample_fn = shard_map(
                sample_inner,
                mesh=mesh,
                in_specs=(P(["dp", "fsdp"]), P(["dp", "fsdp"], "tp")),
                out_specs=P(["dp", "fsdp"]),
                check_rep=False,
            )

            return sample_fn(rngs, logits)

        self.sample_fn = jax.jit(warp_sample_fn)
        self.jit_infer_prefill = jax.jit(self.model.apply)
        self.jit_infer_step = jax.jit(self.infer, donate_argnums=(0,))

        self.prefill_bucket = [128, 256, 512, 1024, 2048, 4096, 8192]

        self.global_collect_method_with_path = partial(_form_global_array, global_mesh=self.mesh)
        self.global_collect_method = partial(_form_global_array, (), global_mesh=self.mesh)

    def infer(self, sample_state: SampleState, params):
        i = sample_state.decoding_step
        last_token = sample_state.token_buffer[:, i].reshape((sample_state.token_buffer.shape[0], 1))
        positions = sample_state.positions
        cache = sample_state.cache
        mask = sample_state.attention_mask

        logits, cache = self.model.apply(
            {"params": params},
            input_ids=last_token,
            position_ids=positions,
            attention_mask=mask,
            cache=cache,
        )

        key, key2 = jax.random.split(sample_state.key)
        sample_state.sample_steps += 1 - sample_state.dones
        next_token_predict = self.sample_fn(key2, logits[:, -1])

        next_token_predict = jnp.where(sample_state.dones, int(self.tokenizer.eos_token_id), next_token_predict)
        dones = sample_state.dones | (next_token_predict == int(self.tokenizer.eos_token_id))
        sample_state.dones = dones

        sample_state.key = key
        sample_state.attention_mask = sample_state.attention_mask.at[:, i + 1].set(1)
        sample_state.positions += 1
        sample_state.token_buffer = sample_state.token_buffer.at[:, i + 1].set(next_token_predict)
        sample_state.next_token_buffer = next_token_predict
        sample_state.decoding_step += 1
        sample_state.cache = cache
        return sample_state

    def find_ceil(self, input):
        for num in self.prefill_bucket:
            if int(num) >= int(input):
                return int(num)
        return None

    def prepare_from_prefill_to_decode(self, cache, input_ids_pad, pad_attention, position_ids, *, max_length: int = 8192):
        _b, prefill_length = input_ids_pad.shape
        cache, input_ids_pad, pad_attention, position_ids = jax.tree_util.tree_map(
            collect_process_data, (cache, input_ids_pad, pad_attention, position_ids)
        )

        position_ids = jnp.max(position_ids * pad_attention, axis=1).reshape((-1, 1)) + 1
        cache = pad_cache_right(cache, int(prefill_length), int(max_length))

        input_ids_pad = jnp.pad(
            input_ids_pad,
            ((0, 0), (0, int(max_length))),
            constant_values=int(self.tokenizer.eos_token_id),
        )
        pad_attention = jnp.pad(pad_attention, ((0, 0), (0, int(max_length))), constant_values=0)
        pad_attention = pad_attention.at[:, int(prefill_length)].set(1)

        return jax.tree_util.tree_map_with_path(
            self.global_collect_method_with_path,
            (cache, input_ids_pad, pad_attention, position_ids),
        )

    def generate(self, input_ids_pad, pad_attention, position_ids, prefill_length, max_length: int = 8192, params=None):
        if params is None:
            raise ValueError("sampler.generate(..., params=...) is required")

        prefill_length_i = int(prefill_length)
        cache = init_cache(
            self.model.config,
            int(input_ids_pad.shape[0]),
            max_cache_length=prefill_length_i,
            dtype=self.dtype,
            shard_method=self.global_collect_method,
        )

        input_ids_pad, pad_attention, position_ids = jax.tree_util.tree_map_with_path(
            self.global_collect_method_with_path,
            (input_ids_pad, pad_attention, position_ids),
        )

        logits, cache = self.jit_infer_prefill(
            {"params": params},
            input_ids=input_ids_pad,
            position_ids=position_ids,
            attention_mask=pad_attention,
            cache=cache,
        )

        cache, input_ids_pad, pad_attention, position_ids = self.prepare_from_prefill_to_decode(
            cache,
            input_ids_pad,
            pad_attention,
            position_ids,
            max_length=self.find_ceil(max_length),
        )

        next_token_logits = jnp.take_along_axis(logits, position_ids[..., None] - 1, axis=1)[:, -1]
        next_token_predict = self.sample_fn(self.key, next_token_logits)

        input_ids_pad = input_ids_pad.at[:, prefill_length_i].set(next_token_predict)
        sample_state = create_sample_state(
            input_ids_pad=input_ids_pad,
            position_ids=position_ids,
            cache=cache,
            pad_attention=pad_attention,
            true_length=prefill_length_i,
            decoding_step=prefill_length_i,
            key=self.key,
        )

        for _ in range(int(max_length) - 1):
            sample_state = self.jit_infer_step(sample_state, params)
            if jnp.all(sample_state.dones):
                break

        local_sample_step = collect_process_data(sample_state.sample_steps)
        local_token_buffer = collect_process_data(sample_state.token_buffer)
        local_attention_mask = collect_process_data(sample_state.attention_mask)

        self.key = sample_state.key
        return {
            "local_token_buffer": local_token_buffer,
            "local_sample_step": local_sample_step,
            "local_attention_mask": local_attention_mask,
        }


__all__ = ["SampleState", "Sampler", "create_sample_state", "get_model", "get_params"]
