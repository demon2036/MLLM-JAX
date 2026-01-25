from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from transformers import AutoConfig, AutoTokenizer

from plugins.common.tokenizer import prepare_tokenizer
from plugins.llm.dtypes import parse_dtype
from plugins.llm.weights import convert_hf_state_dict_to_flax_llama_params, ensure_tied_lm_head, load_hf_state_dict
from plugins.sft.jax.params import VocabResizeResult, resize_lm_vocab


@dataclass(frozen=True)
class LlmBundle:
    model: Any
    params: Any
    tokenizer: Any
    pad_token_id: int
    hf_config: Any
    vocab_resize: VocabResizeResult | None


def _pad_vocab_size(vocab_size: int, multiple: int) -> int:
    multiple = int(multiple)
    if multiple <= 1:
        return int(vocab_size)
    r = int(vocab_size) % multiple
    return int(vocab_size) if r == 0 else int(vocab_size) + (multiple - r)


def build_llm_bundle(
    *,
    mesh: Any,
    model_path: str,
    tokenizer: Any | None = None,
    param_dtype: str = "float32",
    compute_dtype: str = "bfloat16",
    trust_remote_code: bool = True,
    padding_side: str = "right",
    allow_torch_fallback: bool = True,
    init_seed: int = 0,
    only_model: bool = False,
) -> LlmBundle:
    """Build (model, params, tokenizer) on a given JAX mesh."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding

    from MLLM_JAX.language.llama.llama import LlamaForCausalLM, LlamaJaxConfig
    from MLLM_JAX.language.qwen2.modular_qwen2 import Qwen2ForCausalLM
    from MLLM_JAX.utils import get_partition_rules_llama, match_partition_rules

    model_path = str(model_path)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=bool(trust_remote_code))
    tokenizer, pad_token_id = prepare_tokenizer(tokenizer, padding_side=str(padding_side))

    tokenizer_vocab_size = int(len(tokenizer))
    fsdp = int(getattr(mesh, "shape", {}).get("fsdp", 1))
    tp = int(getattr(mesh, "shape", {}).get("tp", 1))
    padded_vocab_size = _pad_vocab_size(tokenizer_vocab_size, fsdp * tp)

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=bool(trust_remote_code))
    hf_config.vocab_size = int(padded_vocab_size)

    attention_mesh = mesh if jax.devices()[0].platform == "tpu" else None
    jax_config = LlamaJaxConfig(mesh=attention_mesh, dtype=parse_dtype(compute_dtype), param_dtype=parse_dtype(param_dtype))

    model_type = str(getattr(hf_config, "model_type", "") or "")
    if model_type in {"qwen2"}:
        model = Qwen2ForCausalLM(hf_config, jax_config)
    elif model_type in {"llama"}:
        model = LlamaForCausalLM(hf_config, jax_config=jax_config)
    else:
        raise ValueError(f"Unsupported model_type={model_type!r} (expected qwen2/llama)")

    if only_model:
        return LlmBundle(model=model, params=None, tokenizer=tokenizer, pad_token_id=int(pad_token_id), hf_config=hf_config, vocab_resize=None)

    state_dict = ensure_tied_lm_head(load_hf_state_dict(model_path, allow_torch_fallback=allow_torch_fallback))
    params = convert_hf_state_dict_to_flax_llama_params(state_dict)
    params = jax.tree_util.tree_map(lambda x: np.asarray(x), params)

    rng = jax.random.PRNGKey(int(init_seed))
    params, vocab_resize = resize_lm_vocab(params=params, new_vocab_size=int(padded_vocab_size), rng=rng)

    pdtype = parse_dtype(param_dtype)
    params = jax.tree_util.tree_map(lambda x: np.asarray(x, dtype=np.dtype(pdtype)), params)

    shapes = jax.eval_shape(lambda x: x, params)
    partitions = match_partition_rules(get_partition_rules_llama(), shapes)
    shardings = jax.tree_util.tree_map(lambda spec: NamedSharding(mesh, spec), partitions)
    params = jax.tree_util.tree_map(lambda x, sh: jax.device_put(jnp.asarray(x, dtype=pdtype), sh), params, shardings)

    return LlmBundle(
        model=model,
        params=params,
        tokenizer=tokenizer,
        pad_token_id=int(pad_token_id),
        hf_config=hf_config,
        vocab_resize=vocab_resize,
    )


__all__ = ["LlmBundle", "build_llm_bundle"]
