from __future__ import annotations

from typing import Any

from plugins.common.hf_safetensors import load_hf_safetensors_state_dict


def load_hf_state_dict(model_path: str, *, allow_torch_fallback: bool = True) -> dict[str, Any]:
    """Load HF weights as a state_dict (safetensors-first).

    When safetensors are unavailable and `allow_torch_fallback=True`, falls back
    to `transformers.AutoModelForCausalLM(...).state_dict()`.
    """
    try:
        return load_hf_safetensors_state_dict(model_path)
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
        _ = safetensors_error
        return state_dict


def ensure_tied_lm_head(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Ensure `lm_head.weight` exists for tied-weights HF repos."""
    if "lm_head.weight" in state_dict:
        return state_dict
    tied = state_dict.get("model.embed_tokens.weight")
    if tied is None:
        return state_dict
    out = dict(state_dict)
    out["lm_head.weight"] = tied
    return out


def _set_nested(tree: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cur: dict[str, Any] = tree
    for key in path[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[path[-1]] = value


def convert_hf_state_dict_to_flax_llama_params(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert a HF llama-like state_dict into the MLLM_JAX Flax param tree.

    This wraps `MLLM_JAX.language.llama.llama.convert_torch_to_flax_llama(...)`
    and patches missing bias params when present in HF weights.
    """
    from MLLM_JAX.language.llama.llama import convert_torch_to_flax_llama

    params: dict[str, Any] = convert_torch_to_flax_llama(state_dict)

    i = 0
    while f"model.layers.{i}.self_attn.q_proj.weight" in state_dict:
        # Attention biases (HF opt-in via `attention_bias=True`).
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            key = f"model.layers.{i}.self_attn.{proj}.bias"
            if key in state_dict:
                _set_nested(params, ("model", f"layers_{i}", "self_attn", proj, "bias"), state_dict[key])

        # MLP biases (HF opt-in via `mlp_bias=True`).
        for proj in ("gate_proj", "up_proj", "down_proj"):
            key = f"model.layers.{i}.mlp.{proj}.bias"
            if key in state_dict:
                _set_nested(params, ("model", f"layers_{i}", "mlp", proj, "bias"), state_dict[key])

        i += 1

    return params


__all__ = [
    "convert_hf_state_dict_to_flax_llama_params",
    "ensure_tied_lm_head",
    "load_hf_state_dict",
]
