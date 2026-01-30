from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class VocabResizeResult:
    original_vocab_size: int
    new_vocab_size: int
    added_tokens: int


def _get_nested(d: Any, path: tuple[str, ...]) -> Any:
    cur = d
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            raise KeyError("Missing params path: " + "/".join(path))
        cur = cur[key]
    return cur


def resize_lm_vocab(
    *,
    params: Mapping[str, Any],
    new_vocab_size: int,
    rng: jax.Array,
    init_std: float = 0.02,
) -> tuple[dict[str, Any], VocabResizeResult]:
    """Resize token embedding + lm_head to `new_vocab_size`.

    Expected param paths (MLLM_JAX Qwen/Llama):
    - `params["model"]["embed_tokens"]["embedding"]`: [vocab, hidden]
    - `params["lm_head"]["kernel"]`: [hidden, vocab]
    """
    embed_path = ("model", "embed_tokens", "embedding")
    head_path = ("lm_head", "kernel")

    embed = _get_nested(params, embed_path)
    head = _get_nested(params, head_path)

    old_vocab = int(embed.shape[0])
    new_vocab = int(new_vocab_size)
    if new_vocab <= 0:
        raise ValueError(f"new_vocab_size must be > 0, got {new_vocab}")

    if new_vocab == old_vocab:
        return params, VocabResizeResult(original_vocab_size=old_vocab, new_vocab_size=old_vocab, added_tokens=0)

    if int(head.shape[-1]) != old_vocab:
        raise ValueError(f"lm_head vocab mismatch: embed={old_vocab}, lm_head={int(head.shape[-1])}")

    if new_vocab < old_vocab:
        embed_resized = jnp.asarray(embed)[:new_vocab, :]
        head_resized = jnp.asarray(head)[:, :new_vocab]

        out: dict[str, Any] = dict(params)
        model = dict(_get_nested(params, ("model",)))
        embed_tokens = dict(_get_nested(params, ("model", "embed_tokens")))
        embed_tokens["embedding"] = embed_resized
        model["embed_tokens"] = embed_tokens
        out["model"] = model

        lm_head = dict(_get_nested(params, ("lm_head",)))
        lm_head["kernel"] = head_resized
        out["lm_head"] = lm_head

        return out, VocabResizeResult(original_vocab_size=old_vocab, new_vocab_size=new_vocab, added_tokens=new_vocab - old_vocab)

    added = new_vocab - old_vocab
    hidden = int(embed.shape[1])
    if int(head.shape[0]) != hidden:
        raise ValueError(f"lm_head hidden mismatch: embed_hidden={hidden}, lm_head_in={int(head.shape[0])}")

    rng_embed, rng_head = jax.random.split(rng, 2)
    new_embed = jax.random.normal(rng_embed, (added, hidden), dtype=jnp.float32) * float(init_std)
    new_head = jax.random.normal(rng_head, (hidden, added), dtype=jnp.float32) * float(init_std)

    embed_resized = jnp.concatenate([jnp.asarray(embed), new_embed.astype(jnp.asarray(embed).dtype)], axis=0)
    head_resized = jnp.concatenate([jnp.asarray(head), new_head.astype(jnp.asarray(head).dtype)], axis=1)

    # Avoid mutating the input pytree: copy only the dict nodes along the modified paths.
    out: dict[str, Any] = dict(params)
    model = dict(_get_nested(params, ("model",)))
    embed_tokens = dict(_get_nested(params, ("model", "embed_tokens")))
    embed_tokens["embedding"] = embed_resized
    model["embed_tokens"] = embed_tokens
    out["model"] = model

    lm_head = dict(_get_nested(params, ("lm_head",)))
    lm_head["kernel"] = head_resized
    out["lm_head"] = lm_head

    return out, VocabResizeResult(original_vocab_size=old_vocab, new_vocab_size=new_vocab, added_tokens=added)


__all__ = ["VocabResizeResult", "resize_lm_vocab"]
