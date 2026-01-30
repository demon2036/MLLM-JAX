from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jnp

from MLLM_JAX.language.qwen2.configuration_qwen2 import init_cache, pad_cache_right

from plugins.sample.decoding.sid3_constrained_beam_search import BeamSearchOutput


def _repeat_for_beams(tree: Any, repeats: int) -> Any:
    return jax.tree_util.tree_map(lambda x: jnp.repeat(x, repeats, axis=0), tree)


def _gather_beams(tree: Any, parent: jax.Array) -> Any:
    """Gather beam axis=1 using per-sample parent indices."""

    def gather_leaf(x: jax.Array) -> jax.Array:
        # x: [B, K, ...], parent: [B, K]
        return jax.vmap(lambda xb, pb: xb[pb])(x, parent)

    return jax.tree_util.tree_map(gather_leaf, tree)


def beam_search_sid3_prefill(
    *,
    model: Any,
    params: Any,
    prompt_input_ids: jax.Array,  # int32 [B, L_prefill]
    num_beams: int,
    max_cache_length: int,
    eos_token_id: int,
    suffix_token_ids: Sequence[int] | None = None,
    prompt_true_len: jax.Array | None = None,
) -> BeamSearchOutput:
    """Pure beam search for 3-token SIDs using sampler-style prefill buckets.

    This matches `constrained_beam_search_sid3_prefill`, but does not apply the
    SID trie constraints (i.e., searches over the full vocab at each step).
    """

    bsz, prefill_len = prompt_input_ids.shape
    k = int(num_beams)
    if k <= 0:
        raise ValueError("num_beams must be > 0")

    true_len = prompt_true_len
    if true_len is None:
        true_len = jnp.full((int(bsz),), int(prefill_len), dtype=jnp.int32)
    else:
        true_len = jnp.asarray(true_len, dtype=jnp.int32)
        if true_len.ndim == 0:
            true_len = jnp.full((int(bsz),), true_len, dtype=jnp.int32)
        elif true_len.ndim != 1:
            raise ValueError(f"prompt_true_len must be scalar or rank-1, got shape={true_len.shape}")

    suffix = [int(x) for x in (suffix_token_ids or [])]
    suffix.append(int(eos_token_id))
    max_pos = int(prefill_len) + 2 + len(suffix)
    if int(max_cache_length) <= max_pos:
        raise ValueError(
            f"max_cache_length too small: need > {max_pos} (prefill_len={int(prefill_len)}, suffix_len={len(suffix)})"
        )

    prompt_mask = (jnp.arange(int(prefill_len), dtype=jnp.int32)[None, :] < true_len[:, None]).astype(jnp.int32)
    position_ids = prompt_mask.cumsum(-1) - 1
    position_ids = jnp.where(prompt_mask == 0, 1, position_ids).astype(jnp.int32)

    cache = init_cache(model.config, int(bsz), max_cache_length=int(prefill_len), dtype=jnp.bfloat16)
    logits, cache = model.apply(
        {"params": params},
        input_ids=prompt_input_ids,
        position_ids=position_ids,
        attention_mask=prompt_mask,
        cache=cache,
    )

    extra_cache = int(max_cache_length) - int(prefill_len)
    if extra_cache < 0:
        raise ValueError(f"max_cache_length too small: {int(max_cache_length)} < prefill_len={int(prefill_len)}")
    if extra_cache:
        cache = pad_cache_right(cache, int(prefill_len), int(extra_cache))
        base_mask = jnp.pad(prompt_mask, ((0, 0), (0, int(extra_cache))), constant_values=0)
    else:
        base_mask = prompt_mask

    base_mask_k = jnp.repeat(base_mask, k, axis=0)  # [B*K, max_cache_length]
    pos_axis = jnp.arange(int(max_cache_length), dtype=jnp.int32)[None, :]

    def _step_mask(decode_end_index: int) -> jax.Array:
        decode = ((pos_axis >= int(prefill_len)) & (pos_axis <= int(decode_end_index))).astype(jnp.int32)
        return jnp.maximum(base_mask_k, decode)

    # --- Step 0: token_1 ---
    idx0 = jnp.clip(true_len - jnp.asarray(1, dtype=jnp.int32), 0, int(prefill_len) - 1).astype(jnp.int32)
    next_logits = logits[jnp.arange(int(bsz), dtype=jnp.int32), idx0]  # [B, V]
    log_probs0 = jax.nn.log_softmax(next_logits.astype(jnp.float32), axis=-1)
    vocab = int(log_probs0.shape[1])
    if k > vocab:
        raise ValueError(f"num_beams={k} exceeds vocab={vocab}")
    top0_scores, tok1 = jax.lax.top_k(log_probs0, k=k)  # [B, K]

    cache_k = _repeat_for_beams(cache, k)  # [B*K, ...]
    tok1_flat = tok1.reshape((bsz * k,))
    true_len_flat = jnp.repeat(true_len, k, axis=0).astype(jnp.int32)
    pos1 = true_len_flat[:, None]
    logits1, cache1 = model.apply(
        {"params": params},
        input_ids=tok1_flat[:, None],
        position_ids=pos1,
        attention_mask=_step_mask(int(prefill_len)),
        cache=cache_k,
    )
    log_probs1 = jax.nn.log_softmax(logits1[:, -1, :].astype(jnp.float32), axis=-1)  # [B*K, V]

    # Top-K candidates per beam, then global top-K per sample.
    top1_local_scores, tok2_local = jax.lax.top_k(log_probs1, k=k)  # [B*K, K]
    scores1_local = top0_scores.reshape((bsz * k, 1)) + top1_local_scores  # [B*K, K]
    scores1 = scores1_local.reshape((bsz, k * k))
    top1_scores, top1_idx = jax.lax.top_k(scores1, k=k)  # [B, K]
    parent1 = (top1_idx // k).astype(jnp.int32)
    off1 = (top1_idx % k).astype(jnp.int32)

    tok1_sel = jnp.take_along_axis(tok1, parent1, axis=1)
    tok2_local_3d = tok2_local.reshape((bsz, k, k))

    def _select_child(children_b: jax.Array, parent_b: jax.Array, off_b: jax.Array) -> jax.Array:
        chosen = children_b[parent_b]  # [K, K]
        return jnp.take_along_axis(chosen, off_b[:, None], axis=1)[:, 0]

    tok2_sel = jax.vmap(_select_child)(tok2_local_3d, parent1, off1)

    cache1_reshaped = jax.tree_util.tree_map(lambda x: x.reshape((bsz, k) + x.shape[1:]), cache1)
    cache1_sel = _gather_beams(cache1_reshaped, parent1)
    cache1_sel_flat = jax.tree_util.tree_map(lambda x: x.reshape((bsz * k,) + x.shape[2:]), cache1_sel)

    # --- Step 1: token_2 -> token_3 ---
    tok2_flat = tok2_sel.reshape((bsz * k,))
    pos2 = (true_len_flat + jnp.asarray(1, dtype=jnp.int32))[:, None]
    logits2, cache2 = model.apply(
        {"params": params},
        input_ids=tok2_flat[:, None],
        position_ids=pos2,
        attention_mask=_step_mask(int(prefill_len) + 1),
        cache=cache1_sel_flat,
    )
    log_probs2 = jax.nn.log_softmax(logits2[:, -1, :].astype(jnp.float32), axis=-1)  # [B*K, V]

    top2_local_scores, tok3_local = jax.lax.top_k(log_probs2, k=k)  # [B*K, K]
    scores2_local = top1_scores.reshape((bsz * k, 1)) + top2_local_scores  # [B*K, K]
    scores2 = scores2_local.reshape((bsz, k * k))
    top2_scores, top2_idx = jax.lax.top_k(scores2, k=k)  # [B, K]
    parent2 = (top2_idx // k).astype(jnp.int32)
    off2 = (top2_idx % k).astype(jnp.int32)

    tok1_final = jnp.take_along_axis(tok1_sel, parent2, axis=1)
    tok2_final = jnp.take_along_axis(tok2_sel, parent2, axis=1)
    tok3_local_3d = tok3_local.reshape((bsz, k, k))
    tok3_final = jax.vmap(_select_child)(tok3_local_3d, parent2, off2)

    cache2_reshaped = jax.tree_util.tree_map(lambda x: x.reshape((bsz, k) + x.shape[1:]), cache2)
    cache2_sel = _gather_beams(cache2_reshaped, parent2)
    cache2_sel_flat = jax.tree_util.tree_map(lambda x: x.reshape((bsz * k,) + x.shape[2:]), cache2_sel)

    # --- Score deterministic suffix tokens + EOS (align with HF generate) ---
    tok3_flat = tok3_final.reshape((bsz * k,))
    pos3 = (true_len_flat + jnp.asarray(2, dtype=jnp.int32))[:, None]
    logits3, cache3 = model.apply(
        {"params": params},
        input_ids=tok3_flat[:, None],
        position_ids=pos3,
        attention_mask=_step_mask(int(prefill_len) + 2),
        cache=cache2_sel_flat,
    )

    scores_flat = top2_scores.reshape((bsz * k,))
    log_probs = jax.nn.log_softmax(logits3[:, -1, :].astype(jnp.float32), axis=-1)
    cache_cur = cache3
    for i, token_id in enumerate(suffix):
        scores_flat = scores_flat + log_probs[:, int(token_id)]
        if i == len(suffix) - 1:
            break
        pos = (true_len_flat + jnp.asarray(3 + i, dtype=jnp.int32))[:, None]
        logits_next, cache_next = model.apply(
            {"params": params},
            input_ids=jnp.full((bsz * k, 1), int(token_id), dtype=jnp.int32),
            position_ids=pos,
            attention_mask=_step_mask(int(prefill_len) + 3 + i),
            cache=cache_cur,
        )
        cache_cur = cache_next
        log_probs = jax.nn.log_softmax(logits_next[:, -1, :].astype(jnp.float32), axis=-1)

    final_scores = scores_flat.reshape((bsz, k))
    sorted_scores, sorted_idx = jax.lax.top_k(final_scores, k=k)

    tokens = jnp.stack([tok1_final, tok2_final, tok3_final], axis=-1).astype(jnp.int32)
    idx = jnp.broadcast_to(sorted_idx[..., None], tokens.shape)
    tokens_sorted = jnp.take_along_axis(tokens, idx, axis=1)
    return BeamSearchOutput(token_ids=tokens_sorted, scores=sorted_scores)


__all__ = ["beam_search_sid3_prefill"]

