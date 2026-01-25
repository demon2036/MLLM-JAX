from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import jax
import jax.numpy as jnp

from MLLM_JAX.language.qwen2.configuration_qwen2 import init_cache, pad_cache

from plugins.sample.constraints.sid_trie import SidTrie


@dataclass(frozen=True)
class BeamSearchOutput:
    token_ids: jax.Array  # int32 [B, K, 3]
    scores: jax.Array  # float32 [B, K]


def _repeat_for_beams(tree: Any, repeats: int) -> Any:
    return jax.tree_util.tree_map(lambda x: jnp.repeat(x, repeats, axis=0), tree)


def _gather_beams(tree: Any, parent: jax.Array) -> Any:
    """Gather beam axis=1 using per-sample parent indices."""

    def gather_leaf(x: jax.Array) -> jax.Array:
        # x: [B, K, ...], parent: [B, K]
        return jax.vmap(lambda xb, pb: xb[pb])(x, parent)

    return jax.tree_util.tree_map(gather_leaf, tree)


def constrained_beam_search_sid3(
    *,
    model: Any,
    params: Any,
    prompt_input_ids: jax.Array,  # int32 [B, L]
    trie: SidTrie,
    num_beams: int,
    max_cache_length: int,
    suffix_token_ids: Sequence[int] | None = None,
    prompt_true_len: jax.Array | None = None,
) -> BeamSearchOutput:
    """Constrained beam search for 3-token SIDs.

    Assumes prompts already end with `### Response:\\n` (MiniOneRec format), so we only
    generate SID tokens (t1,t2,t3) and stop (EOS is implicit for scoring/metrics).
    """
    bsz, prompt_len = prompt_input_ids.shape
    has_prompt_true_len = prompt_true_len is not None

    true_len = prompt_true_len
    if true_len is None:
        true_len = jnp.asarray(int(prompt_len), dtype=jnp.int32)
    else:
        true_len = jnp.asarray(true_len, dtype=jnp.int32)

    first_ids = jnp.asarray(trie.first_ids, dtype=jnp.int32)
    second_table = jnp.asarray(trie.second_table, dtype=jnp.int32)
    third_table = jnp.asarray(trie.third_table, dtype=jnp.int32)
    pad_id = int(trie.pad_id)

    n1 = int(first_ids.shape[0])
    k = int(num_beams)
    if k <= 0:
        raise ValueError("num_beams must be > 0")
    if n1 <= 0:
        raise ValueError("SID trie has no first-token ids")

    suffix = [int(x) for x in (suffix_token_ids or [])]
    suffix.append(int(trie.eos_token_id))
    max_pos = int(prompt_len) + 2 + len(suffix)
    if not has_prompt_true_len:
        if int(max_cache_length) <= max_pos:
            raise ValueError(
                f"max_cache_length too small: need > {max_pos} (prompt_len={int(prompt_len)}, suffix_len={len(suffix)})"
            )
    elif int(max_cache_length) < int(prompt_len):
        raise ValueError(
            f"max_cache_length too small: need >= {int(prompt_len)} (prompt_len={int(prompt_len)}) when using prompt_true_len"
        )

    def _pad_cache_variable(cache_in: Any, prefill_length: int, target_cache_length: int, end_index: jax.Array):
        """Like `pad_cache`, but supports per-sample end_index (vector)."""
        end_index = jnp.asarray(end_index, dtype=jnp.int32)
        if int(target_cache_length) == int(prefill_length):
            for i in range(len(cache_in)):
                cache_in[f"layer_{i}"]["end_index"] = end_index
            return cache_in

        for i in range(len(cache_in)):
            cache_in[f"layer_{i}"]["k"] = jnp.pad(
                cache_in[f"layer_{i}"]["k"],
                ((0, 0), (0, 0), (0, int(target_cache_length) - int(prefill_length)), (0, 0)),
                constant_values=0,
            )
            cache_in[f"layer_{i}"]["v"] = jnp.pad(
                cache_in[f"layer_{i}"]["v"],
                ((0, 0), (0, 0), (0, int(target_cache_length) - int(prefill_length)), (0, 0)),
                constant_values=0,
            )
            cache_in[f"layer_{i}"]["end_index"] = end_index
        return cache_in

    # --- Prefill ---
    attention_mask = jnp.ones((bsz, prompt_len), dtype=jnp.int32)
    position_ids = jnp.arange(prompt_len, dtype=jnp.int32)[None, :]
    # Prefill cache must match prompt length for the model's n>1 attention mask path.
    cache = init_cache(model.config, int(bsz), max_cache_length=int(prompt_len), dtype=jnp.bfloat16)
    logits, cache = model.apply(
        {"params": params},
        input_ids=prompt_input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        cache=cache,
    )
    # Reset cache end_index to the true (unpadded) prompt length so decoding
    # overwrites any right-padding slots (and masks them out via step masks).
    if true_len.ndim == 0:
        cache = pad_cache(cache, int(prompt_len), int(max_cache_length), true_len)
    elif true_len.ndim == 1:
        cache = _pad_cache_variable(cache, int(prompt_len), int(max_cache_length), true_len)
    else:
        raise ValueError(f"prompt_true_len must be scalar or rank-1, got shape={true_len.shape}")

    # Grab logits at the last *true* prompt token (not the right-padding).
    if true_len.ndim == 0:
        idx0 = jnp.clip(true_len - jnp.asarray(1, dtype=jnp.int32), 0, int(prompt_len) - 1)
        next_logits = jnp.take(logits, idx0, axis=1)  # [B, V]
    else:
        idx0 = jnp.clip(true_len - jnp.asarray(1, dtype=jnp.int32), 0, int(prompt_len) - 1).astype(jnp.int32)
        next_logits = logits[jnp.arange(bsz, dtype=jnp.int32), idx0]  # [B, V]
    log_probs0 = jax.nn.log_softmax(next_logits.astype(jnp.float32), axis=-1)
    # Step0 needs to support `num_beams > len(first_ids)` (e.g. Industrial has fewer
    # unique <a_*> tokens than beam width). Match Transformers by doing a masked
    # full-vocab top-k when needed.
    if n1 < k:
        vocab = int(log_probs0.shape[1])
        first_mask = jnp.zeros((vocab,), dtype=jnp.bool_)
        first_mask = first_mask.at[first_ids].set(True)
        masked0 = jnp.where(first_mask[None, :], log_probs0, -jnp.inf)
        top0_scores, tok1 = jax.lax.top_k(masked0, k=k)  # tok1 are token ids

        # Map token id -> row index in `first_ids` (sorted).
        row = jnp.searchsorted(first_ids, tok1)
        row = jnp.clip(row, 0, n1 - 1)
        row_tok = first_ids[row]
        tok1_valid = tok1 == row_tok
        tok1_row = row.astype(jnp.int32)
        top0_scores = jnp.where(tok1_valid, top0_scores, -jnp.inf)
    else:
        log_probs0_allowed = jnp.take(log_probs0, first_ids, axis=1)  # [B, N1]
        top0_scores, top0_idx = jax.lax.top_k(log_probs0_allowed, k=k)  # [B, K]
        tok1 = jnp.take(first_ids, top0_idx, axis=0)  # [B, K]
        tok1_row = top0_idx.astype(jnp.int32)  # row index into second/third tables
        tok1_valid = jnp.ones_like(tok1_row, dtype=jnp.bool_)

    # Repeat cache for beams.
    cache_k = _repeat_for_beams(cache, k)  # [B*K, ...]

    # --- Step 1: score token_2 (after updating cache with token_1) ---
    tok1_flat = tok1.reshape((bsz * k,))
    if true_len.ndim == 0:
        pos1 = jnp.zeros((bsz * k, 1), dtype=jnp.int32) + true_len
        # Mask includes the new slot at `true_len` for the token we insert.
        step1_mask = (jnp.arange(int(max_cache_length), dtype=jnp.int32) <= true_len).astype(jnp.int32)
        step1_mask = jnp.broadcast_to(step1_mask[None, :], (bsz * k, int(max_cache_length)))
    else:
        true_len_flat = jnp.repeat(true_len, k, axis=0).astype(jnp.int32)
        pos1 = true_len_flat[:, None]
        step1_mask = (
            jnp.arange(int(max_cache_length), dtype=jnp.int32)[None, :] <= true_len_flat[:, None]
        ).astype(jnp.int32)
    logits1, cache1 = model.apply(
        {"params": params},
        input_ids=tok1_flat[:, None],
        position_ids=pos1,
        attention_mask=step1_mask,
        cache=cache_k,
    )
    log_probs1 = jax.nn.log_softmax(logits1[:, -1, :].astype(jnp.float32), axis=-1)  # [B*K, V]

    # Allowed token_2 lists keyed by token_1 row index (top0_idx already indexes second_table).
    tok1_valid_flat = tok1_valid.reshape((bsz * k,))
    allowed2 = jnp.take(second_table, tok1_row.reshape((bsz * k,)), axis=0)  # [B*K, M2]
    allowed2 = jnp.where(tok1_valid_flat[:, None], allowed2, jnp.full_like(allowed2, int(pad_id)))
    valid2 = allowed2 != int(pad_id)
    safe2 = jnp.where(valid2, allowed2, 0).astype(jnp.int32)
    lp2 = jnp.take_along_axis(log_probs1, safe2, axis=1)
    lp2 = jnp.where(valid2, lp2, -jnp.inf)

    scores1 = top0_scores.reshape((bsz * k, 1)) + lp2  # [B*K, M2]
    m2 = int(scores1.shape[1])
    scores1 = scores1.reshape((bsz, k * m2))
    top1_scores, top1_idx = jax.lax.top_k(scores1, k=k)  # [B, K]
    parent1 = (top1_idx // m2).astype(jnp.int32)  # [B, K]
    off1 = (top1_idx % m2).astype(jnp.int32)  # [B, K]

    # Gather token_1 and token_2 for selected beams.
    tok1_sel = jnp.take_along_axis(tok1, parent1, axis=1)  # [B, K]
    tok1_row_sel = jnp.take_along_axis(tok1_row, parent1, axis=1)  # [B, K]
    tok2_col_sel = off1  # [B, K]
    allowed2_3d = allowed2.reshape((bsz, k, m2))

    def _select_tok2(allowed_b, parent_b, off_b):
        chosen = allowed_b[parent_b]  # [K, M2]
        return jnp.take_along_axis(chosen, off_b[:, None], axis=1)[:, 0]

    tok2_sel = jax.vmap(_select_tok2)(allowed2_3d, parent1, off1)

    # Gather caches after token_1 for selected beams (may duplicate parents).
    cache1_reshaped = jax.tree_util.tree_map(lambda x: x.reshape((bsz, k) + x.shape[1:]), cache1)
    cache1_sel = _gather_beams(cache1_reshaped, parent1)
    cache1_sel_flat = jax.tree_util.tree_map(lambda x: x.reshape((bsz * k,) + x.shape[2:]), cache1_sel)

    # --- Step 2: score token_3 ---
    tok2_flat = tok2_sel.reshape((bsz * k,))
    if true_len.ndim == 0:
        pos2 = jnp.zeros((bsz * k, 1), dtype=jnp.int32) + (true_len + jnp.asarray(1, dtype=jnp.int32))
        step2_mask = (jnp.arange(int(max_cache_length), dtype=jnp.int32) <= (true_len + jnp.asarray(1, dtype=jnp.int32))).astype(
            jnp.int32
        )
        step2_mask = jnp.broadcast_to(step2_mask[None, :], (bsz * k, int(max_cache_length)))
    else:
        true_len_flat = jnp.repeat(true_len, k, axis=0).astype(jnp.int32)
        pos2 = (true_len_flat + jnp.asarray(1, dtype=jnp.int32))[:, None]
        step2_mask = (
            jnp.arange(int(max_cache_length), dtype=jnp.int32)[None, :]
            <= (true_len_flat[:, None] + jnp.asarray(1, dtype=jnp.int32))
        ).astype(jnp.int32)
    logits2, _cache2 = model.apply(
        {"params": params},
        input_ids=tok2_flat[:, None],
        position_ids=pos2,
        attention_mask=step2_mask,
        cache=cache1_sel_flat,
    )
    log_probs2 = jax.nn.log_softmax(logits2[:, -1, :].astype(jnp.float32), axis=-1)  # [B*K, V]

    tok1_row_flat = tok1_row_sel.reshape((bsz * k,))
    tok2_col_flat = tok2_col_sel.reshape((bsz * k,))
    allowed3 = third_table[tok1_row_flat, tok2_col_flat]  # [B*K, M3]
    valid3 = allowed3 != int(pad_id)
    safe3 = jnp.where(valid3, allowed3, 0).astype(jnp.int32)
    lp3 = jnp.take_along_axis(log_probs2, safe3, axis=1)
    lp3 = jnp.where(valid3, lp3, -jnp.inf)

    scores2 = top1_scores.reshape((bsz * k, 1)) + lp3  # [B*K, M3]
    m3 = int(scores2.shape[1])
    scores2 = scores2.reshape((bsz, k * m3))
    top2_scores, top2_idx = jax.lax.top_k(scores2, k=k)  # [B, K]
    parent2 = (top2_idx // m3).astype(jnp.int32)  # [B, K]
    off2 = (top2_idx % m3).astype(jnp.int32)  # [B, K]

    tok1_final = jnp.take_along_axis(tok1_sel, parent2, axis=1)
    tok2_final = jnp.take_along_axis(tok2_sel, parent2, axis=1)
    allowed3_3d = allowed3.reshape((bsz, k, m3))

    def _select_tok3(allowed_b, parent_b, off_b):
        chosen = allowed_b[parent_b]  # [K, M3]
        return jnp.take_along_axis(chosen, off_b[:, None], axis=1)[:, 0]

    tok3_final = jax.vmap(_select_tok3)(allowed3_3d, parent2, off2)

    # --- Score deterministic suffix tokens + EOS (align with HF generate) ---
    cache2 = _cache2
    cache2_reshaped = jax.tree_util.tree_map(lambda x: x.reshape((bsz, k) + x.shape[1:]), cache2)
    cache2_sel = _gather_beams(cache2_reshaped, parent2)
    cache2_sel_flat = jax.tree_util.tree_map(lambda x: x.reshape((bsz * k,) + x.shape[2:]), cache2_sel)

    tok3_flat = tok3_final.reshape((bsz * k,))
    if true_len.ndim == 0:
        pos3 = jnp.zeros((bsz * k, 1), dtype=jnp.int32) + (true_len + jnp.asarray(2, dtype=jnp.int32))
        step3_mask = (jnp.arange(int(max_cache_length), dtype=jnp.int32) <= (true_len + jnp.asarray(2, dtype=jnp.int32))).astype(
            jnp.int32
        )
        step3_mask = jnp.broadcast_to(step3_mask[None, :], (bsz * k, int(max_cache_length)))
    else:
        true_len_flat = jnp.repeat(true_len, k, axis=0).astype(jnp.int32)
        pos3 = (true_len_flat + jnp.asarray(2, dtype=jnp.int32))[:, None]
        step3_mask = (
            jnp.arange(int(max_cache_length), dtype=jnp.int32)[None, :]
            <= (true_len_flat[:, None] + jnp.asarray(2, dtype=jnp.int32))
        ).astype(jnp.int32)
    logits3, cache3 = model.apply(
        {"params": params},
        input_ids=tok3_flat[:, None],
        position_ids=pos3,
        attention_mask=step3_mask,
        cache=cache2_sel_flat,
    )

    scores_flat = top2_scores.reshape((bsz * k,))
    log_probs = jax.nn.log_softmax(logits3[:, -1, :].astype(jnp.float32), axis=-1)
    cache_cur = cache3
    for i, token_id in enumerate(suffix):
        scores_flat = scores_flat + log_probs[:, int(token_id)]
        if i == len(suffix) - 1:
            break
        if true_len.ndim == 0:
            pos = jnp.zeros((bsz * k, 1), dtype=jnp.int32) + (true_len + jnp.asarray(3 + i, dtype=jnp.int32))
            step_mask = (jnp.arange(int(max_cache_length), dtype=jnp.int32) <= (true_len + jnp.asarray(3 + i, dtype=jnp.int32))).astype(
                jnp.int32
            )
            step_mask = jnp.broadcast_to(step_mask[None, :], (bsz * k, int(max_cache_length)))
        else:
            true_len_flat = jnp.repeat(true_len, k, axis=0).astype(jnp.int32)
            pos = (true_len_flat + jnp.asarray(3 + i, dtype=jnp.int32))[:, None]
            step_mask = (
                jnp.arange(int(max_cache_length), dtype=jnp.int32)[None, :]
                <= (true_len_flat[:, None] + jnp.asarray(3 + i, dtype=jnp.int32))
            ).astype(jnp.int32)
        logits_next, cache_next = model.apply(
            {"params": params},
            input_ids=jnp.full((bsz * k, 1), int(token_id), dtype=jnp.int32),
            position_ids=pos,
            attention_mask=step_mask,
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


__all__ = ["BeamSearchOutput", "constrained_beam_search_sid3"]
