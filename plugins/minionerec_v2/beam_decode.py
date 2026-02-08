from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import jax
import jax.numpy as jnp

from MLLM_JAX.language.qwen2.configuration_qwen2 import init_cache, pad_cache, pad_cache_right

from .constraints import SidTrie


@dataclass(frozen=True)
class BeamSearchOutput:
    token_ids: jax.Array
    scores: jax.Array


def _repeat_for_beams(tree: Any, repeats: int) -> Any:
    return jax.tree_util.tree_map(lambda x: jnp.repeat(x, repeats, axis=0), tree)


def _gather_beams(tree: Any, parent: jax.Array) -> Any:
    def gather_leaf(x: jax.Array) -> jax.Array:
        return jax.vmap(lambda xb, pb: xb[pb])(x, parent)

    return jax.tree_util.tree_map(gather_leaf, tree)


def _select_beam_candidates(
    scores: jax.Array,
    k: int,
    *,
    do_sample: bool,
    temperature: float,
    rng_key: jax.Array | None,
) -> tuple[jax.Array, jax.Array]:
    if not bool(do_sample):
        return jax.lax.top_k(scores, k=int(k))

    if rng_key is None:
        raise ValueError("rng_key is required when do_sample=True")

    temp = float(temperature)
    if temp <= 0.0:
        raise ValueError(f"temperature must be > 0, got: {temp}")

    scaled_scores = scores / jnp.asarray(temp, dtype=scores.dtype)
    row_keys = jax.random.split(rng_key, int(scores.shape[0]))

    def _sample_row(row_scaled: jax.Array, row_scores: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        finite_mask = jnp.isfinite(row_scaled)
        n_valid = jnp.sum(finite_mask.astype(jnp.int32))

        def _sample_without_replacement(_: None) -> jax.Array:
            safe_scores = jnp.where(finite_mask, row_scaled, -jnp.inf)
            gumbel = jax.random.gumbel(key, shape=row_scaled.shape, dtype=row_scaled.dtype)
            _, sampled_idx = jax.lax.top_k(safe_scores + gumbel, k=int(k))
            sampled_scores = jnp.take_along_axis(row_scores, sampled_idx, axis=0)
            order = jnp.argsort(sampled_scores, axis=0)[::-1]
            return jnp.take_along_axis(sampled_idx, order, axis=0).astype(jnp.int32)

        def _fallback_topk(_: None) -> jax.Array:
            _, fallback_idx = jax.lax.top_k(row_scores, k=int(k))
            return fallback_idx.astype(jnp.int32)

        selected_idx = jax.lax.cond(
            n_valid >= int(k),
            _sample_without_replacement,
            _fallback_topk,
            operand=None,
        )
        selected_scores = jnp.take_along_axis(row_scores, selected_idx, axis=0)
        return selected_scores, selected_idx

    selected_scores, selected_idx = jax.vmap(_sample_row)(scaled_scores, scores, row_keys)
    return selected_scores, selected_idx


def constrained_beam_search_sid3(
    *,
    model: Any,
    params: Any,
    prompt_input_ids: jax.Array,
    trie: SidTrie,
    num_beams: int,
    max_cache_length: int,
    eos_token_id: int,
    suffix_token_ids: Sequence[int] | None = None,
    prompt_true_len: jax.Array | None = None,
    do_sample: bool = False,
    temperature: float = 1.0,
    rng_key: jax.Array | None = None,
) -> BeamSearchOutput:
    bsz, prompt_len = prompt_input_ids.shape
    has_prompt_true_len = prompt_true_len is not None

    true_len = prompt_true_len
    if true_len is None:
        true_len = jnp.asarray(int(prompt_len), dtype=jnp.int32)
    else:
        true_len = jnp.asarray(true_len, dtype=jnp.int32)
        if true_len.ndim != 0:
            raise ValueError(
                "constrained_beam_search_sid3 requires scalar prompt_true_len shared across the batch; "
                "use constrained_beam_search_sid3_prefill for mixed lengths."
            )

    first_ids = jnp.asarray(trie.first_ids, dtype=jnp.int32)
    second_table = jnp.asarray(trie.second_table, dtype=jnp.int32)
    third_table = jnp.asarray(trie.third_table, dtype=jnp.int32)
    pad_id = int(trie.pad_id)

    n1 = int(first_ids.shape[0])
    k = int(num_beams)
    sample_mode = bool(do_sample)
    temperature_f = float(temperature)
    if k <= 0:
        raise ValueError("num_beams must be > 0")
    if n1 <= 0:
        raise ValueError("SID trie has no first-token ids")
    if temperature_f <= 0.0:
        raise ValueError(f"temperature must be > 0, got: {temperature_f}")
    if sample_mode and rng_key is None:
        raise ValueError("rng_key is required when do_sample=True")

    sample_stage_keys: list[jax.Array | None] = [None, None, None, None]
    if sample_mode:
        assert rng_key is not None
        split_keys = jax.random.split(rng_key, 4)
        sample_stage_keys = [split_keys[0], split_keys[1], split_keys[2], split_keys[3]]

    suffix = [int(x) for x in (suffix_token_ids or [])]
    suffix.append(int(eos_token_id))
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

    attention_mask = jnp.ones((bsz, prompt_len), dtype=jnp.int32)
    position_ids = jnp.arange(prompt_len, dtype=jnp.int32)[None, :]
    cache = init_cache(model.config, int(bsz), max_cache_length=int(prompt_len), dtype=jnp.bfloat16)
    logits, cache = model.apply(
        {"params": params},
        input_ids=prompt_input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        cache=cache,
    )
    if true_len.ndim == 0:
        cache = pad_cache(cache, int(prompt_len), int(max_cache_length), true_len)
    elif true_len.ndim == 1:
        cache = _pad_cache_variable(cache, int(prompt_len), int(max_cache_length), true_len)
    else:
        raise ValueError(f"prompt_true_len must be scalar or rank-1, got shape={true_len.shape}")

    if true_len.ndim == 0:
        idx0 = jnp.clip(true_len - jnp.asarray(1, dtype=jnp.int32), 0, int(prompt_len) - 1)
        next_logits = jnp.take(logits, idx0, axis=1)
    else:
        idx0 = jnp.clip(true_len - jnp.asarray(1, dtype=jnp.int32), 0, int(prompt_len) - 1).astype(jnp.int32)
        next_logits = logits[jnp.arange(bsz, dtype=jnp.int32), idx0]
    log_probs0 = jax.nn.log_softmax(next_logits.astype(jnp.float32), axis=-1)
    if n1 < k:
        vocab = int(log_probs0.shape[1])
        first_mask = jnp.zeros((vocab,), dtype=jnp.bool_)
        first_mask = first_mask.at[first_ids].set(True)
        masked0 = jnp.where(first_mask[None, :], log_probs0, -jnp.inf)
        top0_scores, tok1 = _select_beam_candidates(
            masked0,
            k,
            do_sample=sample_mode,
            temperature=temperature_f,
            rng_key=sample_stage_keys[0],
        )

        row = jnp.searchsorted(first_ids, tok1)
        row = jnp.clip(row, 0, n1 - 1)
        row_tok = first_ids[row]
        tok1_valid = tok1 == row_tok
        tok1_row = row.astype(jnp.int32)
        top0_scores = jnp.where(tok1_valid, top0_scores, -jnp.inf)
    else:
        log_probs0_allowed = jnp.take(log_probs0, first_ids, axis=1)
        top0_scores, top0_idx = _select_beam_candidates(
            log_probs0_allowed,
            k,
            do_sample=sample_mode,
            temperature=temperature_f,
            rng_key=sample_stage_keys[0],
        )
        tok1 = jnp.take(first_ids, top0_idx, axis=0)
        tok1_row = top0_idx.astype(jnp.int32)
        tok1_valid = jnp.ones_like(tok1_row, dtype=jnp.bool_)

    cache_k = _repeat_for_beams(cache, k)

    tok1_flat = tok1.reshape((bsz * k,))
    if true_len.ndim == 0:
        pos1 = jnp.zeros((bsz * k, 1), dtype=jnp.int32) + true_len
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
    log_probs1 = jax.nn.log_softmax(logits1[:, -1, :].astype(jnp.float32), axis=-1)

    tok1_valid_flat = tok1_valid.reshape((bsz * k,))
    allowed2 = jnp.take(second_table, tok1_row.reshape((bsz * k,)), axis=0)
    allowed2 = jnp.where(tok1_valid_flat[:, None], allowed2, jnp.full_like(allowed2, int(pad_id)))
    valid2 = allowed2 != int(pad_id)
    safe2 = jnp.where(valid2, allowed2, 0).astype(jnp.int32)
    lp2 = jnp.take_along_axis(log_probs1, safe2, axis=1)
    lp2 = jnp.where(valid2, lp2, -jnp.inf)

    scores1 = top0_scores.reshape((bsz * k, 1)) + lp2
    m2 = int(scores1.shape[1])
    scores1 = scores1.reshape((bsz, k * m2))
    top1_scores, top1_idx = _select_beam_candidates(
        scores1,
        k,
        do_sample=sample_mode,
        temperature=temperature_f,
        rng_key=sample_stage_keys[1],
    )
    parent1 = (top1_idx // m2).astype(jnp.int32)
    off1 = (top1_idx % m2).astype(jnp.int32)

    tok1_sel = jnp.take_along_axis(tok1, parent1, axis=1)
    tok1_row_sel = jnp.take_along_axis(tok1_row, parent1, axis=1)
    tok2_col_sel = off1
    allowed2_3d = allowed2.reshape((bsz, k, m2))

    def _select_tok2(allowed_b, parent_b, off_b):
        chosen = allowed_b[parent_b]
        return jnp.take_along_axis(chosen, off_b[:, None], axis=1)[:, 0]

    tok2_sel = jax.vmap(_select_tok2)(allowed2_3d, parent1, off1)

    cache1_reshaped = jax.tree_util.tree_map(lambda x: x.reshape((bsz, k) + x.shape[1:]), cache1)
    cache1_sel = _gather_beams(cache1_reshaped, parent1)
    cache1_sel_flat = jax.tree_util.tree_map(lambda x: x.reshape((bsz * k,) + x.shape[2:]), cache1_sel)

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
    log_probs2 = jax.nn.log_softmax(logits2[:, -1, :].astype(jnp.float32), axis=-1)

    tok1_row_flat = tok1_row_sel.reshape((bsz * k,))
    tok2_col_flat = tok2_col_sel.reshape((bsz * k,))
    allowed3 = third_table[tok1_row_flat, tok2_col_flat]
    valid3 = allowed3 != int(pad_id)
    safe3 = jnp.where(valid3, allowed3, 0).astype(jnp.int32)
    lp3 = jnp.take_along_axis(log_probs2, safe3, axis=1)
    lp3 = jnp.where(valid3, lp3, -jnp.inf)

    scores2 = top1_scores.reshape((bsz * k, 1)) + lp3
    m3 = int(scores2.shape[1])
    scores2 = scores2.reshape((bsz, k * m3))
    top2_scores, top2_idx = _select_beam_candidates(
        scores2,
        k,
        do_sample=sample_mode,
        temperature=temperature_f,
        rng_key=sample_stage_keys[2],
    )
    parent2 = (top2_idx // m3).astype(jnp.int32)
    off2 = (top2_idx % m3).astype(jnp.int32)

    tok1_final = jnp.take_along_axis(tok1_sel, parent2, axis=1)
    tok2_final = jnp.take_along_axis(tok2_sel, parent2, axis=1)
    allowed3_3d = allowed3.reshape((bsz, k, m3))

    def _select_tok3(allowed_b, parent_b, off_b):
        chosen = allowed_b[parent_b]
        return jnp.take_along_axis(chosen, off_b[:, None], axis=1)[:, 0]

    tok3_final = jax.vmap(_select_tok3)(allowed3_3d, parent2, off2)

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


def constrained_beam_search_sid3_prefill(
    *,
    model: Any,
    params: Any,
    prompt_input_ids: jax.Array,
    trie: SidTrie,
    num_beams: int,
    max_cache_length: int,
    eos_token_id: int,
    suffix_token_ids: Sequence[int] | None = None,
    prompt_true_len: jax.Array | None = None,
    do_sample: bool = False,
    temperature: float = 1.0,
    rng_key: jax.Array | None = None,
) -> BeamSearchOutput:
    bsz, prefill_len = prompt_input_ids.shape

    true_len = prompt_true_len
    if true_len is None:
        true_len = jnp.full((int(bsz),), int(prefill_len), dtype=jnp.int32)
    else:
        true_len = jnp.asarray(true_len, dtype=jnp.int32)
        if true_len.ndim == 0:
            true_len = jnp.full((int(bsz),), true_len, dtype=jnp.int32)
        elif true_len.ndim != 1:
            raise ValueError(f"prompt_true_len must be scalar or rank-1, got shape={true_len.shape}")

    first_ids = jnp.asarray(trie.first_ids, dtype=jnp.int32)
    second_table = jnp.asarray(trie.second_table, dtype=jnp.int32)
    third_table = jnp.asarray(trie.third_table, dtype=jnp.int32)
    pad_id = int(trie.pad_id)

    n1 = int(first_ids.shape[0])
    k = int(num_beams)
    sample_mode = bool(do_sample)
    temperature_f = float(temperature)
    if k <= 0:
        raise ValueError("num_beams must be > 0")
    if n1 <= 0:
        raise ValueError("SID trie has no first-token ids")
    if temperature_f <= 0.0:
        raise ValueError(f"temperature must be > 0, got: {temperature_f}")
    if sample_mode and rng_key is None:
        raise ValueError("rng_key is required when do_sample=True")

    sample_stage_keys: list[jax.Array | None] = [None, None, None, None]
    if sample_mode:
        assert rng_key is not None
        split_keys = jax.random.split(rng_key, 4)
        sample_stage_keys = [split_keys[0], split_keys[1], split_keys[2], split_keys[3]]

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

    base_mask_k = jnp.repeat(base_mask, k, axis=0)
    pos_axis = jnp.arange(int(max_cache_length), dtype=jnp.int32)[None, :]

    def _step_mask(decode_end_index: int) -> jax.Array:
        decode = ((pos_axis >= int(prefill_len)) & (pos_axis <= int(decode_end_index))).astype(jnp.int32)
        return jnp.maximum(base_mask_k, decode)

    idx0 = jnp.clip(true_len - jnp.asarray(1, dtype=jnp.int32), 0, int(prefill_len) - 1).astype(jnp.int32)
    next_logits = logits[jnp.arange(int(bsz), dtype=jnp.int32), idx0]
    log_probs0 = jax.nn.log_softmax(next_logits.astype(jnp.float32), axis=-1)

    if n1 < k:
        vocab = int(log_probs0.shape[1])
        first_mask = jnp.zeros((vocab,), dtype=jnp.bool_)
        first_mask = first_mask.at[first_ids].set(True)
        masked0 = jnp.where(first_mask[None, :], log_probs0, -jnp.inf)
        top0_scores, tok1 = _select_beam_candidates(
            masked0,
            k,
            do_sample=sample_mode,
            temperature=temperature_f,
            rng_key=sample_stage_keys[0],
        )

        row = jnp.searchsorted(first_ids, tok1)
        row = jnp.clip(row, 0, n1 - 1)
        row_tok = first_ids[row]
        tok1_valid = tok1 == row_tok
        tok1_row = row.astype(jnp.int32)
        top0_scores = jnp.where(tok1_valid, top0_scores, -jnp.inf)
    else:
        log_probs0_allowed = jnp.take(log_probs0, first_ids, axis=1)
        top0_scores, top0_idx = _select_beam_candidates(
            log_probs0_allowed,
            k,
            do_sample=sample_mode,
            temperature=temperature_f,
            rng_key=sample_stage_keys[0],
        )
        tok1 = jnp.take(first_ids, top0_idx, axis=0)
        tok1_row = top0_idx.astype(jnp.int32)
        tok1_valid = jnp.ones_like(tok1_row, dtype=jnp.bool_)

    cache_k = _repeat_for_beams(cache, k)
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
    log_probs1 = jax.nn.log_softmax(logits1[:, -1, :].astype(jnp.float32), axis=-1)

    tok1_valid_flat = tok1_valid.reshape((bsz * k,))
    allowed2 = jnp.take(second_table, tok1_row.reshape((bsz * k,)), axis=0)
    allowed2 = jnp.where(tok1_valid_flat[:, None], allowed2, jnp.full_like(allowed2, int(pad_id)))
    valid2 = allowed2 != int(pad_id)
    safe2 = jnp.where(valid2, allowed2, 0).astype(jnp.int32)
    lp2 = jnp.take_along_axis(log_probs1, safe2, axis=1)
    lp2 = jnp.where(valid2, lp2, -jnp.inf)

    scores1 = top0_scores.reshape((bsz * k, 1)) + lp2
    m2 = int(scores1.shape[1])
    scores1 = scores1.reshape((bsz, k * m2))
    top1_scores, top1_idx = _select_beam_candidates(
        scores1,
        k,
        do_sample=sample_mode,
        temperature=temperature_f,
        rng_key=sample_stage_keys[1],
    )
    parent1 = (top1_idx // m2).astype(jnp.int32)
    off1 = (top1_idx % m2).astype(jnp.int32)

    tok1_sel = jnp.take_along_axis(tok1, parent1, axis=1)
    tok1_row_sel = jnp.take_along_axis(tok1_row, parent1, axis=1)
    tok2_col_sel = off1
    allowed2_3d = allowed2.reshape((bsz, k, m2))

    def _select_tok2(allowed_b, parent_b, off_b):
        chosen = allowed_b[parent_b]
        return jnp.take_along_axis(chosen, off_b[:, None], axis=1)[:, 0]

    tok2_sel = jax.vmap(_select_tok2)(allowed2_3d, parent1, off1)

    cache1_reshaped = jax.tree_util.tree_map(lambda x: x.reshape((bsz, k) + x.shape[1:]), cache1)
    cache1_sel = _gather_beams(cache1_reshaped, parent1)
    cache1_sel_flat = jax.tree_util.tree_map(lambda x: x.reshape((bsz * k,) + x.shape[2:]), cache1_sel)

    tok2_flat = tok2_sel.reshape((bsz * k,))
    pos2 = (true_len_flat + jnp.asarray(1, dtype=jnp.int32))[:, None]
    logits2, _cache2 = model.apply(
        {"params": params},
        input_ids=tok2_flat[:, None],
        position_ids=pos2,
        attention_mask=_step_mask(int(prefill_len) + 1),
        cache=cache1_sel_flat,
    )
    log_probs2 = jax.nn.log_softmax(logits2[:, -1, :].astype(jnp.float32), axis=-1)

    tok1_row_flat = tok1_row_sel.reshape((bsz * k,))
    tok2_col_flat = tok2_col_sel.reshape((bsz * k,))
    allowed3 = third_table[tok1_row_flat, tok2_col_flat]
    valid3 = allowed3 != int(pad_id)
    safe3 = jnp.where(valid3, allowed3, 0).astype(jnp.int32)
    lp3 = jnp.take_along_axis(log_probs2, safe3, axis=1)
    lp3 = jnp.where(valid3, lp3, -jnp.inf)

    scores2 = top1_scores.reshape((bsz * k, 1)) + lp3
    m3 = int(scores2.shape[1])
    scores2 = scores2.reshape((bsz, k * m3))
    top2_scores, top2_idx = _select_beam_candidates(
        scores2,
        k,
        do_sample=sample_mode,
        temperature=temperature_f,
        rng_key=sample_stage_keys[2],
    )
    parent2 = (top2_idx // m3).astype(jnp.int32)
    off2 = (top2_idx % m3).astype(jnp.int32)

    tok1_final = jnp.take_along_axis(tok1_sel, parent2, axis=1)
    tok2_final = jnp.take_along_axis(tok2_sel, parent2, axis=1)
    allowed3_3d = allowed3.reshape((bsz, k, m3))

    def _select_tok3(allowed_b, parent_b, off_b):
        chosen = allowed_b[parent_b]
        return jnp.take_along_axis(chosen, off_b[:, None], axis=1)[:, 0]

    tok3_final = jax.vmap(_select_tok3)(allowed3_3d, parent2, off2)

    cache2 = _cache2
    cache2_reshaped = jax.tree_util.tree_map(lambda x: x.reshape((bsz, k) + x.shape[1:]), cache2)
    cache2_sel = _gather_beams(cache2_reshaped, parent2)
    cache2_sel_flat = jax.tree_util.tree_map(lambda x: x.reshape((bsz * k,) + x.shape[2:]), cache2_sel)

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


__all__ = ["BeamSearchOutput", "constrained_beam_search_sid3", "constrained_beam_search_sid3_prefill"]
