from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from plugins.sft.constrained_decoding import load_valid_sids_from_info
from plugins.sft.jax.beam_search import constrained_beam_search_sid3
from plugins.sft.jax.sid_trie import build_sid_trie_from_index
from plugins.sft.metrics import RankingMetrics, compute_hr_ndcg


def _decode_sid_triplet(tokenizer: Any, triplet: list[int]) -> str:
    toks = tokenizer.convert_ids_to_tokens([int(x) for x in triplet])
    return "".join(str(t) for t in toks)


def _newline_suffix_token_ids(tokenizer: Any) -> list[int]:
    """Token ids for the trailing '\\n' in completions, in non-initial context."""
    base = list(tokenizer.encode("a", add_special_tokens=False))
    with_nl = list(tokenizer.encode("a\n", add_special_tokens=False))
    lcp = 0
    for x, y in zip(base, with_nl, strict=False):
        if int(x) != int(y):
            break
        lcp += 1
    suffix = [int(x) for x in with_nl[lcp:]]
    return suffix


def evaluate_sid_next_item_jax(
    *,
    model: Any,
    params: Any,
    tokenizer: Any,
    eval_dataset: Any,
    sid_index_path: str,
    info_file: str,
    batch_size: int,
    num_beams: int,
    max_cache_length: int,
    topk: list[int],
    output_predictions_json: str | None,
) -> tuple[list[list[str]], RankingMetrics]:
    valid_sids = load_valid_sids_from_info(info_file)
    trie = build_sid_trie_from_index(tokenizer=tokenizer, sid_index_path=sid_index_path, eos_token_id=int(getattr(tokenizer, "eos_token_id")))

    n = int(len(eval_dataset))
    targets = list(getattr(eval_dataset, "get_targets")())
    if len(targets) != n:
        raise ValueError("eval_dataset.get_targets() length mismatch")

    newline_suffix = _newline_suffix_token_ids(tokenizer)

    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover - optional dep
        tqdm = None  # type: ignore[assignment]

    # Precompute prompt lengths to batch only equal-length prompts (KV-cache assumes shared prompt_len).
    prompt_lens = [len(eval_dataset[i]["input_ids"]) for i in range(n)]
    buckets: dict[int, list[int]] = {}
    for i, l in enumerate(prompt_lens):
        buckets.setdefault(int(l), []).append(i)

    predictions: list[list[str]] = [None] * n  # type: ignore[list-item]
    print(
        f"[eval] samples={n} buckets={len(buckets)} batch_size={int(batch_size)} "
        f"num_beams={int(num_beams)} max_cache_length={int(max_cache_length)}"
    )

    def _beam_search(params_in: Any, prompt_input_ids: jax.Array):
        out = constrained_beam_search_sid3(
            model=model,
            params=params_in,
            prompt_input_ids=prompt_input_ids,
            trie=trie,
            num_beams=int(num_beams),
            max_cache_length=int(max_cache_length),
            suffix_token_ids=newline_suffix,
        )
        return out.token_ids

    beam_search_jit = jax.jit(_beam_search)

    for prompt_len, idxs in sorted(buckets.items()):
        n_bucket = int(len(idxs))
        chunks = (n_bucket + int(batch_size) - 1) // int(batch_size)
        print(f"[eval] prompt_len={int(prompt_len)} bucket_samples={n_bucket} chunks={chunks}")

        compiled = False
        starts = range(0, len(idxs), int(batch_size))
        if tqdm is not None:
            starts = tqdm(starts, total=chunks, desc=f"eval len={int(prompt_len)}", mininterval=1.0)

        for start in starts:
            chunk = idxs[start : start + int(batch_size)]
            real_chunk = int(len(chunk))
            if real_chunk < int(batch_size):
                chunk = chunk + [chunk[-1]] * (int(batch_size) - real_chunk)

            prompt_ids = [eval_dataset[i]["input_ids"] for i in chunk]
            prompt = jnp.asarray(prompt_ids, dtype=jnp.int32)

            if not compiled:
                t0 = time.perf_counter()
                token_ids = beam_search_jit(params, prompt)
                tok_np = np.asarray(token_ids)  # triggers compilation + blocks
                dt = time.perf_counter() - t0
                compiled = True
                print(f"[eval] prompt_len={int(prompt_len)} compiled_dt={dt:.2f}s")
            else:
                token_ids = beam_search_jit(params, prompt)
                tok_np = np.asarray(token_ids)  # [B, K, 3]

            for row, sample_idx in enumerate(chunk[:real_chunk]):
                preds = [_decode_sid_triplet(tokenizer, tok_np[row, b].tolist()) for b in range(tok_np.shape[1])]
                predictions[sample_idx] = preds

    metrics = compute_hr_ndcg(predictions=predictions, targets=targets, topk=topk, valid_items=set(valid_sids))

    if output_predictions_json:
        payload = []
        for target, preds in zip(targets, predictions, strict=True):
            payload.append({"output": target, "predict": preds})
        path = Path(output_predictions_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        metrics_path = str(path.with_suffix(".metrics.json"))
        Path(metrics_path).write_text(json.dumps(asdict(metrics), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return predictions, metrics


__all__ = ["evaluate_sid_next_item_jax"]
