from __future__ import annotations

import json
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

    # Precompute prompt lengths to batch only equal-length prompts (KV-cache assumes shared prompt_len).
    prompt_lens = [len(eval_dataset[i]["input_ids"]) for i in range(n)]
    buckets: dict[int, list[int]] = {}
    for i, l in enumerate(prompt_lens):
        buckets.setdefault(int(l), []).append(i)

    predictions: list[list[str]] = [None] * n  # type: ignore[list-item]
    for prompt_len, idxs in sorted(buckets.items()):
        for start in range(0, len(idxs), int(batch_size)):
            chunk = idxs[start : start + int(batch_size)]
            prompt_ids = [eval_dataset[i]["input_ids"] for i in chunk]
            prompt = jnp.asarray(prompt_ids, dtype=jnp.int32)

            out = constrained_beam_search_sid3(
                model=model,
                params=params,
                prompt_input_ids=prompt,
                trie=trie,
                num_beams=int(num_beams),
                max_cache_length=int(max_cache_length),
                suffix_token_ids=newline_suffix,
            )
            tok_np = np.asarray(out.token_ids)  # [B, K, 3]

            for row, sample_idx in enumerate(chunk):
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
