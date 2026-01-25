from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from plugins.sample.decoding.sid3_constrained_beam_search import constrained_beam_search_sid3
from plugins.sample.constraints.sid_trie import build_sid_trie_from_index
from plugins.sft.metrics import RankingMetrics, compute_hr_ndcg
from plugins.sft.sid_utils import load_valid_sids_from_info


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

    # Precompute prompt lengths.
    #
    # NOTE: The underlying model cache assumes a shared `end_index` across the
    # batch, so we still bucket by true prompt length. However, to avoid JIT
    # recompilation per bucket, we right-pad prompts to a fixed `pad_len`
    # (same shape for all buckets) and pass the bucket's true prompt length to
    # the beam search as `prompt_true_len`.
    prompt_lens = [len(eval_dataset[i]["input_ids"]) for i in range(n)]
    if not prompt_lens:
        raise ValueError("Empty eval_dataset")
    pad_len = int(max(prompt_lens))
    buckets: dict[int, list[int]] = {}
    for i, l in enumerate(prompt_lens):
        buckets.setdefault(int(l), []).append(i)

    predictions: list[list[str]] = [None] * n  # type: ignore[list-item]
    print(
        f"[eval] samples={n} buckets={len(buckets)} batch_size={int(batch_size)} "
        f"num_beams={int(num_beams)} max_cache_length={int(max_cache_length)} pad_len={int(pad_len)}"
    )

    def _beam_search(params_in: Any, prompt_input_ids: jax.Array, prompt_true_len: jax.Array):
        out = constrained_beam_search_sid3(
            model=model,
            params=params_in,
            prompt_input_ids=prompt_input_ids,
            trie=trie,
            num_beams=int(num_beams),
            max_cache_length=int(max_cache_length),
            suffix_token_ids=newline_suffix,
            prompt_true_len=prompt_true_len,
        )
        return out.token_ids

    beam_search_jit = jax.jit(_beam_search)
    pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)

    compiled_once = False
    for prompt_len, idxs in sorted(buckets.items()):
        n_bucket = int(len(idxs))
        chunks = (n_bucket + int(batch_size) - 1) // int(batch_size)
        print(f"[eval] prompt_len={int(prompt_len)} bucket_samples={n_bucket} chunks={chunks}")

        starts = range(0, len(idxs), int(batch_size))
        if tqdm is not None:
            starts = tqdm(starts, total=chunks, desc=f"eval len={int(prompt_len)}", mininterval=1.0)

        for start in starts:
            chunk = idxs[start : start + int(batch_size)]
            real_chunk = int(len(chunk))
            if real_chunk < int(batch_size):
                chunk = chunk + [chunk[-1]] * (int(batch_size) - real_chunk)

            prompt_ids = [eval_dataset[i]["input_ids"] for i in chunk]
            # Right-pad prompts to `pad_len` so shapes stay constant across buckets.
            prompt_np = np.full((int(batch_size), int(pad_len)), pad_token_id, dtype=np.int32)
            for row, ids in enumerate(prompt_ids):
                ids_np = np.asarray(ids, dtype=np.int32)
                prompt_np[row, : ids_np.shape[0]] = ids_np
            prompt = jnp.asarray(prompt_np, dtype=jnp.int32)
            true_len = jnp.asarray(int(prompt_len), dtype=jnp.int32)

            if not compiled_once:
                t0 = time.perf_counter()
                token_ids = beam_search_jit(params, prompt, true_len)
                tok_np = np.asarray(token_ids)  # triggers compilation + blocks
                dt = time.perf_counter() - t0
                compiled_once = True
                print(f"[eval] compiled_dt={dt:.2f}s")
            else:
                token_ids = beam_search_jit(params, prompt, true_len)
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


class SidNextItemJaxEvaluator:
    """Reusable evaluator for repeated SID next-item HR/NDCG checks.

    This caches the SID trie, valid-items set, dataset bucketing, and the JIT
    compiled constrained beam-search so epoch-level eval does not recompile.
    """

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        eval_dataset: Any,
        sid_index_path: str,
        info_file: str,
        batch_size: int,
        num_beams: int,
        max_cache_length: int,
        topk: list[int],
        show_progress: bool = False,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._eval_dataset = eval_dataset

        self._batch_size = int(batch_size)
        if self._batch_size <= 0:
            raise ValueError("batch_size must be >= 1")

        self._num_beams = int(num_beams)
        self._max_cache_length = int(max_cache_length)
        self._topk = [int(k) for k in topk]
        self._show_progress = bool(show_progress)

        self._valid_items = set(load_valid_sids_from_info(info_file))
        self._trie = build_sid_trie_from_index(
            tokenizer=tokenizer,
            sid_index_path=sid_index_path,
            eos_token_id=int(getattr(tokenizer, "eos_token_id")),
        )
        self._newline_suffix = _newline_suffix_token_ids(tokenizer)

        n = int(len(eval_dataset))
        if n <= 0:
            raise ValueError("Empty eval_dataset")
        self._n = n

        targets = list(getattr(eval_dataset, "get_targets")())
        if len(targets) != n:
            raise ValueError("eval_dataset.get_targets() length mismatch")
        self._targets = targets

        self._pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
        prompt_lens = [len(eval_dataset[i]["input_ids"]) for i in range(n)]
        pad_len = int(max(prompt_lens))
        if pad_len <= 0:
            raise ValueError("Invalid prompt length (pad_len <= 0)")
        self._pad_len = pad_len

        buckets: dict[int, list[int]] = {}
        for i, l in enumerate(prompt_lens):
            buckets.setdefault(int(l), []).append(i)
        self._buckets = buckets

        if self._pad_len > self._max_cache_length:
            raise ValueError(
                f"max_cache_length too small for padded prompts: pad_len={self._pad_len} > max_cache_length={self._max_cache_length}"
            )

        def _beam_search(params_in: Any, prompt_input_ids: jax.Array, prompt_true_len: jax.Array):
            out = constrained_beam_search_sid3(
                model=self._model,
                params=params_in,
                prompt_input_ids=prompt_input_ids,
                trie=self._trie,
                num_beams=self._num_beams,
                max_cache_length=self._max_cache_length,
                suffix_token_ids=self._newline_suffix,
                prompt_true_len=prompt_true_len,
            )
            return out.token_ids

        self._beam_search_jit = jax.jit(_beam_search)
        self._compiled = False

    def _maybe_compile(self, params: Any) -> None:
        if self._compiled:
            return

        # Warm-up compile on the largest prompt shape; values do not matter.
        prompt = jnp.full((self._batch_size, self._pad_len), self._pad_token_id, dtype=jnp.int32)
        true_len = jnp.asarray(int(min(self._buckets.keys())), dtype=jnp.int32)
        t0 = time.perf_counter()
        token_ids = self._beam_search_jit(params, prompt, true_len)
        _ = np.asarray(token_ids)  # block on compilation
        dt = time.perf_counter() - t0
        print(f"[eval] compiled_dt={dt:.2f}s")
        self._compiled = True

    def evaluate(
        self,
        *,
        params: Any,
        output_predictions_json: str | None = None,
    ) -> tuple[list[list[str]], RankingMetrics]:
        self._maybe_compile(params)

        try:
            from tqdm import tqdm  # type: ignore
        except Exception:  # pragma: no cover - optional dep
            tqdm = None  # type: ignore[assignment]

        predictions: list[list[str]] = [None] * self._n  # type: ignore[list-item]
        print(
            f"[eval] samples={self._n} buckets={len(self._buckets)} batch_size={self._batch_size} "
            f"num_beams={self._num_beams} max_cache_length={self._max_cache_length} pad_len={self._pad_len}"
        )

        for prompt_len, idxs in sorted(self._buckets.items()):
            n_bucket = int(len(idxs))
            chunks = (n_bucket + int(self._batch_size) - 1) // int(self._batch_size)
            print(f"[eval] prompt_len={int(prompt_len)} bucket_samples={n_bucket} chunks={chunks}")

            starts = range(0, len(idxs), int(self._batch_size))
            if self._show_progress and tqdm is not None:
                starts = tqdm(starts, total=chunks, desc=f"eval len={int(prompt_len)}", mininterval=1.0)

            for start in starts:
                chunk = idxs[start : start + int(self._batch_size)]
                real_chunk = int(len(chunk))
                if real_chunk < int(self._batch_size):
                    chunk = chunk + [chunk[-1]] * (int(self._batch_size) - real_chunk)

                prompt_ids = [self._eval_dataset[i]["input_ids"] for i in chunk]
                prompt_np = np.full((int(self._batch_size), int(self._pad_len)), self._pad_token_id, dtype=np.int32)
                for row, ids in enumerate(prompt_ids):
                    ids_np = np.asarray(ids, dtype=np.int32)
                    prompt_np[row, : ids_np.shape[0]] = ids_np

                prompt = jnp.asarray(prompt_np, dtype=jnp.int32)
                true_len = jnp.asarray(int(prompt_len), dtype=jnp.int32)

                token_ids = self._beam_search_jit(params, prompt, true_len)
                tok_np = np.asarray(token_ids)  # [B, K, 3]

                for row, sample_idx in enumerate(chunk[:real_chunk]):
                    preds = [_decode_sid_triplet(self._tokenizer, tok_np[row, b].tolist()) for b in range(tok_np.shape[1])]
                    predictions[sample_idx] = preds

        metrics = compute_hr_ndcg(
            predictions=predictions,
            targets=self._targets,
            topk=self._topk,
            valid_items=self._valid_items,
        )

        if output_predictions_json:
            payload = []
            for target, preds in zip(self._targets, predictions, strict=True):
                payload.append({"output": target, "predict": preds})
            path = Path(output_predictions_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

            metrics_path = str(path.with_suffix(".metrics.json"))
            Path(metrics_path).write_text(json.dumps(asdict(metrics), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        return predictions, metrics


__all__ = ["SidNextItemJaxEvaluator", "evaluate_sid_next_item_jax"]
