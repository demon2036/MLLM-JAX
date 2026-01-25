from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.multihost_utils import process_allgather

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


def _broadcast_vec_from_process0(vec: np.ndarray) -> np.ndarray:
    """Broadcast a small 1D vector from process 0 to all JAX processes."""
    if int(jax.process_count()) <= 1:
        return np.asarray(vec)
    gathered = np.asarray(process_allgather(np.asarray(vec)))
    # Do not assume process_allgather returns rows in process_index order. Find
    # the row that corresponds to process 0 explicitly.
    proc_ids = np.asarray(process_allgather(np.asarray([int(jax.process_index())], dtype=np.int32))).reshape((-1,))
    idx = np.where(proc_ids == 0)[0]
    if idx.size != 1:
        raise RuntimeError(f"process_allgather returned invalid process indices: {proc_ids.tolist()}")
    return np.asarray(gathered[int(idx[0])])


def _pack_metrics_vec(*, metrics: RankingMetrics, topk_list: list[int]) -> np.ndarray:
    hr_vals = [float(metrics.hr[int(k)]) for k in topk_list]
    ndcg_vals = [float(metrics.ndcg[int(k)]) for k in topk_list]
    invalid = float(int(metrics.invalid_prediction_count))
    return np.asarray([*hr_vals, *ndcg_vals, invalid], dtype=np.float32)


def _unpack_metrics_vec(*, vec: np.ndarray, topk_list: list[int], n_samples: int, n_beams: int) -> RankingMetrics:
    k = int(len(topk_list))
    if vec.shape != (2 * k + 1,):
        raise ValueError(f"Invalid metrics vec shape {vec.shape}, expected {(2 * k + 1,)}")
    hr = {int(topk_list[i]): float(vec[i]) for i in range(k)}
    ndcg = {int(topk_list[i]): float(vec[i + k]) for i in range(k)}
    invalid = int(round(float(vec[2 * k])))
    return RankingMetrics(
        hr=hr,
        ndcg=ndcg,
        topk=list(topk_list),
        n_samples=int(n_samples),
        n_beams=int(n_beams),
        invalid_prediction_count=invalid,
    )


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
    evaluator = SidNextItemJaxEvaluator(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        sid_index_path=sid_index_path,
        info_file=info_file,
        batch_size=int(batch_size),
        num_beams=int(num_beams),
        max_cache_length=int(max_cache_length),
        topk=list(topk),
        show_progress=False,
    )
    return evaluator.evaluate(params=params, output_predictions_json=output_predictions_json)


class SidNextItemJaxEvaluator:
    """Reusable evaluator for repeated SID next-item HR/NDCG checks.

    This caches the SID trie, valid-items set, dataset padding metadata, and the
    JIT compiled constrained beam-search so epoch-level eval does not recompile.
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
        self._is_coordinator = int(jax.process_index()) == 0

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

        self._process_count = int(jax.process_count())
        self._process_index = int(jax.process_index())
        self._gather_slot_to_process_index = list(range(int(self._process_count)))
        if int(self._process_count) > 1:
            proc_ids = np.asarray(
                process_allgather(np.asarray([int(self._process_index)], dtype=np.int32)),
            ).reshape((-1,))
            if proc_ids.shape != (int(self._process_count),):
                raise RuntimeError(f"Unexpected gathered process indices shape: {proc_ids.shape}")
            slot_to_proc = [int(x) for x in proc_ids.tolist()]
            if sorted(slot_to_proc) != list(range(int(self._process_count))):
                raise RuntimeError(f"process_allgather returned unexpected process indices: {slot_to_proc}")
            self._gather_slot_to_process_index = slot_to_proc
            if self._is_coordinator:
                print(f"[eval] process_allgather_order={slot_to_proc}")

        self._pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
        prompt_lens = [len(eval_dataset[i]["input_ids"]) for i in range(n)]
        if any(int(x) <= 0 for x in prompt_lens):
            raise ValueError("Invalid prompt length (found <= 0)")
        pad_len = int(max(prompt_lens))
        if pad_len <= 0:
            raise ValueError("Invalid prompt length (pad_len <= 0)")
        self._prompt_lens = [int(x) for x in prompt_lens]
        self._pad_len = pad_len

        global_batch = int(self._batch_size) * int(self._process_count)
        n_padded = ((int(self._n) + int(global_batch) - 1) // int(global_batch)) * int(global_batch)
        self._local_n = int(n_padded) // int(self._process_count)
        self._local_start = int(self._process_index) * int(self._local_n)

        if int(self._process_count) > 1:
            mid = int(self._n) // 2
            sig = np.asarray(
                [int(self._n), int(self._pad_len), int(self._prompt_lens[0]), int(self._prompt_lens[mid]), int(self._prompt_lens[-1])],
                dtype=np.int32,
            )
            sig_all = np.asarray(process_allgather(np.asarray(sig)))
            if sig_all.ndim != 2 or sig_all.shape[1] != sig.shape[0]:
                raise RuntimeError(f"Unexpected process_allgather signature shape: {sig_all.shape}")
            if not np.all(sig_all == sig_all[0]):
                raise RuntimeError(
                    "Eval dataset appears to differ across processes (lengths/padding mismatch). "
                    "Sharded multi-host eval requires all processes to construct the same eval_dataset order."
                )

        # Constrained decode inserts token_1..3 after the padded prompt length, so
        # we need a few extra cache slots beyond `pad_len`.
        suffix_len = int(len(self._newline_suffix)) + 1  # newline suffix + EOS
        max_pos = int(self._pad_len) + 2 + suffix_len
        if int(self._max_cache_length) <= int(max_pos):
            raise ValueError(
                "max_cache_length too small for padded prompts + constrained decode: "
                f"need > {max_pos} (pad_len={self._pad_len}, suffix_len={suffix_len}), got {self._max_cache_length}. "
                "Increase cfg.jax.max_cache_length or reduce cfg.data.max_len."
            )

        def _beam_search(params_in: Any, prompt_input_ids: jax.Array, prompt_true_lens: jax.Array):
            out = constrained_beam_search_sid3(
                model=self._model,
                params=params_in,
                prompt_input_ids=prompt_input_ids,
                trie=self._trie,
                num_beams=self._num_beams,
                max_cache_length=self._max_cache_length,
                suffix_token_ids=self._newline_suffix,
                prompt_true_len=prompt_true_lens,
            )
            return out.token_ids

        self._beam_search_jit = jax.jit(_beam_search)
        self._compiled = False

    def _maybe_compile(self, params: Any) -> None:
        if self._compiled:
            return

        # Warm-up compile on the padded prompt shape; values do not matter.
        prompt = jnp.full((self._batch_size, self._pad_len), self._pad_token_id, dtype=jnp.int32)
        min_len = int(min(self._prompt_lens))
        true_lens = jnp.full((self._batch_size,), min_len, dtype=jnp.int32)
        t0 = time.perf_counter()
        token_ids = self._beam_search_jit(params, prompt, true_lens)
        if self._is_coordinator:
            _ = np.asarray(token_ids)  # block on compilation
        else:
            token_ids.block_until_ready()
        dt = time.perf_counter() - t0
        if self._is_coordinator:
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

        predictions: list[list[str]] | None = [None] * self._n if self._is_coordinator else None  # type: ignore[list-item]
        if self._is_coordinator:
            global_batch = int(self._batch_size) * int(self._process_count)
            n_padded = int(self._local_n) * int(self._process_count)
            print(
                f"[eval] samples={self._n} process_count={self._process_count} batch_size={self._batch_size} "
                f"global_batch={global_batch} padded_samples={n_padded} num_beams={self._num_beams} "
                f"max_cache_length={self._max_cache_length} pad_len={self._pad_len}"
            )

        starts = range(0, int(self._local_n), int(self._batch_size))
        if self._show_progress and tqdm is not None and self._is_coordinator:
            starts = tqdm(starts, total=int(self._local_n) // int(self._batch_size), desc="eval", mininterval=1.0)

        for start in starts:
            base = int(self._local_start) + int(start)
            batch_indices = []
            true_lens = np.zeros((int(self._batch_size),), dtype=np.int32)
            for row in range(int(self._batch_size)):
                global_i = int(base) + int(row)
                dataset_i = int(global_i) if int(global_i) < int(self._n) else int(self._n) - 1
                batch_indices.append(int(dataset_i))
                true_lens[row] = int(self._prompt_lens[dataset_i])

            prompt_np = np.full((int(self._batch_size), int(self._pad_len)), self._pad_token_id, dtype=np.int32)
            for row, dataset_i in enumerate(batch_indices):
                ids_np = np.asarray(self._eval_dataset[int(dataset_i)]["input_ids"], dtype=np.int32)
                prompt_np[row, : ids_np.shape[0]] = ids_np

            prompt = jnp.asarray(prompt_np, dtype=jnp.int32)
            true_lens_arr = jnp.asarray(true_lens, dtype=jnp.int32)

            token_ids = self._beam_search_jit(params, prompt, true_lens_arr)
            tok_np = np.asarray(process_allgather(np.asarray(token_ids)))  # [P, B, K, 3]

            if self._is_coordinator:
                assert predictions is not None
                for slot in range(int(self._process_count)):
                    proc_index = int(self._gather_slot_to_process_index[int(slot)])
                    proc_base = int(proc_index) * int(self._local_n) + int(start)
                    for row in range(int(self._batch_size)):
                        sample_idx = int(proc_base) + int(row)
                        if sample_idx >= int(self._n):
                            continue
                        preds = [_decode_sid_triplet(self._tokenizer, tok_np[int(slot), row, b].tolist()) for b in range(tok_np.shape[2])]
                        predictions[sample_idx] = preds

        topk_list = sorted({int(k) for k in self._topk if int(k) > 0 and int(k) <= int(self._num_beams)})
        if not topk_list:
            raise ValueError(f"No valid topk <= num_beams={int(self._num_beams)}: {list(self._topk)}")

        if self._is_coordinator:
            assert predictions is not None
            if any(p is None for p in predictions):
                raise RuntimeError("Eval produced incomplete predictions (some samples were not filled).")
            metrics = compute_hr_ndcg(
                predictions=predictions,
                targets=self._targets,
                topk=topk_list,
                valid_items=self._valid_items,
            )
            metrics_vec = _pack_metrics_vec(metrics=metrics, topk_list=topk_list)
        else:
            metrics_vec = np.zeros((2 * len(topk_list) + 1,), dtype=np.float32)

        metrics_vec = _broadcast_vec_from_process0(metrics_vec)
        metrics = _unpack_metrics_vec(vec=metrics_vec, topk_list=topk_list, n_samples=int(self._n), n_beams=int(self._num_beams))

        if output_predictions_json and self._is_coordinator:
            assert predictions is not None
            payload = []
            for target, preds in zip(self._targets, predictions, strict=True):
                payload.append({"output": target, "predict": preds})
            path = Path(output_predictions_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

            metrics_path = str(path.with_suffix(".metrics.json"))
            Path(metrics_path).write_text(json.dumps(asdict(metrics), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        return (predictions or []), metrics


__all__ = ["SidNextItemJaxEvaluator", "evaluate_sid_next_item_jax"]
