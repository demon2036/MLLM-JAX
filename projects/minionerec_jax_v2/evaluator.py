from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from plugins.minionerec_v2.beam_decode import constrained_beam_search_sid3, constrained_beam_search_sid3_prefill
from plugins.minionerec_v2.constraints import SidTrie
from projects.minionerec_jax_v2.metrics import RankingMetrics, compute_hr_ndcg

_PROGRESS_FILE_VERSION = 1
_PROGRESS_SAVE_EVERY_CHUNKS = 16


def _split_eval_rng(base_key: jax.Array, chunk_index: int) -> tuple[jax.Array, jax.Array]:
    key = jax.random.fold_in(base_key, jnp.asarray(int(chunk_index), dtype=jnp.uint32))
    return jax.random.split(key)


def _decode_sid_triplet(tokenizer: Any, triplet: list[int]) -> str:
    toks = tokenizer.convert_ids_to_tokens([int(x) for x in triplet])
    return "".join(str(t) for t in toks)


def _newline_suffix_token_ids(tokenizer: Any) -> list[int]:
    base = list(tokenizer.encode("a", add_special_tokens=False))
    with_nl = list(tokenizer.encode("a\n", add_special_tokens=False))
    lcp = 0
    for x, y in zip(base, with_nl, strict=False):
        if int(x) != int(y):
            break
        lcp += 1
    return [int(x) for x in with_nl[lcp:]]


_DEFAULT_PREFILL_BUCKETS = (128, 256, 512, 1024, 2048, 4096, 8192)


def _find_prefill_length(desired_length: int) -> int:
    desired = int(desired_length)
    for value in _DEFAULT_PREFILL_BUCKETS:
        if int(value) >= desired:
            return int(value)
    raise ValueError(f"No prefill bucket found for desired_length={desired}")


def _normalize_prefill_mode(mode: str | None) -> str:
    m = str(mode or "bucket").strip().lower()
    if m in {"bucket", "buckets"}:
        return "bucket"
    if m in {"fixed", "single", "one"}:
        return "fixed"
    if m in {"exact", "length", "per-length", "per_length"}:
        return "exact"
    raise ValueError(f"Unknown prefill_mode={mode!r} (expected: bucket|fixed|exact)")


def _build_prefill_buckets(
    prompt_lens: list[int],
    *,
    max_cache_length: int,
    suffix_len: int,
    prefill_mode: str | None,
    fixed_prefill_len: int | None,
) -> dict[int, list[int]]:
    n = int(len(prompt_lens))
    if n <= 0:
        raise ValueError("Empty prompt_lens")

    mode = _normalize_prefill_mode(prefill_mode)
    max_cache_length_i = int(max_cache_length)
    suffix_len_i = int(suffix_len)

    def _check_cache_len(prefill_len: int) -> None:
        max_pos = int(prefill_len) + 2 + suffix_len_i
        if max_cache_length_i <= max_pos:
            raise ValueError(
                f"max_cache_length too small: need > {max_pos} (prefill_len={int(prefill_len)}, suffix_len={suffix_len_i})"
            )

    if mode == "fixed":
        max_prompt_len = max(int(x) for x in prompt_lens)
        if int(max_prompt_len) <= 0:
            raise ValueError(f"Invalid prompt length: max_prompt_len={int(max_prompt_len)}")

        if fixed_prefill_len is not None and int(fixed_prefill_len) > 0:
            prefill_len = int(fixed_prefill_len)
        else:
            prefill_len = int(max_prompt_len)

        if int(prefill_len) < int(max_prompt_len):
            raise ValueError(
                f"fixed prefill_len too small: prefill_len={int(prefill_len)} < max_prompt_len={int(max_prompt_len)}"
            )

        _check_cache_len(int(prefill_len))
        return {int(prefill_len): list(range(n))}

    if mode == "exact":
        buckets_exact: dict[int, list[int]] = {}
        for idx, prompt_len in enumerate(prompt_lens):
            prefill_len = int(prompt_len)
            if prefill_len <= 0:
                raise ValueError(f"Invalid prompt length at idx={idx}: {prefill_len}")
            _check_cache_len(prefill_len)
            buckets_exact.setdefault(prefill_len, []).append(idx)
        return buckets_exact

    buckets: dict[int, list[int]] = {}
    for idx, prompt_len in enumerate(prompt_lens):
        prefill_len = _find_prefill_length(int(prompt_len))
        _check_cache_len(int(prefill_len))
        buckets.setdefault(int(prefill_len), []).append(idx)
    return buckets


def _progress_state_path(output_predictions_json: str) -> Path:
    return Path(output_predictions_json).with_suffix(".progress.json")


def _progress_payload(
    *,
    predictions: list[list[str] | None],
    num_beams: int,
    topk: list[int],
    do_sample: bool = False,
    temperature: float = 1.0,
) -> dict[str, Any]:
    serialized_predictions = {
        str(idx): preds
        for idx, preds in enumerate(predictions)
        if preds is not None
    }
    return {
        "version": int(_PROGRESS_FILE_VERSION),
        "n_samples": int(len(predictions)),
        "num_beams": int(num_beams),
        "topk": [int(k) for k in topk],
        "do_sample": bool(do_sample),
        "temperature": float(temperature),
        "predictions": serialized_predictions,
    }


def _save_progress_state(
    *,
    path: Path,
    predictions: list[list[str] | None],
    num_beams: int,
    topk: list[int],
    do_sample: bool = False,
    temperature: float = 1.0,
) -> None:
    payload = _progress_payload(
        predictions=predictions,
        num_beams=num_beams,
        topk=topk,
        do_sample=do_sample,
        temperature=temperature,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _load_progress_state(
    *,
    path: Path,
    n_samples: int,
    num_beams: int | None = None,
    topk: list[int] | None = None,
    do_sample: bool | None = None,
    temperature: float | None = None,
) -> list[list[str] | None]:
    predictions: list[list[str] | None] = [None] * int(n_samples)
    if not path.exists():
        return predictions

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[eval] warning: failed to parse progress file={path}: {exc!r}; starting from scratch")
        return predictions

    if not isinstance(payload, dict):
        print(f"[eval] warning: invalid progress payload type={type(payload).__name__}; starting from scratch")
        return predictions

    saved_n = payload.get("n_samples")
    if isinstance(saved_n, int) and int(saved_n) != int(n_samples):
        print(
            f"[eval] warning: progress n_samples mismatch saved={int(saved_n)} current={int(n_samples)}; "
            "starting from scratch"
        )
        return predictions

    if num_beams is not None:
        saved_num_beams = payload.get("num_beams")
        if not isinstance(saved_num_beams, int) or int(saved_num_beams) != int(num_beams):
            print(
                f"[eval] warning: progress num_beams mismatch saved={saved_num_beams!r} current={int(num_beams)}; "
                "starting from scratch"
            )
            return predictions

    if topk is not None:
        saved_topk = payload.get("topk")
        if not isinstance(saved_topk, list):
            print("[eval] warning: progress topk missing/invalid; starting from scratch")
            return predictions
        saved_topk_norm = [int(k) for k in saved_topk]
        current_topk_norm = [int(k) for k in topk]
        if saved_topk_norm != current_topk_norm:
            print(
                f"[eval] warning: progress topk mismatch saved={saved_topk_norm} current={current_topk_norm}; "
                "starting from scratch"
            )
            return predictions

    if do_sample is not None:
        saved_do_sample = payload.get("do_sample")
        if not isinstance(saved_do_sample, bool) or bool(saved_do_sample) != bool(do_sample):
            print(
                f"[eval] warning: progress do_sample mismatch saved={saved_do_sample!r} current={bool(do_sample)}; "
                "starting from scratch"
            )
            return predictions

    if temperature is not None:
        saved_temperature = payload.get("temperature")
        if saved_temperature is None:
            print("[eval] warning: progress temperature missing; starting from scratch")
            return predictions
        try:
            saved_temperature_f = float(saved_temperature)
        except Exception:
            print(f"[eval] warning: progress temperature invalid={saved_temperature!r}; starting from scratch")
            return predictions
        if abs(saved_temperature_f - float(temperature)) > 1e-9:
            print(
                f"[eval] warning: progress temperature mismatch saved={saved_temperature_f} current={float(temperature)}; "
                "starting from scratch"
            )
            return predictions

    saved_predictions = payload.get("predictions")
    if not isinstance(saved_predictions, dict):
        print("[eval] warning: progress file missing 'predictions' dict; starting from scratch")
        return predictions

    restored = 0
    for key, value in saved_predictions.items():
        try:
            idx = int(key)
        except Exception:
            continue
        if idx < 0 or idx >= int(n_samples):
            continue
        if isinstance(value, list):
            predictions[idx] = [str(v) for v in value]
            restored += 1

    print(f"[eval] resumed_from_progress file={path} restored_samples={restored}/{int(n_samples)}")
    return predictions


def evaluate_sid_next_item_jax(
    *,
    model: Any,
    params: Any,
    tokenizer: Any,
    eval_dataset: Any,
    trie: SidTrie,
    valid_sids: set[str],
    batch_size: int,
    num_beams: int,
    max_cache_length: int,
    topk: list[int],
    output_predictions_json: str | None,
    prefill_mode: str | None = "bucket",
    fixed_prefill_len: int | None = None,
    do_sample: bool = False,
    temperature: float = 1.0,
    seed: int = 42,
    show_progress: bool = False,
) -> tuple[list[list[str]], RankingMetrics]:
    evaluator = SidNextItemJaxEvaluator(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        trie=trie,
        valid_sids=valid_sids,
        batch_size=int(batch_size),
        num_beams=int(num_beams),
        max_cache_length=int(max_cache_length),
        topk=[int(k) for k in topk],
        show_progress=bool(show_progress),
        prefill_mode=prefill_mode,
        fixed_prefill_len=fixed_prefill_len,
        do_sample=bool(do_sample),
        temperature=float(temperature),
        seed=int(seed),
    )
    return evaluator.evaluate(params=params, output_predictions_json=output_predictions_json)


def _run_constrained_decode(
    *,
    model: Any,
    params: Any,
    trie: SidTrie,
    prompt_input_ids: jax.Array,
    prompt_true_len: jax.Array,
    num_beams: int,
    max_cache_length: int,
    eos_token_id: int,
    suffix_token_ids: list[int],
    prefill_mode: str,
    do_sample: bool,
    temperature: float,
    rng_key: jax.Array,
) -> jax.Array:
    if str(prefill_mode) in {"fixed", "exact"}:
        out = constrained_beam_search_sid3(
            model=model,
            params=params,
            prompt_input_ids=prompt_input_ids,
            trie=trie,
            num_beams=int(num_beams),
            max_cache_length=int(max_cache_length),
            eos_token_id=int(eos_token_id),
            suffix_token_ids=suffix_token_ids,
            prompt_true_len=prompt_true_len,
            do_sample=bool(do_sample),
            temperature=float(temperature),
            rng_key=rng_key,
        )
        return out.token_ids

    out = constrained_beam_search_sid3_prefill(
        model=model,
        params=params,
        prompt_input_ids=prompt_input_ids,
        trie=trie,
        num_beams=int(num_beams),
        max_cache_length=int(max_cache_length),
        eos_token_id=int(eos_token_id),
        suffix_token_ids=suffix_token_ids,
        prompt_true_len=prompt_true_len,
        do_sample=bool(do_sample),
        temperature=float(temperature),
        rng_key=rng_key,
    )
    return out.token_ids


class SidNextItemJaxEvaluator:
    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        eval_dataset: Any,
        trie: SidTrie,
        valid_sids: set[str],
        batch_size: int,
        num_beams: int,
        max_cache_length: int,
        topk: list[int],
        show_progress: bool = False,
        prefill_mode: str | None = "bucket",
        fixed_prefill_len: int | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        seed: int = 42,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._eval_dataset = eval_dataset
        self._trie = trie
        self._valid_items = set(valid_sids)

        self._batch_size = int(batch_size)
        if self._batch_size <= 0:
            raise ValueError("batch_size must be >= 1")

        self._num_beams = int(num_beams)
        self._max_cache_length = int(max_cache_length)
        self._topk = [int(k) for k in topk]
        self._show_progress = bool(show_progress)
        self._do_sample = bool(do_sample)
        self._temperature = float(temperature)
        if self._temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got: {self._temperature}")
        self._seed = int(seed)
        self._rng_key = jax.random.PRNGKey(self._seed)
        self._compile_rng_base = 2_147_000_000

        self._newline_suffix = _newline_suffix_token_ids(tokenizer)
        self._eos_token_id = int(getattr(tokenizer, "eos_token_id", -1))
        if self._eos_token_id < 0:
            raise ValueError("tokenizer.eos_token_id is required for constrained beam decode")

        self._prefill_mode = _normalize_prefill_mode(prefill_mode)
        self._fixed_prefill_len = None if fixed_prefill_len is None else int(fixed_prefill_len)

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
        suffix_len = int(len(self._newline_suffix) + 1)

        prefill_buckets = _build_prefill_buckets(
            prompt_lens,
            max_cache_length=int(self._max_cache_length),
            suffix_len=int(suffix_len),
            prefill_mode=self._prefill_mode,
            fixed_prefill_len=self._fixed_prefill_len,
        )

        self._prefill_len: int | None = None
        self._true_len_buckets: dict[int, list[int]] = {}
        self._buckets: dict[int, list[int]] = prefill_buckets

        if self._prefill_mode in {"fixed", "exact"}:
            for i, length in enumerate(prompt_lens):
                self._true_len_buckets.setdefault(int(length), []).append(i)
            if self._prefill_mode == "fixed":
                self._prefill_len = int(next(iter(prefill_buckets.keys())))

        def _decode(params_in: Any, prompt_input_ids: jax.Array, prompt_true_len: jax.Array, rng_key: jax.Array):
            return _run_constrained_decode(
                model=self._model,
                params=params_in,
                trie=self._trie,
                prompt_input_ids=prompt_input_ids,
                prompt_true_len=prompt_true_len,
                num_beams=self._num_beams,
                max_cache_length=self._max_cache_length,
                eos_token_id=int(self._eos_token_id),
                suffix_token_ids=self._newline_suffix,
                prefill_mode=self._prefill_mode,
                do_sample=self._do_sample,
                temperature=self._temperature,
                rng_key=rng_key,
            )

        self._decode_jit = jax.jit(_decode)
        self._compiled = False

    def _maybe_compile(self, params: Any) -> None:
        if self._compiled:
            return

        if self._prefill_mode == "fixed":
            prefill_len = int(self._prefill_len)
            prompt = jnp.full((self._batch_size, prefill_len), self._pad_token_id, dtype=jnp.int32)
            true_len = jnp.asarray(int(min(self._true_len_buckets.keys())), dtype=jnp.int32)
            decode_key, _ = _split_eval_rng(self._rng_key, self._compile_rng_base)
            t0 = time.perf_counter()
            token_ids = self._decode_jit(params, prompt, true_len, decode_key)
            _ = np.asarray(token_ids)
            dt = time.perf_counter() - t0
            print(f"[eval] compiled_dt={dt:.2f}s")
        elif self._prefill_mode == "exact":
            for prompt_len in sorted(self._true_len_buckets.keys()):
                prompt = jnp.full((self._batch_size, int(prompt_len)), self._pad_token_id, dtype=jnp.int32)
                true_len = jnp.asarray(int(prompt_len), dtype=jnp.int32)
                decode_key, _ = _split_eval_rng(self._rng_key, self._compile_rng_base + int(prompt_len))
                t0 = time.perf_counter()
                token_ids = self._decode_jit(params, prompt, true_len, decode_key)
                _ = np.asarray(token_ids)
                dt = time.perf_counter() - t0
                print(f"[eval] exact_len={int(prompt_len)} compiled_dt={dt:.2f}s")
        else:
            for prefill_len in sorted(self._buckets.keys()):
                prompt = jnp.full((self._batch_size, int(prefill_len)), self._pad_token_id, dtype=jnp.int32)
                true_len = jnp.full((self._batch_size,), int(prefill_len), dtype=jnp.int32)
                decode_key, _ = _split_eval_rng(self._rng_key, self._compile_rng_base + int(prefill_len))
                t0 = time.perf_counter()
                token_ids = self._decode_jit(params, prompt, true_len, decode_key)
                _ = np.asarray(token_ids)
                dt = time.perf_counter() - t0
                print(f"[eval] prefill_len={int(prefill_len)} compiled_dt={dt:.2f}s")
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
        except Exception:  # pragma: no cover
            tqdm = None  # type: ignore[assignment]

        progress_path: Path | None = None
        predictions_maybe: list[list[str] | None]
        if output_predictions_json:
            progress_path = _progress_state_path(output_predictions_json)
            predictions_maybe = _load_progress_state(
                path=progress_path,
                n_samples=self._n,
                num_beams=int(self._num_beams),
                topk=self._topk,
                do_sample=bool(self._do_sample),
                temperature=float(self._temperature),
            )
        else:
            predictions_maybe = [None] * self._n

        chunks_since_save = 0
        chunk_counter = 0

        def _maybe_save_progress() -> None:
            nonlocal chunks_since_save
            if progress_path is None:
                return
            if chunks_since_save < int(_PROGRESS_SAVE_EVERY_CHUNKS):
                return
            _save_progress_state(
                path=progress_path,
                predictions=predictions_maybe,
                num_beams=int(self._num_beams),
                topk=self._topk,
                do_sample=bool(self._do_sample),
                temperature=float(self._temperature),
            )
            chunks_since_save = 0

        if self._prefill_mode in {"fixed", "exact"}:
            if self._prefill_mode == "fixed":
                prefill_len = int(self._prefill_len)
                print(
                    f"[eval] samples={self._n} prefill_len={prefill_len} true_len_buckets={len(self._true_len_buckets)} "
                    f"batch_size={self._batch_size} num_beams={self._num_beams} max_cache_length={self._max_cache_length} "
                    f"prefill_mode={self._prefill_mode} do_sample={self._do_sample} temperature={self._temperature}"
                )
            else:
                print(
                    f"[eval] samples={self._n} exact_len_buckets={len(self._true_len_buckets)} "
                    f"batch_size={self._batch_size} num_beams={self._num_beams} max_cache_length={self._max_cache_length} "
                    f"prefill_mode={self._prefill_mode} do_sample={self._do_sample} temperature={self._temperature}"
                )

            for prompt_len, idxs in sorted(self._true_len_buckets.items()):
                bucket_prefill_len = int(self._prefill_len) if self._prefill_mode == "fixed" else int(prompt_len)
                n_bucket = int(len(idxs))
                chunks = (n_bucket + int(self._batch_size) - 1) // int(self._batch_size)
                print(
                    f"[eval] prompt_len={int(prompt_len)} prefill_len={int(bucket_prefill_len)} "
                    f"bucket_samples={n_bucket} chunks={chunks}"
                )

                starts = range(0, len(idxs), int(self._batch_size))
                if self._show_progress and tqdm is not None:
                    starts = tqdm(starts, total=chunks, desc=f"eval len={int(prompt_len)}", mininterval=1.0)

                for start in starts:
                    chunk_id = int(chunk_counter)
                    chunk_counter += 1
                    chunk = idxs[start : start + int(self._batch_size)]
                    real_chunk = int(len(chunk))
                    real_indices = chunk[:real_chunk]
                    if all(predictions_maybe[sample_idx] is not None for sample_idx in real_indices):
                        continue

                    if real_chunk < int(self._batch_size):
                        chunk = chunk + [chunk[-1]] * (int(self._batch_size) - real_chunk)

                    prompt_ids = [self._eval_dataset[i]["input_ids"] for i in chunk]
                    prompt_np = np.full((int(self._batch_size), int(bucket_prefill_len)), self._pad_token_id, dtype=np.int32)
                    for row, ids in enumerate(prompt_ids):
                        ids_np = np.asarray(ids, dtype=np.int32)
                        prompt_np[row, : ids_np.shape[0]] = ids_np

                    prompt = jnp.asarray(prompt_np, dtype=jnp.int32)
                    true_len = jnp.asarray(int(prompt_len), dtype=jnp.int32)

                    decode_key, _ = _split_eval_rng(self._rng_key, chunk_id)
                    token_ids = self._decode_jit(params, prompt, true_len, decode_key)
                    tok_np = np.asarray(token_ids)

                    for row, sample_idx in enumerate(real_indices):
                        preds = [_decode_sid_triplet(self._tokenizer, tok_np[row, beam].tolist()) for beam in range(tok_np.shape[1])]
                        predictions_maybe[sample_idx] = preds

                    if progress_path is not None:
                        chunks_since_save += 1
                        _maybe_save_progress()

        else:
            print(
                f"[eval] samples={self._n} prefill_buckets={len(self._buckets)} batch_size={self._batch_size} "
                f"num_beams={self._num_beams} max_cache_length={self._max_cache_length} prefill_mode={self._prefill_mode} "
                f"do_sample={self._do_sample} temperature={self._temperature}"
            )

            for prefill_len, idxs in sorted(self._buckets.items()):
                n_bucket = int(len(idxs))
                chunks = (n_bucket + int(self._batch_size) - 1) // int(self._batch_size)
                print(f"[eval] prefill_len={int(prefill_len)} bucket_samples={n_bucket} chunks={chunks}")

                starts = range(0, len(idxs), int(self._batch_size))
                if self._show_progress and tqdm is not None:
                    starts = tqdm(starts, total=chunks, desc=f"eval prefill={int(prefill_len)}", mininterval=1.0)

                for start in starts:
                    chunk_id = int(chunk_counter)
                    chunk_counter += 1
                    chunk = idxs[start : start + int(self._batch_size)]
                    real_chunk = int(len(chunk))
                    real_indices = chunk[:real_chunk]
                    if all(predictions_maybe[sample_idx] is not None for sample_idx in real_indices):
                        continue

                    if real_chunk < int(self._batch_size):
                        chunk = chunk + [chunk[-1]] * (int(self._batch_size) - real_chunk)

                    prompt_ids = [self._eval_dataset[i]["input_ids"] for i in chunk]
                    prompt_np = np.full((int(self._batch_size), int(prefill_len)), self._pad_token_id, dtype=np.int32)
                    for row, ids in enumerate(prompt_ids):
                        ids_np = np.asarray(ids, dtype=np.int32)
                        prompt_np[row, : ids_np.shape[0]] = ids_np

                    prompt = jnp.asarray(prompt_np, dtype=jnp.int32)
                    true_len = jnp.asarray(np.asarray([len(x) for x in prompt_ids], dtype=np.int32), dtype=jnp.int32)

                    decode_key, _ = _split_eval_rng(self._rng_key, chunk_id)
                    token_ids = self._decode_jit(params, prompt, true_len, decode_key)
                    tok_np = np.asarray(token_ids)

                    for row, sample_idx in enumerate(real_indices):
                        preds = [_decode_sid_triplet(self._tokenizer, tok_np[row, beam].tolist()) for beam in range(tok_np.shape[1])]
                        predictions_maybe[sample_idx] = preds

                    if progress_path is not None:
                        chunks_since_save += 1
                        _maybe_save_progress()

        if progress_path is not None:
            _save_progress_state(
                path=progress_path,
                predictions=predictions_maybe,
                num_beams=int(self._num_beams),
                topk=self._topk,
                do_sample=bool(self._do_sample),
                temperature=float(self._temperature),
            )

        missing = [idx for idx, preds in enumerate(predictions_maybe) if preds is None]
        if missing:
            head = ",".join(str(idx) for idx in missing[:10])
            raise RuntimeError(
                f"Evaluation finished with missing predictions: {len(missing)} samples (first indices: {head})"
            )

        predictions: list[list[str]] = [
            preds if preds is not None else []
            for preds in predictions_maybe
        ]

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

        if progress_path is not None and progress_path.exists():
            try:
                progress_path.unlink()
            except Exception as exc:  # pragma: no cover
                print(f"[eval] warning: failed to remove progress file={progress_path}: {exc!r}")

        return predictions, metrics


__all__ = ["SidNextItemJaxEvaluator", "evaluate_sid_next_item_jax"]
