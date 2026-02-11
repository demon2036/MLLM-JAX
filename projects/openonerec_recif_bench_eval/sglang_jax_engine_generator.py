from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from transformers import AutoConfig, AutoTokenizer


def _batch_iter(items: List[str], batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, len(items), batch_size):
        end = min(len(items), start + batch_size)
        yield start, end


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _normalize_top_k(value: Any) -> int:
    top_k = _as_int(value, -1)
    if top_k <= 0:
        return -1
    return top_k


def _load_engine_class():
    try:
        from sgl_jax.srt.entrypoints.engine import Engine

        return Engine
    except Exception:
        local_python = (
            Path(__file__).resolve().parents[2] / "workdirs" / "sglang-jax" / "python"
        )
        if local_python.is_dir() and str(local_python) not in sys.path:
            sys.path.insert(0, str(local_python))
        try:
            from sgl_jax.srt.entrypoints.engine import Engine

            return Engine
        except Exception as exc:
            raise RuntimeError(
                "Failed to import sglang-jax Engine. Ensure `sgl_jax` is importable "
                "(install sglang-jax or keep `workdirs/sglang-jax/python` available)."
            ) from exc


def _sum_output_logprob(meta_info: dict[str, Any]) -> float:
    raw = meta_info.get("output_token_logprobs") or []
    total = 0.0
    found = False
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 1:
            continue
        lp = _as_float(item[0], float("-inf"))
        if math.isfinite(lp):
            total += float(lp)
            found = True
    return total if found else float("-inf")


def _extract_token_probs_from_meta(
    meta_info: dict[str, Any],
    *,
    token_ids: Dict[str, int],
) -> Dict[str, float]:
    out: Dict[str, float] = {token: 0.0 for token in token_ids.keys()}
    raw = meta_info.get("output_token_ids_logprobs") or []
    if not raw:
        return out

    first_pos = raw[0]
    if not isinstance(first_pos, list):
        return out

    id_to_prob: Dict[int, float] = {}
    for item in first_pos:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        logprob = _as_float(item[0], float("-inf"))
        token_id = _as_int(item[1], -1)
        if token_id < 0:
            continue
        id_to_prob[token_id] = 0.0 if not math.isfinite(logprob) else float(math.exp(logprob))

    for token, tid in token_ids.items():
        out[token] = float(id_to_prob.get(int(tid), 0.0))

    return out


class SglangJaxEngineGenerator:
    """`sglang-jax` Engine backend compatible with OpenOneRec benchmark generator contract.

    This class intentionally avoids importing upstream benchmark modules at import time.
    The adapter class below (`BenchmarkSglangJaxEngineGenerator`) is responsible for inheriting
    from `benchmark.base_generator.Generator` when upstream benchmark is importable.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        torch_dtype: str | None = "bfloat16",
        trust_remote_code: bool = True,
        prefer_tpu: bool = True,
        batch_size: int = 8,
        max_batch_size: int = 64,
        beam_batch_size: int | None = None,
        engine_kwargs: dict[str, Any] | None = None,
        beam_emulation: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = model_name_or_path
        self.batch_size = int(batch_size)
        self.max_batch_size = int(max_batch_size)
        self.beam_batch_size = int(beam_batch_size) if beam_batch_size is not None else None

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=bool(trust_remote_code),
        )

        self.max_context_length: int | None = None
        try:
            model_cfg = AutoConfig.from_pretrained(
                model_name_or_path,
                trust_remote_code=bool(trust_remote_code),
            )
            max_pos = getattr(model_cfg, "max_position_embeddings", None)
            if max_pos is not None:
                max_pos_int = int(max_pos)
                if max_pos_int > 0:
                    self.max_context_length = max_pos_int
        except Exception:
            self.max_context_length = None

        beam_emulation_cfg = dict(beam_emulation or {})
        self.beam_emulation_temperature = _as_float(
            beam_emulation_cfg.get("temperature"),
            0.7,
        )
        self.beam_emulation_top_p = _as_float(
            beam_emulation_cfg.get("top_p"),
            0.95,
        )
        self.beam_emulation_top_k = _normalize_top_k(beam_emulation_cfg.get("top_k", 50))
        self.beam_emulation_deduplicate = bool(beam_emulation_cfg.get("deduplicate", True))

        cfg = dict(engine_kwargs or {})
        cfg.setdefault("model_path", model_name_or_path)
        cfg.setdefault("trust_remote_code", bool(trust_remote_code))
        cfg.setdefault("device", "tpu" if prefer_tpu else "cpu")
        if torch_dtype is not None:
            cfg.setdefault("dtype", str(torch_dtype))

        engine_class = _load_engine_class()

        self.engine_kwargs = cfg
        self.engine = engine_class(**cfg)
        self.num_params = self._resolve_num_params()

    def _resolve_num_params(self) -> float | None:
        try:
            info = self.engine.get_server_info()
        except Exception:
            return None
        if not isinstance(info, dict):
            return None

        scheduler = info.get("scheduler_info") if isinstance(info.get("scheduler_info"), dict) else None
        model_info_candidates: List[Any] = []
        if scheduler:
            model_info_candidates.append(scheduler)
        model_info_candidates.append(info)

        for payload in model_info_candidates:
            if not isinstance(payload, dict):
                continue
            for key in ("num_params", "model_num_params", "parameter_count"):
                if key in payload:
                    raw = payload.get(key)
                    try:
                        return float(raw)
                    except Exception:
                        continue
        return None

    def _resolve_batch_size(self, kwargs: dict[str, Any]) -> int:
        raw = kwargs.get("worker_batch_size") or kwargs.get("batch_size") or self.batch_size
        return max(1, min(int(raw), self.max_batch_size))

    def _clip_max_new_tokens_for_prompts(self, prompt_texts: List[str], requested_max_new: int) -> int:
        req = max(1, int(requested_max_new))
        if not self.max_context_length:
            return req

        try:
            max_prompt_tokens = 0
            for text in prompt_texts:
                token_count = len(self.tokenizer.encode(str(text), add_special_tokens=False))
                if token_count > max_prompt_tokens:
                    max_prompt_tokens = token_count
        except Exception:
            return req

        allowed = int(self.max_context_length) - int(max_prompt_tokens)
        if allowed <= 0:
            return 1
        return min(req, allowed)

    def _engine_generate(
        self,
        *,
        prompts: List[str],
        sampling_params: dict[str, Any],
        return_logprob: bool,
        token_ids_logprob: List[int] | None = None,
    ) -> List[dict[str, Any]]:
        ret = self.engine.generate(
            prompt=prompts,
            sampling_params=sampling_params,
            return_logprob=bool(return_logprob),
            token_ids_logprob=token_ids_logprob,
            stream=False,
        )
        if isinstance(ret, dict):
            return [ret]
        if isinstance(ret, list):
            return ret
        try:
            return list(ret)
        except Exception as exc:
            raise RuntimeError(f"Unexpected engine.generate return type: {type(ret)}") from exc

    def _build_sampling_params(
        self,
        *,
        kwargs: dict[str, Any],
        n: int = 1,
        stop_sequences: List[str],
        use_beam_emulation: bool,
    ) -> dict[str, Any]:
        max_new_tokens = _as_int(kwargs.get("max_new_tokens", 128), 128)
        max_new_tokens = max(0, max_new_tokens)

        repetition_penalty_raw = kwargs.get("repetition_penalty", 1.0)
        repetition_penalty = _as_float(repetition_penalty_raw, 1.0)

        if use_beam_emulation:
            sampling_params: dict[str, Any] = {
                "n": int(max(1, n)),
                "max_new_tokens": max_new_tokens,
                "temperature": float(max(self.beam_emulation_temperature, 1e-5)),
                "top_p": float(self.beam_emulation_top_p),
                "top_k": int(self.beam_emulation_top_k),
                "repetition_penalty": repetition_penalty,
            }
        else:
            do_sample = bool(kwargs.get("do_sample", True))
            temperature = _as_float(kwargs.get("temperature", 0.7), 0.7)
            top_p = _as_float(kwargs.get("top_p", 0.9), 0.9)
            top_k = _normalize_top_k(kwargs.get("top_k", -1))

            if not do_sample:
                temperature = 0.0
                top_p = 1.0
                top_k = 1

            sampling_params = {
                "n": int(max(1, n)),
                "max_new_tokens": max_new_tokens,
                "temperature": float(max(temperature, 0.0)),
                "top_p": float(top_p),
                "top_k": int(top_k),
                "repetition_penalty": repetition_penalty,
                "frequency_penalty": _as_float(kwargs.get("frequency_penalty", 0.0), 0.0),
                "presence_penalty": _as_float(kwargs.get("presence_penalty", 0.0), 0.0),
            }

        if stop_sequences:
            sampling_params["stop"] = stop_sequences
        return sampling_params

    def generate_standard(
        self,
        prompts: Dict[str, str],
        **kwargs: Any,
    ) -> tuple[Dict[str, List[str]], Dict[str, List[float]], Dict[str, Dict[str, List[Any]]]]:
        sample_ids = list(prompts.keys())
        prompt_texts = [prompts[sid] for sid in sample_ids]
        if not sample_ids:
            return {}, {}, {}

        num_beams_raw = kwargs.get("num_beams", None)
        num_beams = _as_int(num_beams_raw, 0) if num_beams_raw is not None else None
        batch_size = self._resolve_batch_size(kwargs)
        if num_beams is not None and self.beam_batch_size is not None and self.beam_batch_size > 0:
            batch_size = max(1, min(batch_size, int(self.beam_batch_size)))

        num_return_sequences = _as_int(kwargs.get("num_return_sequences", 1), 1)
        num_return_sequences = max(1, num_return_sequences)

        stop_sequences = kwargs.get("stop", []) or []
        if isinstance(stop_sequences, str):
            stop_sequences = [stop_sequences]
        stop_sequences = [str(s) for s in stop_sequences]

        use_beam_emulation = num_beams is not None
        if use_beam_emulation:
            num_beams = max(1, int(num_beams))
            n_candidates = min(num_return_sequences, num_beams)
        else:
            n_candidates = num_return_sequences

        want_logprobs = bool(
            kwargs.get("return_logprobs")
            or kwargs.get("output_scores")
            or kwargs.get("collect_logprobs")
            or use_beam_emulation
        )

        results: Dict[str, List[str]] = {}
        logprobs: Dict[str, List[float]] = {}
        mfu_stats: Dict[str, Dict[str, List[Any]]] = {}

        for start, end in _batch_iter(sample_ids, batch_size):
            batch_ids = sample_ids[start:end]
            batch_texts = prompt_texts[start:end]

            expanded_texts: List[str] = []
            expanded_owner: List[str] = []
            for sid, text in zip(batch_ids, batch_texts):
                for _ in range(n_candidates):
                    expanded_texts.append(text)
                    expanded_owner.append(sid)

            batch_kwargs = dict(kwargs)
            requested_max_new = _as_int(batch_kwargs.get("max_new_tokens", 128), 128)
            batch_kwargs["max_new_tokens"] = self._clip_max_new_tokens_for_prompts(
                expanded_texts,
                requested_max_new,
            )
            sampling_params = self._build_sampling_params(
                kwargs=batch_kwargs,
                n=1,
                stop_sequences=stop_sequences,
                use_beam_emulation=use_beam_emulation,
            )

            t0 = time.time()
            raw_outputs = self._engine_generate(
                prompts=expanded_texts,
                sampling_params=sampling_params,
                return_logprob=want_logprobs,
            )
            batch_dt = time.time() - t0

            expected = len(expanded_texts)
            if len(raw_outputs) != expected:
                raise RuntimeError(
                    "sglang-jax output count mismatch: "
                    f"expected {expected}, got {len(raw_outputs)} "
                    f"(batch={len(batch_ids)}, n={n_candidates}, expanded={len(expanded_texts)})"
                )

            grouped_outputs: Dict[str, List[dict[str, Any]]] = {sid: [] for sid in batch_ids}
            for sid, out in zip(expanded_owner, raw_outputs):
                grouped_outputs[sid].append(out)

            for sample_id in batch_ids:
                row_outputs = grouped_outputs.get(sample_id, [])
                candidates: List[tuple[str, float, int, int]] = []

                for item in row_outputs:
                    text = str(item.get("text", ""))
                    meta = item.get("meta_info") or {}
                    score = _sum_output_logprob(meta)
                    prompt_tokens = _as_int(meta.get("prompt_tokens", 0), 0)
                    completion_tokens = _as_int(
                        meta.get("completion_tokens", len(item.get("output_ids") or [])),
                        0,
                    )
                    candidates.append((text, score, prompt_tokens, completion_tokens))

                if use_beam_emulation:
                    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
                    if self.beam_emulation_deduplicate:
                        deduped: List[tuple[str, float, int, int]] = []
                        duplicated: List[tuple[str, float, int, int]] = []
                        seen_text: set[str] = set()
                        for candidate in candidates:
                            text = candidate[0]
                            if text in seen_text:
                                duplicated.append(candidate)
                            else:
                                deduped.append(candidate)
                                seen_text.add(text)
                        candidates = deduped + duplicated

                candidates = candidates[:n_candidates]
                decoded = [c[0] for c in candidates]

                results[sample_id] = decoded
                if want_logprobs:
                    logprobs[sample_id] = [float(c[1]) for c in candidates]

                prompt_tokens = int(candidates[0][2]) if candidates else 0
                output_tokens = int(sum(max(0, c[3]) for c in candidates))
                mfu_stats[sample_id] = {
                    "input_tokens": [prompt_tokens],
                    "output_tokens": [output_tokens],
                    "times": [batch_dt],
                }

        return results, logprobs, mfu_stats

    def extract_token_logprobs(
        self,
        prompts: Dict[str, str],
        target_tokens: List[str],
        **kwargs: Any,
    ) -> tuple[Dict[str, List[str]], Dict[str, List[float]], Dict[str, Dict[str, List[Any]]]]:
        sample_ids = list(prompts.keys())
        prompt_texts = [prompts[sid] for sid in sample_ids]
        if not sample_ids:
            return {}, {}, {}

        token_ids: Dict[str, int] = {}
        for token in target_tokens:
            ids = self.tokenizer.encode(str(token), add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"target token must map to exactly 1 token id: {token!r} -> {ids}")
            token_ids[str(token)] = int(ids[0])

        batch_size = self._resolve_batch_size(kwargs)
        sampling_params = {
            "n": 1,
            "max_new_tokens": 1,
            "temperature": _as_float(kwargs.get("temperature", 1.0), 1.0),
            "top_p": _as_float(kwargs.get("top_p", 1.0), 1.0),
            "top_k": _normalize_top_k(kwargs.get("top_k", -1)),
            "repetition_penalty": _as_float(kwargs.get("repetition_penalty", 1.0), 1.0),
            "frequency_penalty": _as_float(kwargs.get("frequency_penalty", 0.0), 0.0),
            "presence_penalty": _as_float(kwargs.get("presence_penalty", 0.0), 0.0),
        }

        results: Dict[str, List[str]] = {}
        mfu_stats: Dict[str, Dict[str, List[Any]]] = {}

        requested_token_ids = [int(v) for v in token_ids.values()]

        for start, end in _batch_iter(sample_ids, batch_size):
            batch_ids = sample_ids[start:end]
            batch_texts = prompt_texts[start:end]

            t0 = time.time()
            raw_outputs = self._engine_generate(
                prompts=batch_texts,
                sampling_params=sampling_params,
                return_logprob=True,
                token_ids_logprob=requested_token_ids,
            )
            batch_dt = time.time() - t0

            if len(raw_outputs) != len(batch_ids):
                raise RuntimeError(
                    "sglang-jax logprob output count mismatch: "
                    f"expected {len(batch_ids)}, got {len(raw_outputs)}"
                )

            for row_idx, sample_id in enumerate(batch_ids):
                item = raw_outputs[row_idx]
                meta = item.get("meta_info") or {}
                probs = _extract_token_probs_from_meta(meta, token_ids=token_ids)

                results[sample_id] = [json.dumps(probs, ensure_ascii=False)]
                mfu_stats[sample_id] = {
                    "input_tokens": [_as_int(meta.get("prompt_tokens", 0), 0)],
                    "output_tokens": [1],
                    "times": [batch_dt],
                }

        return results, {}, mfu_stats

    def get_hardware_info(self) -> Dict[str, Any]:
        try:
            import jax

            devices = list(jax.devices())
            device_kind = devices[0].device_kind if devices else "unknown"
            device_count = int(jax.device_count())
        except Exception:
            device_kind = "unknown"
            device_count = 1

        return {
            "gpu_model": device_kind,
            "gpu_count": device_count,
            "gpu_tflops": None,
            "tensor_parallel_size": int(self.engine_kwargs.get("tp_size", 1) or 1),
            "gpu_memory_total_gb": None,
        }


try:
    from benchmark.base_generator import Generator as _BenchmarkGenerator  # type: ignore
except Exception:  # pragma: no cover - only used when upstream benchmark is absent
    _BenchmarkGenerator = None  # type: ignore


class BenchmarkSglangJaxEngineGenerator(_BenchmarkGenerator if _BenchmarkGenerator is not None else object):
    """Adapter that plugs `SglangJaxEngineGenerator` into OpenOneRec benchmark runner."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if _BenchmarkGenerator is not None:
            super().__init__()
        self._impl = SglangJaxEngineGenerator(*args, **kwargs)
        self.model_name = self._impl.model_name
        self.num_params = getattr(self._impl, "num_params", None)

    def _generate_standard(self, prompts: Dict[str, str], **kwargs: Any):
        return self._impl.generate_standard(prompts, **kwargs)

    def extract_token_logprobs(self, prompts: Dict[str, str], target_tokens: List[str], **kwargs: Any):
        return self._impl.extract_token_logprobs(prompts, target_tokens, **kwargs)

    def get_hardware_info(self) -> Dict[str, Any]:
        return self._impl.get_hardware_info()


__all__ = ["SglangJaxEngineGenerator", "BenchmarkSglangJaxEngineGenerator"]
