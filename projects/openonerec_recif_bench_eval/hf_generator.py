from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _unset_socks_proxies() -> None:
    """Avoid httpx failing on `socks://...` proxies (HF hub uses httpx in some paths)."""
    for key in ("ALL_PROXY", "all_proxy"):
        value = os.environ.get(key)
        if value and value.strip().lower().startswith("socks://"):
            os.environ.pop(key, None)


def _as_torch_dtype(value: str | None) -> torch.dtype | None:
    if value is None:
        return None
    raw = str(value).strip().lower()
    if raw in {"auto", ""}:
        return None
    if raw in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if raw in {"fp16", "float16", "f16"}:
        return torch.float16
    if raw in {"fp32", "float32", "f32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {value!r}")


def _get_xla_device() -> torch.device | None:
    try:
        import torch_xla.core.xla_model as xm  # type: ignore
    except Exception:
        return None
    return xm.xla_device()


def _batch_iter(items: List[str], batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, len(items), batch_size):
        end = min(len(items), start + batch_size)
        yield start, end


def _find_first_token(tokens: torch.Tensor, targets: set[int]) -> Optional[int]:
    """Return the first index where tokens[idx] is in targets."""
    for idx, tok in enumerate(tokens.tolist()):
        if tok in targets:
            return idx
    return None


@contextmanager
def _torch_xla_eager_mode(enabled: bool):
    if not enabled:
        yield
        return

    try:
        import torch_xla.experimental as xla_exp  # type: ignore
    except Exception:
        yield
        return

    xla_exp.eager_mode(True)
    try:
        yield
    finally:
        xla_exp.eager_mode(False)


class TransformersGenerator:
    """HuggingFace Transformers generator compatible with OpenOneRec `benchmark.base_generator.Generator`.

    This class intentionally avoids importing OpenOneRec benchmark modules at import time so it can
    be imported before `sys.path` wiring. The runner is expected to wrap it in a subclass of
    `benchmark.base_generator.Generator` (see `projects/openonerec_recif_bench_eval/runner.py`).
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
    ) -> None:
        _unset_socks_proxies()

        self.model_name = model_name_or_path
        self.trust_remote_code = bool(trust_remote_code)
        self.torch_dtype = _as_torch_dtype(torch_dtype)
        self.batch_size = int(batch_size)
        self.max_batch_size = int(max_batch_size)

        device = _get_xla_device() if prefer_tpu else None
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=self.trust_remote_code,
        )
        # Some decoder-only tokenizers (including some Qwen variants) may ship without an explicit
        # pad token. For batched generation we need a pad_token_id; using eos_token is the
        # standard fallback and keeps behavior aligned with common HF defaults.
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            eos_token = getattr(self.tokenizer, "eos_token", None)
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_token is not None:
                self.tokenizer.pad_token = eos_token
            elif eos_token_id is not None:
                self.tokenizer.pad_token_id = int(eos_token_id)
            else:
                raise RuntimeError("Tokenizer is missing pad_token_id and eos_token_id; cannot batch prompts safely.")

        if getattr(self.tokenizer, "padding_side", None) is not None:
            # Decoder-only models generally expect left padding for correct position handling.
            self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
        )
        self.model.eval()
        self.model.to(self.device)

        self.num_params = float(sum(p.numel() for p in self.model.parameters()))

    def _encode_batch(self, texts: List[str]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
        )
        input_ids = encoded.get("input_ids")
        if input_ids is None or input_ids.ndim != 2:
            raise ValueError(f"tokenizer(...) must return input_ids[batch, seq], got: {type(input_ids)}")
        return {k: v.to(self.device) for k, v in encoded.items()}

    def _resolve_batch_size(self, kwargs: dict[str, Any]) -> int:
        raw = kwargs.get("worker_batch_size") or kwargs.get("batch_size") or self.batch_size
        return max(1, min(int(raw), self.max_batch_size))

    def _resolve_stop_token_ids(self, stop_sequences: List[str]) -> List[int]:
        ids: List[int] = []
        for stop in stop_sequences:
            token_ids = self.tokenizer.encode(stop, add_special_tokens=False)
            if len(token_ids) == 1:
                ids.append(int(token_ids[0]))
        return ids

    def generate_standard(
        self,
        prompts: Dict[str, str],
        **kwargs: Any,
    ) -> tuple[Dict[str, List[str]], Dict[str, List[float]], Dict[str, Dict[str, List[Any]]]]:
        """Generate text for prompts.

        Returns:
            (results, logprobs, mfu_stats)
        """
        sample_ids = list(prompts.keys())
        prompt_texts = [prompts[sid] for sid in sample_ids]

        batch_size = self._resolve_batch_size(kwargs)

        num_beams = kwargs.get("num_beams", None)
        num_return_sequences = int(kwargs.get("num_return_sequences", 1) or 1)
        max_new_tokens = int(kwargs.get("max_new_tokens", 128) or 128)
        repetition_penalty = kwargs.get("repetition_penalty", None)

        do_sample = bool(kwargs.get("do_sample", True))
        temperature = float(kwargs.get("temperature", 0.7) or 0.7)
        top_p = float(kwargs.get("top_p", 0.9) or 0.9)
        top_k_raw = kwargs.get("top_k", 0)
        try:
            top_k = int(top_k_raw)
        except Exception:
            top_k = 0
        if top_k < 0:
            top_k = 0

        stop_sequences = kwargs.get("stop", []) or []
        if isinstance(stop_sequences, str):
            stop_sequences = [stop_sequences]
        stop_token_ids = self._resolve_stop_token_ids(list(stop_sequences))
        stop_token_ids_set = set(stop_token_ids)

        results: Dict[str, List[str]] = {}
        logprobs: Dict[str, List[float]] = {}
        mfu_stats: Dict[str, Dict[str, List[Any]]] = {}

        # NOTE: `transformers.GenerationMixin.generate` can build very large XLA graphs when
        # `max_new_tokens` is large (e.g., RecIF-Bench `rec_reason`). Enabling XLA eager mode
        # avoids compiling an enormous single graph and prevents TPU compile OOM.
        use_xla_eager = self.device.type == "xla" and num_beams is None and max_new_tokens >= 512
        if use_xla_eager:
            print(f"[info] torch_xla eager_mode enabled (max_new_tokens={max_new_tokens}, device={self.device})")

        with _torch_xla_eager_mode(use_xla_eager):
            for start, end in _batch_iter(sample_ids, batch_size):
                batch_ids = sample_ids[start:end]
                batch_texts = prompt_texts[start:end]
                encoded = self._encode_batch(batch_texts)
                attention_mask = encoded.get("attention_mask")
                if attention_mask is None:
                    raise ValueError("tokenizer(...) must return attention_mask for batching")
                prompt_lengths = attention_mask.sum(dim=1).to(torch.long)

                gen_kwargs: Dict[str, Any] = {
                    "max_new_tokens": max_new_tokens,
                }
                if repetition_penalty is not None:
                    gen_kwargs["repetition_penalty"] = float(repetition_penalty)

                if num_beams is not None:
                    num_beams_int = int(num_beams)
                    want_scores = bool(
                        kwargs.get("return_logprobs") or kwargs.get("output_scores") or kwargs.get("collect_logprobs")
                    )
                    gen_kwargs.update(
                        num_beams=num_beams_int,
                        num_return_sequences=min(num_return_sequences, num_beams_int),
                        do_sample=False,
                        output_scores=want_scores,
                        return_dict_in_generate=True,
                    )
                else:
                    gen_kwargs.update(
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        num_return_sequences=num_return_sequences,
                        return_dict_in_generate=True,
                    )

                if stop_token_ids:
                    eos_ids = []
                    if getattr(self.tokenizer, "eos_token_id", None) is not None:
                        eos_ids.append(int(self.tokenizer.eos_token_id))
                    eos_ids.extend(stop_token_ids)
                    gen_kwargs["eos_token_id"] = sorted(set(eos_ids))

                t0 = time.time()
                # `torch.inference_mode()` is faster but can trigger issues with some model
                # implementations on XLA/TPU (e.g. buffer casting inside RoPE). Use `no_grad`
                # for maximum compatibility.
                with torch.no_grad():
                    out = self.model.generate(**encoded, **gen_kwargs)
                batch_dt = time.time() - t0

                sequences = out.sequences
                nseq = int(gen_kwargs.get("num_return_sequences", 1))
                sequences = sequences.view(len(batch_ids), nseq, -1)

                batch_scores = None
                if hasattr(out, "sequences_scores") and out.sequences_scores is not None:
                    batch_scores = out.sequences_scores.view(len(batch_ids), nseq).detach().to("cpu")

                for row_idx, sample_id in enumerate(batch_ids):
                    prompt_len = int(prompt_lengths[row_idx].item())
                    decoded: List[str] = []
                    seq_logprobs: List[float] = []

                    for seq_idx in range(nseq):
                        seq_tokens = sequences[row_idx, seq_idx, prompt_len:].detach().to("cpu")
                        if stop_token_ids_set:
                            cut = _find_first_token(seq_tokens, stop_token_ids_set)
                            if cut is not None:
                                seq_tokens = seq_tokens[:cut]
                        decoded.append(self.tokenizer.decode(seq_tokens, skip_special_tokens=True))

                        if batch_scores is not None:
                            seq_logprobs.append(float(batch_scores[row_idx, seq_idx].item()))

                    results[sample_id] = decoded
                    if seq_logprobs:
                        logprobs[sample_id] = seq_logprobs

                    output_tokens = sum(len(self.tokenizer.encode(text, add_special_tokens=False)) for text in decoded)
                    mfu_stats[sample_id] = {
                        "input_tokens": [prompt_len],
                        "output_tokens": [output_tokens],
                        "times": [batch_dt],
                    }

                # TPU/XLA needs explicit step marking in some setups; safe no-op on CPU/CUDA.
                try:
                    import torch_xla.core.xla_model as xm  # type: ignore

                    xm.mark_step()
                except Exception:
                    pass

        return results, logprobs, mfu_stats

    def extract_token_logprobs(
        self,
        prompts: Dict[str, str],
        target_tokens: List[str],
        **kwargs: Any,
    ) -> tuple[Dict[str, List[str]], Dict[str, List[float]], Dict[str, Dict[str, List[Any]]]]:
        """Extract next-token probabilities for target tokens.

        Returns a dict mapping sample_id -> ["{...json...}"].
        """
        sample_ids = list(prompts.keys())
        prompt_texts = [prompts[sid] for sid in sample_ids]

        batch_size = self._resolve_batch_size(kwargs)

        token_ids: Dict[str, int] = {}
        for token in target_tokens:
            ids = self.tokenizer.encode(str(token), add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"target token must map to 1 token id: {token!r} -> {ids}")
            token_ids[str(token)] = int(ids[0])

        results: Dict[str, List[str]] = {}
        mfu_stats: Dict[str, Dict[str, List[Any]]] = {}

        for start, end in _batch_iter(sample_ids, batch_size):
            batch_ids = sample_ids[start:end]
            batch_texts = prompt_texts[start:end]
            encoded = self._encode_batch(batch_texts)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                raise ValueError("tokenizer(...) must return attention_mask for batching")
            lengths = attention_mask.sum(dim=1).to(torch.long)

            t0 = time.time()
            with torch.no_grad():
                out = self.model(**encoded)
                logits = out.logits  # [batch, seq, vocab]
            batch_dt = time.time() - t0

            batch_size_actual = logits.shape[0]
            row = torch.arange(batch_size_actual, device=logits.device)
            last_pos = (lengths - 1).clamp(min=0)
            next_logits = logits[row, last_pos, :]
            probs = torch.softmax(next_logits, dim=-1)
            probs_cpu = probs.detach().to("cpu")

            for idx, sample_id in enumerate(batch_ids):
                payload = {tok: float(probs_cpu[idx, tid].item()) for tok, tid in token_ids.items()}
                results[sample_id] = [json.dumps(payload, ensure_ascii=False)]
                mfu_stats[sample_id] = {
                    "input_tokens": [int(lengths[idx].item())],
                    "output_tokens": [1],
                    "times": [batch_dt],
                }

            try:
                import torch_xla.core.xla_model as xm  # type: ignore

                xm.mark_step()
            except Exception:
                pass

        return results, {}, mfu_stats


try:
    from benchmark.base_generator import Generator as _BenchmarkGenerator  # type: ignore
except Exception:  # pragma: no cover - only used when upstream benchmark is absent
    _BenchmarkGenerator = None  # type: ignore


class BenchmarkTransformersGenerator(_BenchmarkGenerator if _BenchmarkGenerator is not None else object):
    """Adapter that plugs `TransformersGenerator` into OpenOneRec benchmark runner."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if _BenchmarkGenerator is not None:
            super().__init__()
        self._impl = TransformersGenerator(*args, **kwargs)
        self.model_name = self._impl.model_name
        self.num_params = getattr(self._impl, "num_params", None)

    def _generate_standard(self, prompts: Dict[str, str], **kwargs: Any):
        return self._impl.generate_standard(prompts, **kwargs)

    def extract_token_logprobs(self, prompts: Dict[str, str], target_tokens: List[str], **kwargs: Any):
        return self._impl.extract_token_logprobs(prompts, target_tokens, **kwargs)


__all__ = ["TransformersGenerator", "BenchmarkTransformersGenerator"]
