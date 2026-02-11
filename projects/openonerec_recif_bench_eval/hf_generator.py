from __future__ import annotations

import json
import os
import time
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


def _is_xla_compile_oom(exc: BaseException) -> bool:
    message = str(exc)
    return "RESOURCE_EXHAUSTED" in message and "XLA:TPU compile" in message


def _pad_left_token_ids(
    sequences: List[List[int]],
    *,
    pad_token_id: int,
    max_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("sequences must be non-empty")
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    else:
        max_len = int(max_len)
        if max_len <= 0:
            raise ValueError("max_len must be > 0")
        for seq in sequences:
            if len(seq) > max_len:
                raise ValueError(f"sequence length {len(seq)} exceeds max_len={max_len}")
    if max_len <= 0:
        raise ValueError("max_len must be > 0")

    input_ids = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for row, seq in enumerate(sequences):
        if not seq:
            continue
        input_ids[row, -len(seq) :] = torch.tensor(seq, dtype=torch.long)
        attention_mask[row, -len(seq) :] = 1
    return input_ids, attention_mask


def _pad_right_token_ids(
    sequences: List[List[int]],
    *,
    pad_token_id: int,
    max_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("sequences must be non-empty")
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    else:
        max_len = int(max_len)
        if max_len <= 0:
            raise ValueError("max_len must be > 0")
        for seq in sequences:
            if len(seq) > max_len:
                raise ValueError(f"sequence length {len(seq)} exceeds max_len={max_len}")
    if max_len <= 0:
        raise ValueError("max_len must be > 0")

    input_ids = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for row, seq in enumerate(sequences):
        if not seq:
            continue
        input_ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[row, : len(seq)] = 1
    return input_ids, attention_mask


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

    def _generate_greedy_autoregressive_xla(
        self,
        prompt_text: str,
        *,
        max_new_tokens: int,
        end_token_ids: set[int],
        repetition_penalty: float | None,
    ) -> tuple[list[int], int, float]:
        """Greedy decode on XLA with `xm.mark_step()` to avoid unbounded graph growth.

        This intentionally runs *one prompt at a time* to avoid padding/masks. OpenOneRec's
        reported metrics are batch-size invariant, so this is safe as an execution strategy.
        """

        try:
            import torch_xla.core.xla_model as xm  # type: ignore
        except Exception as exc:  # pragma: no cover - only used on TPU
            raise RuntimeError("XLA greedy path requested but torch_xla is unavailable") from exc

        encoded_cpu = self.tokenizer(prompt_text, return_tensors="pt")
        input_ids = encoded_cpu.get("input_ids")
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("tokenizer(...) must return input_ids[1, seq_len] for single prompt")

        input_len = int(input_ids.shape[1])
        input_ids = input_ids.to(self.device)

        max_positions = int(getattr(self.model.config, "max_position_embeddings", input_len + int(max_new_tokens)))
        max_cache_len = min(int(input_len + int(max_new_tokens)), max_positions)

        # Prefer StaticCache to avoid per-step shape growth on XLA (DynamicCache uses torch.cat and changes
        # cache shapes every token, which is very slow on TPU). StaticCache requires building an explicit
        # causal mask for prefill, so we only enable it when the mask size is reasonable.
        use_static_cache = False
        static_cache_max_mask_elems = int(os.environ.get("OPENONEREC_XLA_STATIC_CACHE_MAX_MASK_ELEMS", "64000000"))
        if static_cache_max_mask_elems > 0 and input_len * max_cache_len <= static_cache_max_mask_elems:
            use_static_cache = True

        past = None
        cache_position = None
        next_cache_pos = input_len
        if use_static_cache:
            try:
                from transformers.cache_utils import StaticCache  # type: ignore
            except Exception:
                use_static_cache = False
            else:
                model_dtype = getattr(self.model, "dtype", torch.float32)
                try:
                    # transformers<=4.52.0 (TPU env) requires max_batch_size.
                    past = StaticCache(
                        self.model.config,
                        max_batch_size=1,
                        max_cache_len=max_cache_len,
                        device=self.device,
                        dtype=model_dtype,
                    )
                except TypeError:
                    # Newer transformers moved to a different signature.
                    past = StaticCache(self.model.config, max_cache_len=max_cache_len)
                cache_position = torch.arange(0, input_len, device=self.device)

        cache_mode = "static" if use_static_cache else "dynamic"
        mask_elems = int(input_len * max_cache_len)
        print(f"[info] XLA greedy cache={cache_mode} input_len={input_len} max_cache_len={max_cache_len} mask_elems={mask_elems}")

        seen_mask = None
        penalty = None
        if repetition_penalty is not None:
            penalty = float(repetition_penalty)
        if penalty is not None and abs(penalty - 1.0) > 1e-6:
            vocab_size = int(getattr(self.model.config, "vocab_size"))
            seen_mask = torch.zeros((vocab_size,), dtype=torch.bool, device=self.device)
            # Include prompt tokens in repetition penalty, matching HF `RepetitionPenaltyLogitsProcessor`.
            seen_mask.index_fill_(0, input_ids[0], True)

        end_ids = sorted(end_token_ids)
        end_ids_tensor = (
            torch.tensor(end_ids, dtype=torch.long, device=self.device) if end_ids else None
        )

        t0 = time.time()

        cur_input = input_ids
        done = torch.zeros((1,), dtype=torch.bool, device=self.device)
        pad_id = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)
        token_buffer = torch.full((int(max_new_tokens),), pad_id, dtype=torch.long, device=self.device)
        steps_done = 0

        # Tune execution: mark_step every N tokens, and sync for early-stop checks at the same cadence.
        mark_step_interval = int(os.environ.get("OPENONEREC_XLA_MARK_STEP_INTERVAL", "8"))
        if mark_step_interval <= 0:
            mark_step_interval = 1

        with torch.no_grad():
            while steps_done < int(max_new_tokens):
                out = self.model(
                    input_ids=cur_input,
                    past_key_values=past,
                    use_cache=True,
                    cache_position=cache_position,
                )
                past = out.past_key_values

                logits = out.logits[:, -1, :]
                if seen_mask is not None and penalty is not None:
                    neg = (logits < 0) & seen_mask
                    pos = (logits > 0) & seen_mask
                    logits = torch.where(neg, logits * penalty, logits)
                    logits = torch.where(pos, logits / penalty, logits)

                next_token = torch.argmax(logits, dim=-1)  # [1]
                token_buffer[steps_done] = next_token[0]

                if seen_mask is not None:
                    seen_mask.index_fill_(0, next_token, True)

                if end_ids_tensor is not None:
                    is_end = (next_token[..., None] == end_ids_tensor).any(dim=-1)
                    done = done | is_end

                cur_input = next_token.view(1, 1)
                if use_static_cache:
                    cache_position = torch.tensor([next_cache_pos], device=self.device)
                    next_cache_pos += 1
                else:
                    cache_position = None
                steps_done += 1

                if steps_done % mark_step_interval == 0:
                    xm.mark_step()
                    if bool(done.cpu().item()):
                        break

        xm.mark_step()

        tokens_cpu = token_buffer[:steps_done].detach().to("cpu").tolist()
        cut = None
        if end_token_ids:
            for idx, tok in enumerate(tokens_cpu):
                if int(tok) in end_token_ids:
                    cut = idx
                    break
        if cut is not None:
            tokens_cpu = tokens_cpu[:cut]

        dt = time.time() - t0
        return [int(tok) for tok in tokens_cpu], input_len, dt

    def _generate_greedy_autoregressive_xla_bucketed(
        self,
        sample_ids: List[str],
        prompt_texts: List[str],
        *,
        max_new_tokens: int,
        end_token_ids: set[int],
        repetition_penalty: float | None,
        micro_batch_size: int,
        pad_token_id: int,
    ) -> tuple[Dict[str, List[str]], Dict[str, Dict[str, List[Any]]]]:
        """XLA greedy decode for many prompts with a bounded set of prompt shapes.

        This groups prompts by a padded bucket length (default: multiple of 512 tokens) to avoid compiling one XLA graph
        per distinct prompt length. For very long prompts, padding would create an enormous causal mask; those are
        processed without padding.

        Returns:
            (results, mfu_stats)
        """

        try:
            import torch_xla.core.xla_model as xm  # type: ignore
        except Exception as exc:  # pragma: no cover - only used on TPU
            raise RuntimeError("XLA greedy bucketed path requested but torch_xla is unavailable") from exc

        if len(sample_ids) != len(prompt_texts):
            raise ValueError("sample_ids and prompt_texts length mismatch")
        if not sample_ids:
            return {}, {}

        max_new_tokens_int = int(max_new_tokens)
        if max_new_tokens_int <= 0:
            return {sid: [""] for sid in sample_ids}, {sid: {"input_tokens": [0], "output_tokens": [0], "times": [0.0]} for sid in sample_ids}

        bucket_size = int(os.environ.get("OPENONEREC_XLA_GREEDY_BUCKET_SIZE", "512") or "512")
        if bucket_size <= 0:
            bucket_size = 1
        pad_max_len = int(os.environ.get("OPENONEREC_XLA_GREEDY_PAD_MAX_LEN", "12288") or "12288")
        if pad_max_len <= 0:
            pad_max_len = 0

        mark_step_interval = int(os.environ.get("OPENONEREC_XLA_MARK_STEP_INTERVAL", "8") or "8")
        if mark_step_interval <= 0:
            mark_step_interval = 1

        max_positions = int(getattr(self.model.config, "max_position_embeddings", 0) or 0)
        if max_positions <= 0:
            max_positions = 40960

        # Tokenize once on CPU to avoid padding all prompts to the global max length.
        token_ids_list: List[List[int]] = []
        true_lens: List[int] = []
        for text in prompt_texts:
            token_ids = self.tokenizer(text, add_special_tokens=True).get("input_ids")
            if token_ids is None:
                raise ValueError("tokenizer(...) missing input_ids")
            token_ids = [int(t) for t in token_ids]
            if len(token_ids) > max_positions:
                # Mirror common max-length truncation behavior: keep the most recent tokens.
                token_ids = token_ids[-max_positions:]
            token_ids_list.append(token_ids)
            true_lens.append(len(token_ids))

        # Group indices by padded bucket length (or exact length for long prompts).
        bucket_to_indices: Dict[int, List[int]] = {}
        for idx, seq_len in enumerate(true_lens):
            if pad_max_len and seq_len > pad_max_len:
                bucket_len = seq_len
            else:
                bucket_len = ((seq_len + bucket_size - 1) // bucket_size) * bucket_size
                bucket_len = min(bucket_len, max_positions)
            if bucket_len < seq_len:
                bucket_len = seq_len
            bucket_to_indices.setdefault(int(bucket_len), []).append(idx)

        if repetition_penalty is not None:
            repetition_penalty = float(repetition_penalty)

        end_ids = sorted(end_token_ids)
        end_ids_tensor = torch.tensor(end_ids, dtype=torch.long, device=self.device) if end_ids else None

        results: Dict[str, List[str]] = {}
        mfu_stats: Dict[str, Dict[str, List[Any]]] = {}

        vocab_size = int(getattr(self.model.config, "vocab_size", 0) or 0)
        if vocab_size <= 0 and repetition_penalty is not None and abs(repetition_penalty - 1.0) > 1e-6:
            raise ValueError("model.config.vocab_size is required for repetition_penalty")

        print(
            "[info] XLA greedy bucketed enabled"
            f" (bucket_size={bucket_size}, pad_max_len={pad_max_len}, micro_batch_size={micro_batch_size})"
        )

        # Process buckets from short to long to improve compile cache reuse.
        for bucket_len in sorted(bucket_to_indices):
            indices = bucket_to_indices[bucket_len]
            if not indices:
                continue

            bucket_micro_bs = 1 if bucket_len > pad_max_len else max(1, int(micro_batch_size))

            print(f"[info] XLA greedy bucket seq_len={bucket_len} samples={len(indices)} micro_bs={bucket_micro_bs}")

            for start in range(0, len(indices), bucket_micro_bs):
                batch_indices = indices[start : start + bucket_micro_bs]
                # Keep a stable micro-batch shape by padding with the last entry.
                if len(batch_indices) < bucket_micro_bs:
                    batch_indices = batch_indices + [batch_indices[-1]] * (bucket_micro_bs - len(batch_indices))

                batch_sample_ids = [sample_ids[i] for i in batch_indices]
                batch_token_ids = [token_ids_list[i] for i in batch_indices]
                batch_true_lens = [len(seq) for seq in batch_token_ids]

                # Right padding allows us to skip `attention_mask` during prefill (pads are in the future under
                # causal attention). We still build a decode-time mask to ignore padded KV positions.
                input_ids_cpu, _ = _pad_right_token_ids(
                    batch_token_ids,
                    pad_token_id=pad_token_id,
                    max_len=bucket_len,
                )

                input_ids = input_ids_cpu.to(self.device)

                # Cap new tokens by available cache capacity (StaticCache indices are bounded).
                max_cache_len = min(bucket_len + max_new_tokens_int, max_positions)
                effective_max_new = max(0, max_cache_len - bucket_len)
                if effective_max_new <= 0:
                    for sid, prompt_len in zip(batch_sample_ids, batch_true_lens):
                        results[sid] = [""]
                        mfu_stats[sid] = {"input_tokens": [prompt_len], "output_tokens": [0], "times": [0.0]}
                    continue

                penalty = repetition_penalty if repetition_penalty is not None else None
                use_penalty = penalty is not None and abs(float(penalty) - 1.0) > 1e-6
                seen_mask = None
                if use_penalty:
                    seen_mask_cpu = torch.zeros((bucket_micro_bs, vocab_size), dtype=torch.bool)
                    for row, seq in enumerate(batch_token_ids):
                        seen_mask_cpu[row].index_fill_(0, torch.tensor(seq, dtype=torch.long), True)
                    seen_mask = seen_mask_cpu.to(self.device)

                t0 = time.time()

                # Prefill with DynamicCache (avoids StaticCache prefill mask scaling with max_cache_len).
                # We intentionally avoid `attention_mask` here: with right padding, pads are in the future for real
                # prompt tokens under the causal mask, so they do not affect the logits for the last real token.
                print(f"[info] XLA greedy prefill start seq_len={bucket_len} micro_bs={bucket_micro_bs}", flush=True)
                with torch.no_grad():
                    prefill_out = self.model(
                        input_ids=input_ids,
                        use_cache=True,
                    )

                xm.mark_step()
                print(f"[info] XLA greedy prefill done dt={time.time() - t0:.2f}s", flush=True)

                dyn_past = prefill_out.past_key_values
                logits_full = prefill_out.logits  # [batch, bucket_len, vocab]
                true_len = torch.tensor(batch_true_lens, device=self.device, dtype=torch.long)
                last_pos = (true_len - 1).clamp(min=0)
                row_ids = torch.arange(bucket_micro_bs, device=self.device)
                logits = logits_full[row_ids, last_pos, :]

                if use_penalty and seen_mask is not None and penalty is not None:
                    neg = (logits < 0) & seen_mask
                    pos = (logits > 0) & seen_mask
                    logits = torch.where(neg, logits * penalty, logits)
                    logits = torch.where(pos, logits / penalty, logits)

                next_token = torch.argmax(logits, dim=-1)  # [batch]

                kv_positions = torch.arange(max_cache_len, device=self.device).unsqueeze(0)
                # Mask out right-padding positions in the prefilled cache; allow future cache slots for decoding.
                attention_mask_decode = (kv_positions < true_len.unsqueeze(1)) | (kv_positions >= bucket_len)
                attention_mask_decode = attention_mask_decode.to(torch.bool)

                token_buffer = torch.full(
                    (bucket_micro_bs, effective_max_new),
                    int(pad_token_id),
                    dtype=torch.long,
                    device=self.device,
                )

                done = torch.zeros((bucket_micro_bs,), dtype=torch.bool, device=self.device)

                # Copy DynamicCache -> StaticCache to keep decode shapes fixed.
                static_past = None
                try:
                    from transformers.cache_utils import StaticCache  # type: ignore
                except Exception:
                    static_past = None
                else:
                    if dyn_past is not None and hasattr(dyn_past, "layers"):
                        static_past = StaticCache(self.model.config, max_cache_len=max_cache_len)
                        prefill_cache_position = torch.arange(bucket_len, device=self.device)
                        for layer_idx, layer in enumerate(getattr(dyn_past, "layers", [])):
                            keys = getattr(layer, "keys", None)
                            values = getattr(layer, "values", None)
                            if keys is None or values is None:
                                static_past = None
                                break
                            static_past.update(
                                keys,
                                values,
                                layer_idx,
                                cache_kwargs={"cache_position": prefill_cache_position},
                            )

                if static_past is None:
                    # Fallback: reuse the legacy single-prompt implementation for this micro-batch.
                    # This keeps correctness, but may compile more shapes.
                    for sid, text in zip(batch_sample_ids, [prompt_texts[i] for i in batch_indices]):
                        tok_ids, prompt_len, dt = self._generate_greedy_autoregressive_xla(
                            text,
                            max_new_tokens=effective_max_new,
                            end_token_ids=end_token_ids,
                            repetition_penalty=repetition_penalty,
                        )
                        results[sid] = [self.tokenizer.decode(tok_ids, skip_special_tokens=True)]
                        mfu_stats[sid] = {"input_tokens": [prompt_len], "output_tokens": [len(tok_ids)], "times": [dt]}
                    continue

                xm.mark_step()

                # Decode loop: we already have token0 in `next_token`, feed it to get token1, etc.
                position_next = true_len
                cache_pos = bucket_len
                steps_done = 0

                with torch.no_grad():
                    while steps_done < effective_max_new:
                        token_buffer[:, steps_done] = next_token

                        if use_penalty and seen_mask is not None:
                            seen_mask.scatter_(1, next_token[:, None], True)

                        if end_ids_tensor is not None:
                            done = done | (next_token[..., None] == end_ids_tensor).any(dim=-1)

                        steps_done += 1

                        if steps_done >= effective_max_new:
                            break

                        if steps_done % mark_step_interval == 0:
                            xm.mark_step()
                            if bool(done.all().cpu().item()):
                                break

                        out = self.model(
                            input_ids=next_token.view(bucket_micro_bs, 1),
                            attention_mask=attention_mask_decode,
                            position_ids=position_next.view(bucket_micro_bs, 1),
                            past_key_values=static_past,
                            use_cache=True,
                            cache_position=torch.tensor([cache_pos], device=self.device),
                        )
                        cache_pos += 1
                        position_next += 1

                        logits = out.logits[:, -1, :]
                        if use_penalty and seen_mask is not None and penalty is not None:
                            neg = (logits < 0) & seen_mask
                            pos = (logits > 0) & seen_mask
                            logits = torch.where(neg, logits * penalty, logits)
                            logits = torch.where(pos, logits / penalty, logits)
                        next_token = torch.argmax(logits, dim=-1)

                xm.mark_step()

                tokens_cpu = token_buffer[:, :steps_done].detach().to("cpu").tolist()
                dt = time.time() - t0

                for row, sid in enumerate(batch_sample_ids):
                    seq = [int(tok) for tok in tokens_cpu[row]]
                    cut = None
                    if end_token_ids:
                        for i, tok in enumerate(seq):
                            if tok in end_token_ids:
                                cut = i
                                break
                    if cut is not None:
                        seq = seq[:cut]

                    results[sid] = [self.tokenizer.decode(seq, skip_special_tokens=True)]
                    mfu_stats[sid] = {
                        "input_tokens": [int(batch_true_lens[row])],
                        "output_tokens": [len(seq)],
                        "times": [dt],
                    }

        return results, mfu_stats

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
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        end_token_ids: set[int] = set(stop_token_ids)
        if eos_token_id is not None:
            end_token_ids.add(int(eos_token_id))

        results: Dict[str, List[str]] = {}
        logprobs: Dict[str, List[float]] = {}
        mfu_stats: Dict[str, Dict[str, List[Any]]] = {}

        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            raise RuntimeError("Tokenizer is missing pad_token_id")
        pad_token_id_int = int(pad_token_id)

        disable_xla_greedy = str(os.environ.get("OPENONEREC_DISABLE_XLA_GREEDY", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        use_xla_autoregressive_greedy = (
            self.device.type == "xla"
            and not disable_xla_greedy
            and num_beams is None
            and not do_sample
            and num_return_sequences == 1
            and max_new_tokens >= 128
        )
        if disable_xla_greedy and self.device.type == "xla":
            print("[info] XLA greedy loop disabled by OPENONEREC_DISABLE_XLA_GREEDY")
        elif use_xla_autoregressive_greedy:
            print(f"[info] XLA greedy loop enabled (max_new_tokens={max_new_tokens}, device={self.device})")
        if use_xla_autoregressive_greedy:
            greedy_results, greedy_mfu = self._generate_greedy_autoregressive_xla_bucketed(
                sample_ids,
                prompt_texts,
                max_new_tokens=max_new_tokens,
                end_token_ids=end_token_ids,
                repetition_penalty=float(repetition_penalty) if repetition_penalty is not None else None,
                micro_batch_size=batch_size,
                pad_token_id=pad_token_id_int,
            )
            return greedy_results, {}, greedy_mfu

        for start, end in _batch_iter(sample_ids, batch_size):
            batch_ids = sample_ids[start:end]
            batch_texts = prompt_texts[start:end]

            encoded = self._encode_batch(batch_texts)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                raise ValueError("tokenizer(...) must return attention_mask for batching")
            prompt_lengths = attention_mask.sum(dim=1).to(torch.long)
            padded_prompt_len = int(encoded["input_ids"].shape[1])

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
                if eos_token_id is not None:
                    eos_ids.append(int(eos_token_id))
                eos_ids.extend(stop_token_ids)
                gen_kwargs["eos_token_id"] = sorted(set(eos_ids))

            t0 = time.time()
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
                    # With left padding, generation starts after the padded prompt length, not after the true length.
                    seq_tokens = sequences[row_idx, seq_idx, padded_prompt_len:].detach().to("cpu")
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

            # With left padding, the last position is always the last real prompt token.
            next_logits = logits[:, -1, :]
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
