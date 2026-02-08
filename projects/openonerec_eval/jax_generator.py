from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from plugins.sample.backends.mllm_jax_sampler import get_model
from plugins.sample.constraints.sid_trie import SidTrie
from plugins.sample.decoding.sid3_constrained_beam_search import (
    BeamSearchOutput,
    constrained_beam_search_sid3_prefill,
)
from plugins.training.core.checkpoint.msgpack import load_checkpoint
from plugins.training.sft.jax.train import create_mesh_from_config

SUPPORTED_RECOMMENDATION_TASKS = {"video", "ad", "product", "label_cond", "interactive"}

_PREFILL_BUCKETS = (128, 256, 512, 1024, 2048, 4096, 8192)


class OpenOneRecJaxGenerator:
    """JAX recommendation-task generator using batched constrained SID3 beam search."""

    def __init__(self, *, base_model: str, generation_cfg: Any, jax_cfg: Any, benchmark_data_dir: str):
        self.base_model = str(base_model)
        self.batch_size = int(getattr(generation_cfg, "batch_size", 1))
        self.num_beams = int(getattr(generation_cfg, "num_beams", 4))
        self.num_return_sequences = int(getattr(generation_cfg, "num_return_sequences", 4))
        self.max_new_tokens = int(getattr(generation_cfg, "max_new_tokens", 3))
        self.max_prompt_tokens = int(getattr(generation_cfg, "max_prompt_tokens", 1024) or 1024)
        self.temperature = float(getattr(generation_cfg, "temperature", 1.0) or 1.0)
        self.top_p = float(getattr(generation_cfg, "top_p", 1.0) or 1.0)
        self.top_k = int(getattr(generation_cfg, "top_k", 50) or 50)
        self.prompt_token = str(getattr(generation_cfg, "prompt_token", "") or "")
        self.params_checkpoint_path = getattr(generation_cfg, "params_checkpoint_path", None)
        self.sid_second_cap = int(getattr(generation_cfg, "sid_second_cap", 128) or 128)
        self.sid_third_cap = int(getattr(generation_cfg, "sid_third_cap", 64) or 64)

        if self.sid_second_cap <= 0:
            raise ValueError(f"sid_second_cap must be > 0, got {self.sid_second_cap}")
        if self.sid_third_cap <= 0:
            raise ValueError(f"sid_third_cap must be > 0, got {self.sid_third_cap}")
        if self.max_prompt_tokens <= 0:
            raise ValueError(f"max_prompt_tokens must be > 0, got {self.max_prompt_tokens}")

        self.benchmark_data_dir = Path(benchmark_data_dir).expanduser().resolve()
        self.max_cache_length = int(getattr(jax_cfg, "max_cache_length", 512))
        self._trie_cache: dict[str, SidTrie] = {}
        self._beam_jit_cache: dict[tuple[str, int, int], Any] = {}

        if self.num_return_sequences > self.num_beams:
            raise ValueError("num_return_sequences cannot be greater than num_beams")

        # get_model() reads param dtype from env variable.
        os.environ["MLLM_JAX_PARAM_DTYPE"] = str(getattr(jax_cfg, "param_dtype", "float32"))
        self.mesh = create_mesh_from_config(str(getattr(jax_cfg, "mesh_shape", "1,-1,1")))
        self.model, self.params, self.tokenizer = get_model(mesh=self.mesh, model_path=self.base_model)

        if self.params_checkpoint_path:
            checkpoint_path = Path(str(self.params_checkpoint_path)).expanduser().resolve()
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"params_checkpoint_path not found: {checkpoint_path}")
            checkpoint = load_checkpoint(str(checkpoint_path))
            ckpt_params = checkpoint.get("params")
            if ckpt_params is None:
                raise KeyError(f"Checkpoint has no 'params' key: {checkpoint_path}")

            model_vocab_size = self._infer_vocab_size_from_params(self.params)
            ckpt_vocab_size = self._infer_vocab_size_from_params(ckpt_params)
            if (
                model_vocab_size is not None
                and ckpt_vocab_size is not None
                and int(model_vocab_size) != int(ckpt_vocab_size)
            ):
                self.model.config.vocab_size = int(ckpt_vocab_size)
                print(
                    "[eval] align model vocab_size to checkpoint "
                    f"({int(model_vocab_size)} -> {int(ckpt_vocab_size)})"
                )

            self.params = serialization.from_state_dict(self.params, ckpt_params)

    @staticmethod
    def _infer_vocab_size_from_params(params: Any) -> int | None:
        try:
            emb = params["model"]["embed_tokens"]["embedding"]
            emb_shape = np.asarray(emb).shape
            if len(emb_shape) >= 1:
                return int(emb_shape[0])
        except Exception:
            pass

        try:
            lm_head = params["lm_head"]["kernel"]
            lm_shape = np.asarray(lm_head).shape
            if len(lm_shape) == 2:
                return int(max(lm_shape[0], lm_shape[1]))
        except Exception:
            pass

        return None

    @staticmethod
    def _mapping_filename_for_task(task_name: str) -> str:
        return "sid2iid.json" if str(task_name) == "product" else "sid2pid.json"

    @staticmethod
    def _decode_sid_code(code: int) -> tuple[int, int, int]:
        m1 = 8192 * 8192
        m2 = 8192

        c1 = int(code // m1)
        rem = int(code % m1)
        c2 = int(rem // m2)
        c3 = int(rem % m2)

        if not (0 <= c1 < 8192 and 0 <= c2 < 8192 and 0 <= c3 < 8192):
            raise ValueError(f"Decoded SID code out of range for code={int(code)}: ({c1}, {c2}, {c3})")

        return c1, c2, c3

    def _build_trie_for_task(self, task_name: str) -> SidTrie:
        task_key = str(task_name)
        cached = self._trie_cache.get(task_key)
        if cached is not None:
            return cached

        mapping_path = self.benchmark_data_dir / self._mapping_filename_for_task(task_key)
        if not mapping_path.exists():
            raise FileNotFoundError(f"SID mapping file not found for task={task_key!r}: {mapping_path}")

        with mapping_path.open("r", encoding="utf-8") as f:
            mapping = json.load(f)

        if not isinstance(mapping, dict):
            raise TypeError(
                f"SID mapping file must contain a JSON object for task={task_key!r}: {mapping_path}"
            )

        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            raise ValueError("Tokenizer eos_token_id is required for constrained SID3 decoding")

        unk_token_id = getattr(self.tokenizer, "unk_token_id", None)
        unk_token_id_i = None if unk_token_id is None else int(unk_token_id)

        def _token_to_id(token: str, *, raw_key: Any) -> int:
            token_id_raw = self.tokenizer.convert_tokens_to_ids(token)
            if token_id_raw is None:
                raise ValueError(f"Tokenizer returned None id for token={token!r} key={raw_key!r}")
            token_id = int(token_id_raw)
            if token_id < 0:
                raise ValueError(f"Tokenizer returned invalid id={token_id} for token={token!r} key={raw_key!r}")
            if unk_token_id_i is not None and token_id == unk_token_id_i:
                raise ValueError(
                    f"Tokenizer token mapped to unk_token_id for token={token!r} key={raw_key!r}; "
                    f"unk_token_id={unk_token_id_i}"
                )
            return token_id

        first: set[int] = set()
        second_scores: dict[int, dict[int, float]] = {}
        third_scores: dict[tuple[int, int], dict[int, float]] = {}

        for raw_key, raw_value in mapping.items():
            try:
                sid_code = int(raw_key)
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid SID mapping key (not int-convertible): {raw_key!r}") from exc

            c1, c2, c3 = self._decode_sid_code(sid_code)

            tok1 = f"<s_a_{c1}>"
            tok2 = f"<s_b_{c2}>"
            tok3 = f"<s_c_{c3}>"

            t1 = _token_to_id(tok1, raw_key=raw_key)
            t2 = _token_to_id(tok2, raw_key=raw_key)
            t3 = _token_to_id(tok3, raw_key=raw_key)

            score = 1.0
            if isinstance(raw_value, list) and raw_value and isinstance(raw_value[0], dict):
                first_item = raw_value[0]
                score_raw = first_item.get("count_after_downsample", first_item.get("count", 1))
                try:
                    score = float(score_raw)
                except (TypeError, ValueError):
                    score = 1.0

            first.add(t1)
            second_for_t1 = second_scores.setdefault(t1, {})
            second_for_t1[t2] = float(second_for_t1.get(t2, 0.0)) + float(score)

            third_key = (t1, t2)
            third_for_pair = third_scores.setdefault(third_key, {})
            third_for_pair[t3] = float(third_for_pair.get(t3, 0.0)) + float(score)

        if not first:
            raise ValueError(f"No valid SID keys found in mapping file: {mapping_path}")

        def _select_top_tokens(score_map: dict[int, float], cap: int) -> list[int]:
            ranked = sorted(score_map.items(), key=lambda item: (-float(item[1]), int(item[0])))
            return [int(token_id) for token_id, _ in ranked[: int(cap)]]

        selected_second: dict[int, list[int]] = {}
        selected_third: dict[tuple[int, int], list[int]] = {}
        for t1 in sorted(first):
            selected_t2 = _select_top_tokens(second_scores.get(int(t1), {}), int(self.sid_second_cap))
            selected_second[int(t1)] = selected_t2
            for t2 in selected_t2:
                selected_t3 = _select_top_tokens(
                    third_scores.get((int(t1), int(t2)), {}),
                    int(self.sid_third_cap),
                )
                selected_third[(int(t1), int(t2))] = selected_t3

        pad_id = -1
        first_ids = np.asarray(sorted(selected_second.keys()), dtype=np.int32)
        second_keys = first_ids

        max_second = max(len(selected_second.get(int(t1), [])) for t1 in second_keys)
        if int(max_second) <= 0:
            raise ValueError(f"No second-level SID transitions found in mapping file: {mapping_path}")

        second_table = np.full((len(second_keys), int(max_second)), int(pad_id), dtype=np.int32)
        for i, t1 in enumerate(second_keys):
            vals = selected_second.get(int(t1), [])
            if vals:
                second_table[i, : len(vals)] = np.asarray(vals, dtype=np.int32)

        max_third = 0
        for i, t1 in enumerate(second_keys):
            for j in range(int(max_second)):
                t2 = int(second_table[i, j])
                if t2 == int(pad_id):
                    continue
                max_third = max(max_third, len(selected_third.get((int(t1), t2), [])))

        if int(max_third) <= 0:
            raise ValueError(f"No third-level SID transitions found in mapping file: {mapping_path}")

        third_table = np.full((len(second_keys), int(max_second), int(max_third)), int(pad_id), dtype=np.int32)
        for i, t1 in enumerate(second_keys):
            for j in range(int(max_second)):
                t2 = int(second_table[i, j])
                if t2 == int(pad_id):
                    continue
                vals = selected_third.get((int(t1), t2), [])
                if vals:
                    third_table[i, j, : len(vals)] = np.asarray(vals, dtype=np.int32)

        print(
            f"[eval] trie task={task_key} first={len(first_ids)} "
            f"max_second={int(max_second)} max_third={int(max_third)}"
        )

        trie = SidTrie(
            pad_id=int(pad_id),
            eos_token_id=int(eos_token_id),
            vocab_size=int(len(self.tokenizer)),
            first_ids=first_ids,
            second_keys=second_keys,
            second_table=second_table,
            third_table=third_table,
        )
        self._trie_cache[task_key] = trie
        return trie

    def _prefill_buckets(self, prompt_lens: list[int]) -> dict[int, list[int]]:
        buckets: dict[int, list[int]] = {}
        for sample_idx, prompt_len_raw in enumerate(prompt_lens):
            prompt_len = int(prompt_len_raw)
            chosen_bucket = None
            for candidate in _PREFILL_BUCKETS:
                if int(candidate) >= int(prompt_len):
                    chosen_bucket = int(candidate)
                    break

            if chosen_bucket is None:
                raise ValueError(f"No prefill bucket found for prompt_len={int(prompt_len)}")

            if int(self.max_cache_length) <= int(chosen_bucket) + 2:
                raise ValueError(
                    f"max_cache_length too small for bucket={int(chosen_bucket)}: "
                    f"need > {int(chosen_bucket) + 2}, got {int(self.max_cache_length)}"
                )

            buckets.setdefault(int(chosen_bucket), []).append(int(sample_idx))

        return buckets

    def _get_beam_fn(
        self,
        task_name: str,
        prefill_len: int,
        effective_beams: int,
        trie: SidTrie,
    ):
        key = (str(task_name), int(prefill_len), int(effective_beams))
        cached = self._beam_jit_cache.get(key)
        if cached is not None:
            return cached

        def _beam_search(params: Any, prompt_input_ids: jax.Array, prompt_true_len: jax.Array):
            out: BeamSearchOutput = constrained_beam_search_sid3_prefill(
                model=self.model,
                params=params,
                prompt_input_ids=prompt_input_ids,
                trie=trie,
                num_beams=int(effective_beams),
                max_cache_length=int(self.max_cache_length),
                suffix_token_ids=None,
                prompt_true_len=prompt_true_len,
            )
            return out.token_ids, out.scores

        beam_fn = jax.jit(_beam_search)
        self._beam_jit_cache[key] = beam_fn
        return beam_fn

    def _format_generation_from_triplet(self, triplet_ids: np.ndarray | list[int]) -> str:
        tokens = self.tokenizer.convert_ids_to_tokens([int(x) for x in triplet_ids])
        return "<|sid_begin|>" + "".join(str(t) for t in tokens) + "<|sid_end|>"

    def generate_samples(
        self,
        *,
        task_name: str,
        samples: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, list[str]], dict[str, list[float]], float]:
        if task_name not in SUPPORTED_RECOMMENDATION_TASKS:
            raise ValueError(
                f"JAX generation only supports recommendation tasks {sorted(SUPPORTED_RECOMMENDATION_TASKS)}; "
                f"got task={task_name!r}."
            )

        start_t = time.perf_counter()
        trie = self._build_trie_for_task(task_name)
        effective_beams = int(self.num_beams)
        keep_count = int(self.num_return_sequences)

        sorted_items = sorted(samples.items(), key=lambda x: x[0])
        if not sorted_items:
            return {}, {}, float(time.perf_counter() - start_t)

        sample_ids: list[str] = []
        prompts: list[str] = []
        for sample_id, sample in sorted_items:
            prompt = str(sample.get("prompt", ""))
            if self.prompt_token:
                prompt = f"{prompt}{self.prompt_token}"
            sample_ids.append(str(sample_id))
            prompts.append(prompt)

        tokenized = self.tokenizer(
            prompts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        prompt_ids_raw = tokenized.get("input_ids")
        if not isinstance(prompt_ids_raw, list):
            raise TypeError("Tokenizer output missing input_ids list for batched prompts")

        prompt_ids: list[list[int]] = []
        for ids in prompt_ids_raw:
            int_ids = [int(x) for x in ids]
            if len(int_ids) > self.max_prompt_tokens:
                int_ids = int_ids[-self.max_prompt_tokens :]
            prompt_ids.append(int_ids)
        prompt_lens = [len(ids) for ids in prompt_ids]
        buckets = self._prefill_buckets(prompt_lens)

        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise ValueError("Tokenizer must provide pad_token_id or eos_token_id for batched prompt padding")
        pad_token_id_i = int(pad_token_id)

        generations: dict[str, list[str]] = {}
        logprobs: dict[str, list[float]] = {}

        for prefill_len, sample_indices in sorted(buckets.items()):
            n_bucket = int(len(sample_indices))
            chunk_count = (n_bucket + int(self.batch_size) - 1) // int(self.batch_size)
            print(f"[eval] prefill_len={int(prefill_len)} bucket_samples={n_bucket} chunks={chunk_count}")

            cache_key = (str(task_name), int(prefill_len), int(effective_beams))
            is_first_cache_use = cache_key not in self._beam_jit_cache
            beam_fn = self._get_beam_fn(
                task_name=task_name,
                prefill_len=int(prefill_len),
                effective_beams=int(effective_beams),
                trie=trie,
            )

            compiled_logged = not is_first_cache_use
            for start in range(0, n_bucket, int(self.batch_size)):
                chunk = list(sample_indices[start : start + int(self.batch_size)])
                real_chunk = int(len(chunk))
                if real_chunk < int(self.batch_size):
                    chunk = chunk + [chunk[-1]] * (int(self.batch_size) - real_chunk)

                prompt_np = np.full((int(self.batch_size), int(prefill_len)), int(pad_token_id_i), dtype=np.int32)
                true_len_np = np.zeros((int(self.batch_size),), dtype=np.int32)

                for row, sample_idx in enumerate(chunk):
                    ids = np.asarray(prompt_ids[sample_idx], dtype=np.int32)
                    ids_len = int(ids.shape[0])
                    if ids_len > int(prefill_len):
                        raise ValueError(
                            f"Prompt length exceeds prefill bucket: len={ids_len} prefill_len={int(prefill_len)} "
                            f"sample_id={sample_ids[sample_idx]}"
                        )
                    prompt_np[row, :ids_len] = ids
                    true_len_np[row] = int(ids_len)

                prompt_batch = jnp.asarray(prompt_np, dtype=jnp.int32)
                true_len_batch = jnp.asarray(true_len_np, dtype=jnp.int32)

                if not compiled_logged:
                    t0 = time.perf_counter()
                    token_ids, scores = beam_fn(self.params, prompt_batch, true_len_batch)
                    token_ids_np = np.asarray(token_ids)
                    scores_np = np.asarray(scores, dtype=np.float32)
                    compiled_dt = float(time.perf_counter() - t0)
                    print(f"[eval] prefill_len={int(prefill_len)} compiled_dt={compiled_dt:.2f}s")
                    compiled_logged = True
                else:
                    token_ids, scores = beam_fn(self.params, prompt_batch, true_len_batch)
                    token_ids_np = np.asarray(token_ids)
                    scores_np = np.asarray(scores, dtype=np.float32)

                for row, sample_idx in enumerate(chunk[:real_chunk]):
                    sample_id = sample_ids[sample_idx]
                    sample_generations: list[str] = []
                    sample_logprobs: list[float] = []
                    for beam_idx in range(int(keep_count)):
                        triplet = token_ids_np[row, beam_idx]
                        sample_generations.append(self._format_generation_from_triplet(triplet))
                        sample_logprobs.append(float(scores_np[row, beam_idx]))
                    generations[sample_id] = sample_generations
                    logprobs[sample_id] = sample_logprobs

        total_time = float(time.perf_counter() - start_t)
        return generations, logprobs, total_time


__all__ = ["OpenOneRecJaxGenerator", "SUPPORTED_RECOMMENDATION_TASKS"]
