from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from plugins.sample.backends.mllm_jax_sampler import get_model
from plugins.training.sft.jax.train import create_mesh_from_config

SUPPORTED_RECOMMENDATION_TASKS = {"video", "ad", "product", "label_cond", "interactive"}


@dataclass(frozen=True)
class _BeamCandidate:
    token_ids: tuple[int, ...]
    score: float
    terminated: bool


class OpenOneRecJaxGenerator:
    """JAX recommendation-task generator with deterministic beam expansion."""

    def __init__(self, *, base_model: str, generation_cfg: Any, jax_cfg: Any):
        self.base_model = str(base_model)
        self.batch_size = int(getattr(generation_cfg, "batch_size", 1))
        self.num_beams = int(getattr(generation_cfg, "num_beams", 4))
        self.num_return_sequences = int(getattr(generation_cfg, "num_return_sequences", 4))
        self.max_new_tokens = int(getattr(generation_cfg, "max_new_tokens", 3))
        self.temperature = float(getattr(generation_cfg, "temperature", 1.0) or 1.0)
        self.top_p = float(getattr(generation_cfg, "top_p", 1.0) or 1.0)
        self.top_k = int(getattr(generation_cfg, "top_k", 50) or 50)
        self.prompt_token = str(getattr(generation_cfg, "prompt_token", "") or "")

        # get_model() reads param dtype from env variable.
        os.environ["MLLM_JAX_PARAM_DTYPE"] = str(getattr(jax_cfg, "param_dtype", "float32"))
        self.mesh = create_mesh_from_config(str(getattr(jax_cfg, "mesh_shape", "1,-1,1")))
        self.model, self.params, self.tokenizer = get_model(mesh=self.mesh, model_path=self.base_model)

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        self.eos_token_id = int(eos_id) if eos_id is not None else -1

    def generate_samples(self, *, task_name: str, samples: dict[str, dict[str, Any]]) -> tuple[dict[str, list[str]], dict[str, list[float]], float]:
        if task_name not in SUPPORTED_RECOMMENDATION_TASKS:
            raise ValueError(
                f"JAX generation only supports recommendation tasks {sorted(SUPPORTED_RECOMMENDATION_TASKS)}; "
                f"got task={task_name!r}."
            )

        start_t = time.perf_counter()
        generations: dict[str, list[str]] = {}
        logprobs: dict[str, list[float]] = {}

        # Keep implementation simple and deterministic for eval-first parity checks.
        sorted_items = sorted(samples.items(), key=lambda x: x[0])
        for sample_id, sample in sorted_items:
            prompt = str(sample.get("prompt", ""))
            if self.prompt_token:
                prompt = f"{prompt}{self.prompt_token}"

            seqs, seq_scores = self._generate_single_prompt(prompt)
            generations[sample_id] = seqs
            logprobs[sample_id] = seq_scores

        total_time = float(time.perf_counter() - start_t)
        return generations, logprobs, total_time

    def _generate_single_prompt(self, prompt: str) -> tuple[list[str], list[float]]:
        prompt_ids = [int(x) for x in self.tokenizer.encode(prompt, add_special_tokens=False)]
        if not prompt_ids:
            bos_id = getattr(self.tokenizer, "bos_token_id", None)
            if bos_id is not None:
                prompt_ids = [int(bos_id)]
            else:
                prompt_ids = [0]

        beam_count = max(1, self.num_beams)
        keep_count = max(1, min(self.num_return_sequences, beam_count))
        active: list[_BeamCandidate] = [_BeamCandidate(token_ids=tuple(prompt_ids), score=0.0, terminated=False)]

        for _ in range(max(1, self.max_new_tokens)):
            expanded: list[_BeamCandidate] = []
            for beam in active:
                if beam.terminated:
                    expanded.append(beam)
                    continue

                logits = self._forward_last_logits(list(beam.token_ids))
                temp = max(self.temperature, 1e-5)
                step_logprobs = jax.nn.log_softmax(jnp.asarray(logits, dtype=jnp.float32) / float(temp))

                topk = max(1, min(int(self.top_k), int(step_logprobs.shape[-1])))
                top_values, top_indices = jax.lax.top_k(step_logprobs, topk)
                top_values_np = np.asarray(top_values, dtype=np.float64)
                top_indices_np = np.asarray(top_indices, dtype=np.int32)

                values_np, indices_np = self._apply_top_p(top_values_np, top_indices_np)
                for lp, token_id in zip(values_np.tolist(), indices_np.tolist(), strict=True):
                    token_id_i = int(token_id)
                    token_ids = beam.token_ids + (token_id_i,)
                    terminated = self.eos_token_id >= 0 and token_id_i == self.eos_token_id
                    expanded.append(
                        _BeamCandidate(
                            token_ids=token_ids,
                            score=float(beam.score + float(lp)),
                            terminated=bool(terminated),
                        )
                    )

            if not expanded:
                break

            expanded.sort(key=lambda x: (-x.score, x.token_ids))
            active = expanded[:beam_count]
            if all(x.terminated for x in active):
                break

        active.sort(key=lambda x: (-x.score, x.token_ids))
        selected = active[:keep_count]

        prompt_len = len(prompt_ids)
        generations: list[str] = []
        logprobs: list[float] = []
        for candidate in selected:
            generated_ids = list(candidate.token_ids[prompt_len:])
            if self.eos_token_id >= 0 and self.eos_token_id in generated_ids:
                generated_ids = generated_ids[: generated_ids.index(self.eos_token_id)]

            generations.append(self.tokenizer.decode(generated_ids, skip_special_tokens=False).strip())
            logprobs.append(float(candidate.score))

        return generations, logprobs

    def _apply_top_p(self, top_values: np.ndarray, top_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.top_p >= 1.0:
            return top_values, top_indices

        if len(top_values) == 0:
            return top_values, top_indices

        probs = np.exp(top_values - float(np.max(top_values)))
        probs = probs / max(float(np.sum(probs)), 1e-12)
        cumulative = np.cumsum(probs)
        keep = cumulative <= float(max(self.top_p, 0.0))
        if not np.any(keep):
            keep[0] = True
        return top_values[keep], top_indices[keep]

    def _forward_last_logits(self, token_ids: list[int]) -> np.ndarray:
        input_ids = jnp.asarray(np.asarray([token_ids], dtype=np.int32), dtype=jnp.int32)
        attention_mask = jnp.ones_like(input_ids, dtype=jnp.int32)
        position_ids = jnp.arange(input_ids.shape[1], dtype=jnp.int32)[None, :]

        out = self.model.apply(
            {"params": self.params},
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=None,
        )

        logits = out[0] if isinstance(out, tuple) else out
        return np.asarray(logits)[0, -1, :]


__all__ = ["OpenOneRecJaxGenerator", "SUPPORTED_RECOMMENDATION_TASKS"]
