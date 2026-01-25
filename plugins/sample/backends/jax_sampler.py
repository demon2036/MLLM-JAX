from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from plugins.sample.backends.base import GenerationResult
from plugins.sample.sampling import generate_answers_and_training_batch


@dataclass
class JaxSamplerGenerationBackend:
    """Generation backend backed by the in-process JAX Sampler."""

    sampler: Any

    def generate(
        self,
        *,
        prompts: list[str],
        system_prompt: str,
        global_length: int,
        max_length_sample: int,
        params: Any | None,
    ) -> GenerationResult:
        if params is None:
            raise ValueError("JaxSamplerGenerationBackend requires params for generation.")
        chat_prompts, answers, batch = generate_answers_and_training_batch(
            prompts=list(prompts),
            sampler=self.sampler,
            params=params,
            system_prompt=str(system_prompt),
            global_length=int(global_length),
            max_length_sample=int(max_length_sample),
        )
        return GenerationResult(chat_prompts=chat_prompts, answers=answers, batch=batch)


__all__ = ["JaxSamplerGenerationBackend"]

