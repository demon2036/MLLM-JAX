"""Sampling workflows (domain semantics on top of a backend).

Example: GRPO-style rollout that returns both decoded answers and a training
batch. Canonical implementations live here; `plugins.sample.sampling` remains
as a compatibility re-export.
"""

from plugins.sample.workflows.grpo_sync import build_chat_prompts, generate_answers_and_training_batch

__all__ = ["build_chat_prompts", "generate_answers_and_training_batch"]

