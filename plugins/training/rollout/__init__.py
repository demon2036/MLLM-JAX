"""Rollout phase (prompt -> completion -> trajectory batch).

This package owns rollout-time utilities and swappable generation backends.
"""

from plugins.training.rollout.batching import ceil_div, infer_rollout_passes, round_up_passes_for_divisibility
from plugins.training.rollout.modules import GRPOSyncRollout, RolloutBackendModule
from plugins.training.rollout.sampling import build_chat_prompts, generate_answers_and_training_batch

__all__ = [
    "GRPOSyncRollout",
    "RolloutBackendModule",
    "build_chat_prompts",
    "ceil_div",
    "generate_answers_and_training_batch",
    "infer_rollout_passes",
    "round_up_passes_for_divisibility",
]
