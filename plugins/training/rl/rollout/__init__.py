"""Rollout phase (prompt -> completion -> trajectory batch).

This package owns rollout-time utilities and swappable generation backends.
"""

from plugins.training.rl.rollout.batching import ceil_div, infer_rollout_passes, round_up_passes_for_divisibility
from plugins.training.rl.rollout.dynamic_sampling import (
    DynamicSamplingConfig,
    DynamicSamplingSummary,
    build_dynamic_sampling_summary,
    homogeneous_group_flags,
    select_completion_indices,
)
from plugins.training.rl.rollout.modules import GRPOSyncRollout, RolloutBackendModule
from plugins.training.rl.rollout.sampling import build_chat_prompts, generate_answers_and_training_batch

__all__ = [
    "GRPOSyncRollout",
    "RolloutBackendModule",
    "DynamicSamplingConfig",
    "DynamicSamplingSummary",
    "build_chat_prompts",
    "build_dynamic_sampling_summary",
    "ceil_div",
    "generate_answers_and_training_batch",
    "homogeneous_group_flags",
    "infer_rollout_passes",
    "round_up_passes_for_divisibility",
    "select_completion_indices",
]
