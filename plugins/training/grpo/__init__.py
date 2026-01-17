"""GRPO training building blocks (plugins-first).

This package hosts the non-invasive, reusable implementation of the GRPO-style
training phases used in this repo:
- rollout/sampling
- reward composition
- advantage computation
- PPO/GRPO update loop
"""

from plugins.training.grpo.advantages import compute_grpo_advantages_by_group_id
from plugins.training.grpo.rewarding import compute_weighted_rewards
from plugins.training.grpo.sampling import build_chat_prompts, generate_answers_and_training_batch
from plugins.training.grpo.update import ppo_update

__all__ = [
    "build_chat_prompts",
    "compute_grpo_advantages_by_group_id",
    "compute_weighted_rewards",
    "generate_answers_and_training_batch",
    "ppo_update",
]

