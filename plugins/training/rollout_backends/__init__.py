"""Rollout backends (swappable generation engines).

The runner should depend on this layer rather than calling a specific sampler
implementation directly. This makes it possible to later plug in backends like
`sglang-jax` (à la Tunix) without rewriting the GRPO training loop.
"""

from plugins.training.rollout_backends.base import RolloutBackend
from plugins.training.rollout_backends.factory import SUPPORTED_ROLLOUT_BACKENDS, create_rollout_backend
from plugins.training.rollout_backends.naive_sampler import NaiveSamplerRolloutBackend

__all__ = [
    "NaiveSamplerRolloutBackend",
    "RolloutBackend",
    "SUPPORTED_ROLLOUT_BACKENDS",
    "create_rollout_backend",
]

