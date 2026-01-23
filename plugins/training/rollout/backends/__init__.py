"""Rollout backends (swappable generation engines).

The runner should depend on this layer rather than calling a specific sampler
implementation directly, so the rollout engine can evolve independently.
"""

from plugins.training.rollout.backends.base import RolloutBackend
from plugins.training.rollout.backends.factory import SUPPORTED_ROLLOUT_BACKENDS, create_rollout_backend
from plugins.training.rollout.backends.naive_sampler import NaiveSamplerRolloutBackend

__all__ = [
    "NaiveSamplerRolloutBackend",
    "RolloutBackend",
    "SUPPORTED_ROLLOUT_BACKENDS",
    "create_rollout_backend",
]
