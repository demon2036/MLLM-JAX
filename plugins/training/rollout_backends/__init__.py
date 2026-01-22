"""Rollout backends (swappable generation engines).

The runner should depend on this layer rather than calling a specific sampler
implementation directly, so the rollout engine can evolve independently.
"""

from plugins.training.rollout_backends.base import RolloutBackend
from plugins.training.rollout_backends.factory import SUPPORTED_ROLLOUT_BACKENDS, create_rollout_backend
from plugins.training.rollout_backends.naive_sampler import NaiveSamplerRolloutBackend
from plugins.training.rollout_backends.sglang_jax_engine import SglangJaxRolloutBackend

__all__ = [
    "NaiveSamplerRolloutBackend",
    "RolloutBackend",
    "SglangJaxRolloutBackend",
    "SUPPORTED_ROLLOUT_BACKENDS",
    "create_rollout_backend",
]
