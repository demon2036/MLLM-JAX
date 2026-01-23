"""Algorithm composition layer (registry/factory).

This package wires the four-phase runner to concrete algorithm variants by
choosing implementations for:
- reward module
- advantage module
- update module

The current codebase shares one rollout workflow and one PPO-style update
implementation, so most "algorithm" variation is expressed via the advantage
estimator (baseline/normalization/mixing).
"""

from plugins.training.algorithms.config import AlgoConfig, normalize_algo_name
from plugins.training.algorithms.factory import Algorithm, SUPPORTED_ALGOS, create_algorithm

__all__ = [
    "Algorithm",
    "AlgoConfig",
    "SUPPORTED_ALGOS",
    "create_algorithm",
    "normalize_algo_name",
]

