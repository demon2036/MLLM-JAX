"""Algorithm composition layer (registry/factory).

This package wires the four-phase runner to concrete algorithm variants by
choosing implementations for:
- reward module
- advantage module
- update module

The current codebase shares one rollout workflow and one PPO-style update
implementation, so most "algorithm" variation is expressed via the advantage
estimator (baseline/normalization/mixing). PPO additionally uses a value-head
and GAE path for actor-critic updates.
"""

from plugins.training.rl.algorithms.config import (
    AlgoConfig,
    DEFAULT_ESTIMATOR_KWARGS,
    DEFAULT_UPDATE_KWARGS,
    PluginConfig,
    normalize_algo_config,
    normalize_algo_name,
    normalize_estimator_name,
    normalize_update_name,
    sanitize_algo_config_for_logging,
)
from plugins.training.rl.algorithms.factory import (
    Algorithm,
    SUPPORTED_ALGOS,
    SUPPORTED_ESTIMATORS,
    SUPPORTED_UPDATES,
    create_algorithm,
)

__all__ = [
    "Algorithm",
    "AlgoConfig",
    "DEFAULT_ESTIMATOR_KWARGS",
    "DEFAULT_UPDATE_KWARGS",
    "PluginConfig",
    "SUPPORTED_ALGOS",
    "SUPPORTED_ESTIMATORS",
    "SUPPORTED_UPDATES",
    "create_algorithm",
    "normalize_algo_config",
    "normalize_algo_name",
    "normalize_estimator_name",
    "normalize_update_name",
    "sanitize_algo_config_for_logging",
]
