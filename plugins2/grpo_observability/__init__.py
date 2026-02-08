from plugins2.grpo_observability.config import (
    Plugins2ObservabilityConfig,
    Plugins2RolloutConfig,
    Plugins2TrainConfig,
    Plugins2WandbConfig,
    Plugins2WebConfig,
    load_plugins2_config,
)
from plugins2.grpo_observability.types import RunRequest



def __getattr__(name: str):
    if name == "GRPOObservabilityEngine":
        from plugins2.grpo_observability.engine import GRPOObservabilityEngine

        return GRPOObservabilityEngine
    if name == "serve_observability_http":
        from plugins2.grpo_observability.server import serve_observability_http

        return serve_observability_http
    raise AttributeError(name)


__all__ = [
    "GRPOObservabilityEngine",
    "Plugins2ObservabilityConfig",
    "Plugins2RolloutConfig",
    "Plugins2TrainConfig",
    "Plugins2WandbConfig",
    "Plugins2WebConfig",
    "RunRequest",
    "load_plugins2_config",
    "serve_observability_http",
]
