"""Training API layer: contracts and validation helpers."""

from plugins.training.api.batch_schema import BatchSchemaError, validate_grpo_batch
from plugins.training.api.interfaces import (
    AdvantageModule,
    AdvantageResult,
    Batch,
    GRPOWorkflow,
    RewardFunction,
    RewardModule,
    RewardResult,
    RolloutModule,
    RolloutResult,
    RolloutSampler,
    UpdateModule,
    UpdateResult,
)

__all__ = [
    "AdvantageModule",
    "AdvantageResult",
    "Batch",
    "BatchSchemaError",
    "GRPOWorkflow",
    "RewardFunction",
    "RewardModule",
    "RewardResult",
    "RolloutModule",
    "RolloutResult",
    "RolloutSampler",
    "UpdateModule",
    "UpdateResult",
    "validate_grpo_batch",
]
