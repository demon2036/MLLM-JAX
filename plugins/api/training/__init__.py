"""Training contracts (shared across RL/SFT and infra)."""

from plugins.api.training.rl import (
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
from plugins.api.training.schemas.grpo_batch import BatchSchemaError, validate_grpo_batch

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

