from __future__ import annotations

from dataclasses import dataclass, field


def normalize_algo_name(name: str) -> str:
    """Normalize algorithm names to stable identifiers."""
    raw = str(name or "").strip().lower()
    if raw in {"", "default"}:
        return "grpo"
    aliases = {
        "pg": "reinforce",
        "policy_gradient": "reinforce",
        "reinforce++": "reinforce++",
        "reinforcepp": "reinforce++",
        "reinforce_plus_plus": "reinforce++",
        "rloo": "rloo",
        "grpo": "grpo",
        "ppo": "ppo",
        "dapo": "dapo",
        "reinforce": "reinforce",
    }
    return aliases.get(raw, raw)


def normalize_estimator_name(name: str) -> str:
    """Normalize advantage estimator names to stable identifiers."""
    raw = str(name or "").strip().lower()
    if raw in {"", "auto", "default"}:
        return "auto"
    aliases = {
        "pg": "reinforce",
        "policy_gradient": "reinforce",
        "global": "reinforce",
        "global_norm": "reinforce",
        "global_normalized": "reinforce",
        "reinforce++": "reinforce++",
        "reinforcepp": "reinforce++",
        "reinforce_plus_plus": "reinforce++",
        "gae": "gae",
        "ppo": "gae",
        "rloo": "rloo",
        "grpo": "grpo",
        "dapo": "dapo",
        "reinforce": "reinforce",
    }
    return aliases.get(raw, raw)


def normalize_update_name(name: str) -> str:
    """Normalize update method names to stable identifiers."""
    raw = str(name or "").strip().lower()
    if raw in {"", "auto", "default"}:
        return "auto"
    aliases = {
        "pg": "policy_gradient",
        "policy_gradient": "policy_gradient",
        "reinforce": "policy_gradient",
        "grpo": "policy_gradient",
        "ppo": "ppo",
    }
    return aliases.get(raw, raw)


@dataclass(frozen=True)
class EstimatorConfig:
    """Advantage estimator configuration (baseline/normalization/GAE)."""

    # Estimator name (auto chooses a default based on algo.name).
    # One of: auto, reinforce, grpo, rloo, dapo, reinforce++, gae
    name: str = "auto"

    # Numerical stabilizer for normalization denominators.
    eps: float = 1e-4

    # Optional symmetric clipping of the final advantages (applied after normalization).
    clip_range: float | None = None

    # Whether to whiten advantages after computing an RLOO baseline.
    rloo_whiten: bool = True

    # DAPO-style mixing coefficient (used when estimator.name == "dapo").
    dapo_alpha: float = 0.2

    # GAE knobs (used when estimator.name == "gae").
    gae_gamma: float = 1.0
    gae_lambda: float = 0.95
    gae_normalize: bool = True


@dataclass(frozen=True)
class UpdateConfig:
    """Update-loop configuration (PPO/PG update behaviors)."""

    # Update name (auto chooses a default based on algo.name).
    # One of: auto, ppo, policy_gradient
    name: str = "auto"

    # PPO-specific knobs (used when update.name == "ppo").
    # Weight of value (critic) loss in total loss.
    value_coef: float = 0.5
    # Optional clipping range for value loss; None disables clipping.
    value_clip_range: float | None = 0.2
    # Optional entropy coefficient (added to PPO loss if > 0).
    entropy_coef: float = 0.0


@dataclass(frozen=True)
class AlgoConfig:
    """Algorithm selector that composes estimator + update.

    Notes
    -----
    - Most algorithms in this repo are "outcome PG" variants (sequence-level reward),
      implemented by swapping the advantage estimator.
    - `train.ppo_epochs` remains the update-loop knob (multi-epoch PPO/PG update).
    """

    # High-level algorithm alias (used for defaults + logging).
    # One of: reinforce, ppo, grpo, rloo, dapo, reinforce++
    name: str = "grpo"
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    update: UpdateConfig = field(default_factory=UpdateConfig)


__all__ = [
    "AlgoConfig",
    "EstimatorConfig",
    "UpdateConfig",
    "normalize_algo_name",
    "normalize_estimator_name",
    "normalize_update_name",
]
