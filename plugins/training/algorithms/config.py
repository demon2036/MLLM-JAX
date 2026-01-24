from __future__ import annotations

from dataclasses import dataclass


def normalize_algo_name(name: str) -> str:
    """Normalize human-friendly algorithm names to stable identifiers."""
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


@dataclass(frozen=True)
class AlgoConfig:
    """Algorithm selector + knobs.

    Notes
    -----
    - Most algorithms in this repo are "outcome PG" variants (sequence-level reward),
      implemented by swapping the advantage estimator.
    - `ppo_epochs` still lives in `train.*` because it is purely an update-loop knob.
    """

    # One of: reinforce, ppo, grpo, rloo, dapo, reinforce++
    name: str = "grpo"

    # Numerical stabilizer for normalization denominators.
    eps: float = 1e-4

    # DAPO-style mixing coefficient (used when name == "dapo").
    dapo_alpha: float = 0.2

    # Whether to whiten advantages after computing an RLOO baseline.
    # - "rloo": when True, normalize (adv-mean)/std globally.
    # - "reinforce++": always whitens (this flag is still recorded for transparency).
    rloo_whiten: bool = True

    # Optional symmetric clipping of the final advantages (applied after normalization).
    clip_range: float | None = None

    # PPO-specific knobs (used when name == "ppo").
    # Advantage estimator for PPO ("gae" recommended).
    ppo_advantage_estimator: str = "gae"
    # Discount factor for GAE.
    ppo_gamma: float = 1.0
    # GAE lambda.
    ppo_gae_lambda: float = 0.95
    # Weight of value (critic) loss in total loss.
    ppo_value_coef: float = 0.5
    # Optional clipping range for value loss; None disables clipping.
    ppo_value_clip_range: float | None = 0.2
    # Whether to normalize PPO advantages across the batch.
    ppo_advantage_norm: bool = True
    # Optional entropy coefficient (added to PPO loss if > 0).
    ppo_entropy_coef: float = 0.0


__all__ = ["AlgoConfig", "normalize_algo_name"]
