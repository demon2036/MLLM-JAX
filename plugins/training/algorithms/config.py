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


__all__ = ["AlgoConfig", "normalize_algo_name"]

