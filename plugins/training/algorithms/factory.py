from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from plugins.training.algorithms.config import AlgoConfig, normalize_algo_name
from plugins.training.reward.modules import WeightedRewardModule
from plugins.training.update.modules import PPOUpdateModule

SUPPORTED_ALGOS = ("reinforce", "ppo", "grpo", "rloo", "dapo", "reinforce++")


@dataclass(frozen=True)
class Algorithm:
    """Concrete algorithm wiring for the 4-phase runner."""

    name: str
    reward_module: Any
    advantage_module: Any
    update_module: Any


def _validate_algo_config(cfg: AlgoConfig) -> tuple[str, float, float, bool, float | None]:
    name = normalize_algo_name(cfg.name)
    if name not in SUPPORTED_ALGOS:
        raise ValueError(f"Unsupported algo.name={cfg.name!r} (normalized to {name!r}); supported={SUPPORTED_ALGOS}")

    eps = float(cfg.eps)
    if eps <= 0:
        raise ValueError("algo.eps must be > 0")

    dapo_alpha = float(cfg.dapo_alpha)
    rloo_whiten = bool(cfg.rloo_whiten)

    clip_range = cfg.clip_range
    if clip_range is not None:
        clip_range = float(clip_range)
        if clip_range <= 0:
            raise ValueError("algo.clip_range must be > 0 when set")
    return name, eps, dapo_alpha, rloo_whiten, clip_range


def create_algorithm(
    cfg: AlgoConfig,
    *,
    reward_funcs: Sequence[Callable[..., float]],
    reward_weights: Sequence[float],
) -> Algorithm:
    """Create algorithm wiring from config.

    This keeps the runner free of per-algo if/else branches.
    """

    from plugins.training.advantage.modules import (
        DAPOAdvantageModule,
        GlobalNormAdvantageModule,
        GroupIdGRPOAdvantageModule,
        ReinforcePlusPlusAdvantageModule,
        RLOOAdvantageModule,
    )

    name, eps, dapo_alpha, rloo_whiten, clip_range = _validate_algo_config(cfg)

    reward_module = WeightedRewardModule(reward_funcs=list(reward_funcs), reward_weights=tuple(float(x) for x in reward_weights))
    update_module = PPOUpdateModule()

    if name == "grpo":
        advantage_module = GroupIdGRPOAdvantageModule(eps=eps, clip_range=clip_range)
    elif name in {"reinforce", "ppo"}:
        advantage_module = GlobalNormAdvantageModule(eps=eps, clip_range=clip_range)
    elif name == "rloo":
        advantage_module = RLOOAdvantageModule(eps=eps, whiten=rloo_whiten, clip_range=clip_range)
    elif name == "dapo":
        advantage_module = DAPOAdvantageModule(eps=eps, alpha=dapo_alpha, clip_range=clip_range)
    elif name == "reinforce++":
        advantage_module = ReinforcePlusPlusAdvantageModule(eps=eps, clip_range=clip_range)
    else:  # pragma: no cover
        raise RuntimeError(f"unreachable algo name: {name}")

    return Algorithm(
        name=name,
        reward_module=reward_module,
        advantage_module=advantage_module,
        update_module=update_module,
    )


__all__ = ["Algorithm", "SUPPORTED_ALGOS", "create_algorithm"]

