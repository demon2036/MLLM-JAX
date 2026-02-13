from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from plugins.training.rl.algorithms.config import AlgoConfig, normalize_algo_config
from plugins.training.rl.reward.modules import WeightedRewardModule
from plugins.training.rl.update.modules import PolicyGradientUpdateModule, PPOUpdateModule

SUPPORTED_ALGOS = ("reinforce", "ppo", "grpo", "rloo", "dapo", "reinforce++", "maxrl", "remax")
SUPPORTED_ESTIMATORS = ("reinforce", "grpo", "rloo", "dapo", "reinforce++", "maxrl", "remax", "gae")
SUPPORTED_UPDATES = ("ppo", "policy_gradient", "remax")


@dataclass(frozen=True)
class Algorithm:
    """Concrete algorithm wiring for the 4-phase runner."""

    name: str
    reward_module: Any
    advantage_module: Any | None
    update_module: Any
    estimator_name: str
    update_name: str
    requires_value_head: bool


def create_algorithm(
    cfg: AlgoConfig,
    *,
    reward_funcs: Sequence[Callable[..., float]],
    reward_weights: Sequence[float],
) -> Algorithm:
    """Create algorithm wiring from config.

    This keeps the runner free of per-algo if/else branches.
    """

    from plugins.training.rl.advantage.modules import (
        DAPOAdvantageModule,
        GlobalNormAdvantageModule,
        GroupIdGRPOAdvantageModule,
        MaxRLAdvantageModule,
        ReMaxGreedyBaselineAdvantageModule,
        ReinforcePlusPlusAdvantageModule,
        RLOOAdvantageModule,
    )

    normalized, algo_name, estimator_name, update_name = normalize_algo_config(cfg)

    reward_module = WeightedRewardModule(
        reward_funcs=list(reward_funcs),
        reward_weights=tuple(float(x) for x in reward_weights),
    )

    if update_name == "ppo":
        update_module = PPOUpdateModule()
    else:
        update_module = PolicyGradientUpdateModule()

    estimator_kwargs = dict(normalized.estimator.kwargs)
    eps = float(estimator_kwargs.get("eps", 1e-4))
    clip_range_raw = estimator_kwargs.get("clip_range")
    clip_range = None if clip_range_raw is None else float(clip_range_raw)

    if estimator_name == "grpo":
        pos_adv_scale = float(estimator_kwargs.get("pos_adv_scale", 1.0))
        advantage_module = GroupIdGRPOAdvantageModule(eps=eps, clip_range=clip_range, pos_adv_scale=pos_adv_scale)
    elif estimator_name == "reinforce":
        advantage_module = GlobalNormAdvantageModule(eps=eps, clip_range=clip_range)
    elif estimator_name == "rloo":
        whiten = bool(estimator_kwargs.get("whiten", True))
        advantage_module = RLOOAdvantageModule(eps=eps, whiten=whiten, clip_range=clip_range)
    elif estimator_name == "dapo":
        alpha = float(estimator_kwargs.get("alpha", 0.2))
        advantage_module = DAPOAdvantageModule(eps=eps, alpha=alpha, clip_range=clip_range)
    elif estimator_name == "reinforce++":
        advantage_module = ReinforcePlusPlusAdvantageModule(eps=eps, clip_range=clip_range)
    elif estimator_name == "maxrl":
        advantage_module = MaxRLAdvantageModule(eps=eps, clip_range=clip_range)
    elif estimator_name == "remax":
        baseline_position = int(estimator_kwargs.get("baseline_position", 0))
        advantage_module = ReMaxGreedyBaselineAdvantageModule(
            baseline_position=baseline_position,
            clip_range=clip_range,
        )
    elif estimator_name == "gae":
        advantage_module = None
    else:  # pragma: no cover
        raise RuntimeError(f"unreachable estimator name: {estimator_name}")

    requires_value_head = update_name == "ppo" or estimator_name == "gae"

    return Algorithm(
        name=algo_name,
        reward_module=reward_module,
        advantage_module=advantage_module,
        update_module=update_module,
        estimator_name=estimator_name,
        update_name=update_name,
        requires_value_head=requires_value_head,
    )


__all__ = [
    "Algorithm",
    "SUPPORTED_ALGOS",
    "SUPPORTED_ESTIMATORS",
    "SUPPORTED_UPDATES",
    "create_algorithm",
]
