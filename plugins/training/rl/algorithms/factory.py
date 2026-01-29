from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from plugins.training.rl.algorithms.config import (
    AlgoConfig,
    EstimatorConfig,
    UpdateConfig,
    normalize_algo_name,
    normalize_estimator_name,
    normalize_update_name,
)
from plugins.training.rl.reward.modules import WeightedRewardModule
from plugins.training.rl.update.modules import PolicyGradientUpdateModule, PPOUpdateModule

SUPPORTED_ALGOS = ("reinforce", "ppo", "grpo", "rloo", "dapo", "reinforce++")
SUPPORTED_ESTIMATORS = ("reinforce", "grpo", "rloo", "dapo", "reinforce++", "gae")
SUPPORTED_UPDATES = ("ppo", "policy_gradient")

DEFAULT_ESTIMATOR_FOR_ALGO = {
    "reinforce": "reinforce",
    "ppo": "gae",
    "grpo": "grpo",
    "rloo": "rloo",
    "dapo": "dapo",
    "reinforce++": "reinforce++",
}
DEFAULT_UPDATE_FOR_ALGO = {
    "reinforce": "policy_gradient",
    "ppo": "ppo",
    "grpo": "policy_gradient",
    "rloo": "policy_gradient",
    "dapo": "policy_gradient",
    "reinforce++": "policy_gradient",
}


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


def _resolve_algo_name(cfg: AlgoConfig) -> str:
    name = normalize_algo_name(cfg.name)
    if name not in SUPPORTED_ALGOS:
        raise ValueError(f"Unsupported algo.name={cfg.name!r} (normalized to {name!r}); supported={SUPPORTED_ALGOS}")
    return name


def _resolve_estimator_name(cfg: AlgoConfig, *, algo_name: str) -> str:
    estimator_cfg = cfg.estimator if cfg.estimator is not None else EstimatorConfig()
    raw = normalize_estimator_name(estimator_cfg.name)
    if raw == "auto":
        raw = DEFAULT_ESTIMATOR_FOR_ALGO.get(algo_name, "reinforce")
    if raw not in SUPPORTED_ESTIMATORS:
        raise ValueError(
            f"Unsupported algo.estimator.name={estimator_cfg.name!r} (normalized to {raw!r}); "
            f"supported={SUPPORTED_ESTIMATORS}"
        )
    return raw


def _resolve_update_name(cfg: AlgoConfig, *, algo_name: str) -> str:
    update_cfg = cfg.update if cfg.update is not None else UpdateConfig()
    raw = normalize_update_name(update_cfg.name)
    if raw == "auto":
        raw = DEFAULT_UPDATE_FOR_ALGO.get(algo_name, "policy_gradient")
    if raw not in SUPPORTED_UPDATES:
        raise ValueError(
            f"Unsupported algo.update.name={update_cfg.name!r} (normalized to {raw!r}); supported={SUPPORTED_UPDATES}"
        )
    return raw


def _validate_estimator_config(cfg: EstimatorConfig) -> tuple[float, float, bool, float | None]:
    eps = float(cfg.eps)
    if eps <= 0:
        raise ValueError("algo.estimator.eps must be > 0")

    dapo_alpha = float(cfg.dapo_alpha)
    if dapo_alpha < 0:
        raise ValueError("algo.estimator.dapo_alpha must be >= 0")

    rloo_whiten = bool(cfg.rloo_whiten)

    clip_range = cfg.clip_range
    if clip_range is not None:
        clip_range = float(clip_range)
        if clip_range <= 0:
            raise ValueError("algo.estimator.clip_range must be > 0 when set")
    return eps, dapo_alpha, rloo_whiten, clip_range


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
        ReinforcePlusPlusAdvantageModule,
        RLOOAdvantageModule,
    )

    name = _resolve_algo_name(cfg)
    estimator_name = _resolve_estimator_name(cfg, algo_name=name)
    update_name = _resolve_update_name(cfg, algo_name=name)
    eps, dapo_alpha, rloo_whiten, clip_range = _validate_estimator_config(
        cfg.estimator if cfg.estimator is not None else EstimatorConfig()
    )

    reward_module = WeightedRewardModule(
        reward_funcs=list(reward_funcs),
        reward_weights=tuple(float(x) for x in reward_weights),
    )
    if update_name == "ppo":
        update_module = PPOUpdateModule()
    else:
        update_module = PolicyGradientUpdateModule()

    if estimator_name == "grpo":
        advantage_module = GroupIdGRPOAdvantageModule(eps=eps, clip_range=clip_range)
    elif estimator_name == "reinforce":
        advantage_module = GlobalNormAdvantageModule(eps=eps, clip_range=clip_range)
    elif estimator_name == "rloo":
        advantage_module = RLOOAdvantageModule(eps=eps, whiten=rloo_whiten, clip_range=clip_range)
    elif estimator_name == "dapo":
        advantage_module = DAPOAdvantageModule(eps=eps, alpha=dapo_alpha, clip_range=clip_range)
    elif estimator_name == "reinforce++":
        advantage_module = ReinforcePlusPlusAdvantageModule(eps=eps, clip_range=clip_range)
    elif estimator_name == "gae":
        advantage_module = None
    else:  # pragma: no cover
        raise RuntimeError(f"unreachable estimator name: {estimator_name}")

    if estimator_name == "gae" and update_name != "ppo":
        raise ValueError("algo.estimator.name=gae requires algo.update.name=ppo (value head)")

    requires_value_head = update_name == "ppo" or estimator_name == "gae"

    return Algorithm(
        name=name,
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
