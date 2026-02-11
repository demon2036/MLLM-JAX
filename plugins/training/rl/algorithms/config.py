from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


DEFAULT_ESTIMATOR_KWARGS: dict[str, dict[str, Any]] = {
    "reinforce": {"eps": 1e-4, "clip_range": None},
    "grpo": {"eps": 1e-4, "clip_range": None},
    "rloo": {"eps": 1e-4, "clip_range": None, "whiten": True},
    "dapo": {"eps": 1e-4, "clip_range": None, "alpha": 0.2},
    "reinforce++": {"eps": 1e-4, "clip_range": None},
    "maxrl": {"eps": 1e-6, "clip_range": None},
    "gae": {"eps": 1e-4, "clip_range": None, "gamma": 1.0, "gae_lambda": 0.95, "normalize": True},
}

DEFAULT_UPDATE_KWARGS: dict[str, dict[str, Any]] = {
    "policy_gradient": {"token_focus": None},
    "ppo": {
        "value_coef": 0.5,
        "value_clip_range": 0.2,
        "entropy_coef": 0.0,
        "token_focus": None,
    },
}


@dataclass(frozen=True)
class PluginConfig:
    """Generic plugin config with strict `name + kwargs` shape."""

    name: str = "auto"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlgoConfig:
    """Algorithm selector composed by estimator/update plugins."""

    name: str = "grpo"
    estimator: PluginConfig = field(default_factory=PluginConfig)
    update: PluginConfig = field(default_factory=PluginConfig)


def _as_dict(value: Any, *, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping, got {type(value).__name__}")
    return dict(value)


def _as_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{label} must be a boolean, got {value!r}")


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
        "maxrl": "maxrl",
        "max-rl": "maxrl",
        "max_rl": "maxrl",
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
        "maxrl": "maxrl",
        "max-rl": "maxrl",
        "max_rl": "maxrl",
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


def _normalize_estimator_kwargs(estimator_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    defaults = dict(DEFAULT_ESTIMATOR_KWARGS[estimator_name])
    unknown = sorted(set(kwargs.keys()) - set(defaults.keys()))
    if unknown:
        raise ValueError(
            "Unsupported algo.estimator.kwargs keys "
            + f"for estimator {estimator_name!r}: {unknown}; allowed={sorted(defaults.keys())}"
        )

    merged = {**defaults, **kwargs}
    normalized: dict[str, Any] = {
        "eps": float(merged["eps"]),
        "clip_range": None if merged["clip_range"] is None else float(merged["clip_range"]),
    }
    if normalized["eps"] <= 0:
        raise ValueError("algo.estimator.kwargs.eps must be > 0")
    if normalized["clip_range"] is not None and float(normalized["clip_range"]) <= 0:
        raise ValueError("algo.estimator.kwargs.clip_range must be > 0 when set")

    if estimator_name == "rloo":
        normalized["whiten"] = _as_bool(merged["whiten"], label="algo.estimator.kwargs.whiten")
    elif estimator_name == "dapo":
        alpha = float(merged["alpha"])
        if alpha < 0:
            raise ValueError("algo.estimator.kwargs.alpha must be >= 0")
        normalized["alpha"] = alpha
    elif estimator_name == "gae":
        gamma = float(merged["gamma"])
        gae_lambda = float(merged["gae_lambda"])
        normalized["gamma"] = gamma
        normalized["gae_lambda"] = gae_lambda
        normalized["normalize"] = _as_bool(merged["normalize"], label="algo.estimator.kwargs.normalize")

    return normalized


def _normalize_update_kwargs(update_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    defaults = dict(DEFAULT_UPDATE_KWARGS[update_name])
    unknown = sorted(set(kwargs.keys()) - set(defaults.keys()))
    if unknown:
        raise ValueError(
            "Unsupported algo.update.kwargs keys "
            + f"for update {update_name!r}: {unknown}; allowed={sorted(defaults.keys())}"
        )

    merged = {**defaults, **kwargs}
    token_focus_raw = merged.get("token_focus")
    token_focus: dict[str, Any] | None
    if token_focus_raw is None:
        token_focus = None
    elif isinstance(token_focus_raw, bool):
        token_focus = {
            "enabled": bool(token_focus_raw),
            "prob_threshold": 0.3,
            "max_tokens_per_sequence": 10,
            "use_old_logps": True,
        }
    elif isinstance(token_focus_raw, dict):
        allowed = {"enabled", "prob_threshold", "max_tokens_per_sequence", "use_old_logps"}
        unknown_tf = sorted(set(token_focus_raw.keys()) - allowed)
        if unknown_tf:
            raise ValueError(
                "Unsupported algo.update.kwargs.token_focus keys: "
                f"{unknown_tf}; allowed={sorted(allowed)}"
            )
        enabled = _as_bool(token_focus_raw.get("enabled", False), label="algo.update.kwargs.token_focus.enabled")
        prob_threshold = float(token_focus_raw.get("prob_threshold", 0.3))
        if not (0.0 < prob_threshold < 1.0):
            raise ValueError("algo.update.kwargs.token_focus.prob_threshold must be in (0, 1)")
        max_tokens = int(token_focus_raw.get("max_tokens_per_sequence", 10))
        if max_tokens < 1:
            raise ValueError("algo.update.kwargs.token_focus.max_tokens_per_sequence must be >= 1")
        use_old_logps = _as_bool(
            token_focus_raw.get("use_old_logps", True),
            label="algo.update.kwargs.token_focus.use_old_logps",
        )
        token_focus = {
            "enabled": bool(enabled),
            "prob_threshold": float(prob_threshold),
            "max_tokens_per_sequence": int(max_tokens),
            "use_old_logps": bool(use_old_logps),
        }
    else:
        raise ValueError("algo.update.kwargs.token_focus must be a dict or boolean when provided")

    if update_name == "policy_gradient":
        return {} if token_focus is None else {"token_focus": token_focus}

    normalized: dict[str, Any] = {
        "value_coef": float(merged["value_coef"]),
        "value_clip_range": None if merged["value_clip_range"] is None else float(merged["value_clip_range"]),
        "entropy_coef": float(merged["entropy_coef"]),
    }
    if token_focus is not None:
        normalized["token_focus"] = token_focus
    return normalized


def normalize_algo_config(cfg: AlgoConfig) -> tuple[AlgoConfig, str, str, str]:
    """Return normalized AlgoConfig plus resolved names.

    - enforce strict `name + kwargs` schema
    - fill defaults only for active plugins
    - validate cross-field constraints
    """

    algo_name = normalize_algo_name(cfg.name)
    supported_algos = {"reinforce", "ppo", "grpo", "rloo", "dapo", "reinforce++", "maxrl"}
    if algo_name not in supported_algos:
        raise ValueError(f"Unsupported algo.name={cfg.name!r} (normalized to {algo_name!r}); supported={sorted(supported_algos)}")

    default_estimator_for_algo = {
        "reinforce": "reinforce",
        "ppo": "gae",
        "grpo": "grpo",
        "rloo": "rloo",
        "dapo": "dapo",
        "reinforce++": "reinforce++",
        "maxrl": "maxrl",
    }
    default_update_for_algo = {
        "reinforce": "policy_gradient",
        "ppo": "ppo",
        "grpo": "policy_gradient",
        "rloo": "policy_gradient",
        "dapo": "policy_gradient",
        "reinforce++": "policy_gradient",
        "maxrl": "policy_gradient",
    }

    estimator_name = normalize_estimator_name(cfg.estimator.name)
    if estimator_name == "auto":
        estimator_name = default_estimator_for_algo.get(algo_name, "reinforce")

    update_name = normalize_update_name(cfg.update.name)
    if update_name == "auto":
        update_name = default_update_for_algo.get(algo_name, "policy_gradient")

    supported_estimators = set(DEFAULT_ESTIMATOR_KWARGS.keys())
    if estimator_name not in supported_estimators:
        raise ValueError(f"Unsupported algo.estimator.name={cfg.estimator.name!r} (normalized to {estimator_name!r})")

    supported_updates = set(DEFAULT_UPDATE_KWARGS.keys())
    if update_name not in supported_updates:
        raise ValueError(f"Unsupported algo.update.name={cfg.update.name!r} (normalized to {update_name!r})")

    estimator_kwargs = _normalize_estimator_kwargs(estimator_name, _as_dict(cfg.estimator.kwargs, label="algo.estimator.kwargs"))
    update_kwargs = _normalize_update_kwargs(update_name, _as_dict(cfg.update.kwargs, label="algo.update.kwargs"))

    if estimator_name == "gae" and update_name != "ppo":
        raise ValueError("algo.estimator.name=gae requires algo.update.name=ppo")

    normalized = AlgoConfig(
        name=algo_name,
        estimator=PluginConfig(name=estimator_name, kwargs=estimator_kwargs),
        update=PluginConfig(name=update_name, kwargs=update_kwargs),
    )
    return normalized, algo_name, estimator_name, update_name


def sanitize_algo_config_for_logging(cfg: AlgoConfig) -> dict[str, object]:
    """Return logging-friendly dict containing only active estimator/update plugins."""

    normalized, algo_name, estimator_name, update_name = normalize_algo_config(cfg)
    return {
        "name": algo_name,
        "estimator": {
            "name": estimator_name,
            "kwargs": dict(normalized.estimator.kwargs),
        },
        "update": {
            "name": update_name,
            "kwargs": dict(normalized.update.kwargs),
        },
    }


__all__ = [
    "AlgoConfig",
    "DEFAULT_ESTIMATOR_KWARGS",
    "DEFAULT_UPDATE_KWARGS",
    "PluginConfig",
    "normalize_algo_config",
    "normalize_algo_name",
    "normalize_estimator_name",
    "normalize_update_name",
    "sanitize_algo_config_for_logging",
]
