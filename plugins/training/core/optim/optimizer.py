from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


def _default_lr_schedule_kwargs() -> dict[str, Any]:
    return {
        "init_value": 0.0,
        "peak_value": 1e-6,
        "end_value": 0.0,
        "warmup_ratio": 0.05,
        "warmup_steps": None,
    }


def _default_optimizer_kwargs() -> dict[str, Any]:
    return {
        "clip_norm": 1.0,
        "weight_decay": 1e-8,
    }


def _default_muon_kwargs() -> dict[str, Any]:
    return {
        "aux_lr": 3e-4,
        "momentum": 0.95,
        "nesterov": True,
        "ns_steps": 5,
        "eps": 1e-7,
        "max_dim": 10_000,
    }


@dataclass(frozen=True)
class LRScheduleConfig:
    """Learning-rate schedule plugin config.

    Schema:
    - name: schedule implementation name
    - kwargs: schedule parameters
    """

    name: str = "warmup_cosine"
    kwargs: dict[str, Any] = field(default_factory=_default_lr_schedule_kwargs)


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer plugin config.

    Schema:
    - name: optimizer implementation name
    - kwargs: optimizer parameters
    - lr_schedule: lr-schedule plugin config
    """

    name: str = "lion"
    kwargs: dict[str, Any] = field(default_factory=_default_optimizer_kwargs)
    lr_schedule: LRScheduleConfig = field(default_factory=LRScheduleConfig)


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


def normalize_lr_schedule_name(name: str) -> str:
    raw = str(name or "").strip().lower()
    if raw in {"", "default", "auto"}:
        return "warmup_cosine"
    aliases = {
        "warmup_cosine": "warmup_cosine",
        "warmup_cosine_decay": "warmup_cosine",
        "warmup_linear": "warmup_linear",
        "warmup_linear_decay": "warmup_linear",
        "linear_warmup": "warmup_linear",
        "constant": "constant",
        "const": "constant",
    }
    return aliases.get(raw, raw)


def normalize_optimizer_name(name: str) -> str:
    raw = str(name or "").strip().lower()
    if raw in {"", "default", "auto"}:
        return "lion"
    aliases = {
        "lion": "lion",
        "adam": "adamw",
        "adamw": "adamw",
        "sgd": "sgd",
        "muon": "muon",
    }
    return aliases.get(raw, raw)


def _as_dict(value: Any, *, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping, got {type(value).__name__}")
    return dict(value)


def normalize_lr_schedule_config(cfg: LRScheduleConfig) -> LRScheduleConfig:
    schedule_name = normalize_lr_schedule_name(cfg.name)
    supported = {"warmup_cosine", "warmup_linear", "constant"}
    if schedule_name not in supported:
        raise ValueError(
            f"Unsupported lr_schedule.name={cfg.name!r} (normalized to {schedule_name!r}); supported={sorted(supported)}"
        )

    raw_kwargs = _as_dict(cfg.kwargs, label="optimizer.lr_schedule.kwargs")
    defaults = _default_lr_schedule_kwargs()
    unknown = sorted(set(raw_kwargs.keys()) - set(defaults.keys()))
    if unknown:
        raise ValueError(
            "Unsupported optimizer.lr_schedule.kwargs keys "
            + f"for schedule {schedule_name!r}: {unknown}; allowed={sorted(defaults.keys())}"
        )

    merged = {**defaults, **raw_kwargs}
    warmup_steps_raw = merged.get("warmup_steps")
    warmup_steps = None if warmup_steps_raw is None else int(warmup_steps_raw)
    if warmup_steps is not None and warmup_steps < 0:
        raise ValueError(f"optimizer.lr_schedule.kwargs.warmup_steps must be >= 0, got {warmup_steps}")

    warmup_ratio = float(merged["warmup_ratio"])
    if warmup_ratio < 0:
        raise ValueError(f"optimizer.lr_schedule.kwargs.warmup_ratio must be >= 0, got {warmup_ratio}")

    normalized_kwargs: dict[str, Any] = {
        "init_value": float(merged["init_value"]),
        "peak_value": float(merged["peak_value"]),
        "end_value": float(merged["end_value"]),
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps,
    }
    return LRScheduleConfig(name=schedule_name, kwargs=normalized_kwargs)


def normalize_optimizer_config(cfg: OptimizerConfig) -> OptimizerConfig:
    optimizer_name = normalize_optimizer_name(cfg.name)
    supported = {"lion", "adamw", "sgd", "muon"}
    if optimizer_name not in supported:
        raise ValueError(
            f"Unsupported optimizer.name={cfg.name!r} (normalized to {optimizer_name!r}); supported={sorted(supported)}"
        )

    raw_kwargs = _as_dict(cfg.kwargs, label="optimizer.kwargs")
    defaults = _default_optimizer_kwargs()
    if optimizer_name == "muon":
        defaults = {**defaults, **_default_muon_kwargs()}

    unknown = sorted(set(raw_kwargs.keys()) - set(defaults.keys()))
    if unknown:
        raise ValueError(
            "Unsupported optimizer.kwargs keys "
            + f"for optimizer {optimizer_name!r}: {unknown}; allowed={sorted(defaults.keys())}"
        )

    merged = {**defaults, **raw_kwargs}
    clip_norm = float(merged["clip_norm"])
    if clip_norm < 0:
        raise ValueError(f"optimizer.kwargs.clip_norm must be >= 0, got {clip_norm}")

    weight_decay = float(merged["weight_decay"])
    if weight_decay < 0:
        raise ValueError(f"optimizer.kwargs.weight_decay must be >= 0, got {weight_decay}")

    normalized_kwargs: dict[str, Any] = {
        "clip_norm": clip_norm,
        "weight_decay": weight_decay,
    }

    if optimizer_name == "muon":
        aux_lr = float(merged["aux_lr"])
        if aux_lr <= 0:
            raise ValueError(f"optimizer.kwargs.aux_lr must be > 0, got {aux_lr}")

        momentum = float(merged["momentum"])
        ns_steps = int(merged["ns_steps"])
        eps = float(merged["eps"])
        max_dim = int(merged["max_dim"])
        if ns_steps <= 0:
            raise ValueError(f"optimizer.kwargs.ns_steps must be > 0, got {ns_steps}")
        if eps <= 0:
            raise ValueError(f"optimizer.kwargs.eps must be > 0, got {eps}")
        if max_dim <= 0:
            raise ValueError(f"optimizer.kwargs.max_dim must be > 0, got {max_dim}")

        normalized_kwargs.update(
            {
                "aux_lr": aux_lr,
                "momentum": momentum,
                "nesterov": _as_bool(merged["nesterov"], label="optimizer.kwargs.nesterov"),
                "ns_steps": ns_steps,
                "eps": eps,
                "max_dim": max_dim,
            }
        )

    normalized_lr = normalize_lr_schedule_config(cfg.lr_schedule)
    return OptimizerConfig(name=optimizer_name, kwargs=normalized_kwargs, lr_schedule=normalized_lr)


def sanitize_optimizer_config_for_logging(cfg: OptimizerConfig) -> dict[str, object]:
    """Return a logging-friendly optimizer config containing only active keys."""

    normalized = normalize_optimizer_config(cfg)
    return {
        "name": str(normalized.name),
        "kwargs": dict(normalized.kwargs),
        "lr_schedule": {
            "name": str(normalized.lr_schedule.name),
            "kwargs": dict(normalized.lr_schedule.kwargs),
        },
    }


def build_lr_schedule(*, training_steps: int, cfg: LRScheduleConfig):
    import optax

    steps = int(training_steps)
    if steps <= 0:
        raise ValueError(f"training_steps must be > 0, got {steps}")

    schedule_cfg = normalize_lr_schedule_config(cfg)
    schedule_name = schedule_cfg.name
    kwargs = schedule_cfg.kwargs

    if schedule_name == "warmup_cosine":
        warmup_steps = kwargs["warmup_steps"]
        if warmup_steps is None:
            warmup_steps = int(round(float(steps) * float(kwargs["warmup_ratio"])))
        warmup_steps = int(warmup_steps)
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        return optax.warmup_cosine_decay_schedule(
            init_value=float(kwargs["init_value"]),
            peak_value=float(kwargs["peak_value"]),
            warmup_steps=warmup_steps,
            decay_steps=steps,
            end_value=float(kwargs["end_value"]),
        )

    if schedule_name == "warmup_linear":
        warmup_steps = kwargs["warmup_steps"]
        if warmup_steps is None:
            warmup_steps = int(round(float(steps) * float(kwargs["warmup_ratio"])))
        warmup_steps = int(warmup_steps)
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        warmup_steps = min(warmup_steps, steps)
        if warmup_steps == 0:
            return optax.linear_schedule(
                init_value=float(kwargs["peak_value"]),
                end_value=float(kwargs["end_value"]),
                transition_steps=steps,
            )

        warmup = optax.linear_schedule(
            init_value=float(kwargs["init_value"]),
            end_value=float(kwargs["peak_value"]),
            transition_steps=warmup_steps,
        )
        decay_steps = max(1, steps - warmup_steps)
        decay = optax.linear_schedule(
            init_value=float(kwargs["peak_value"]),
            end_value=float(kwargs["end_value"]),
            transition_steps=decay_steps,
        )
        return optax.join_schedules([warmup, decay], [warmup_steps])

    if schedule_name == "constant":
        return optax.constant_schedule(float(kwargs["peak_value"]))

    raise ValueError(
        f"Unsupported lr_schedule.name={schedule_cfg.name!r} (expected warmup_cosine|warmup_linear|constant)"
    )


def _scale_by_muon(
    *,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    eps: float = 1e-7,
    mu_dtype: Any | None = None,
    weight_dimension_numbers: Any | None = None,
):
    """Compatibility wrapper around `optax.contrib.scale_by_muon`.

    Some project code (e.g. `projects/nano_gpt/runner.py`) uses this helper to
    build a custom Muon+AdamW split optimizer. We keep it here so project code
    can depend on a stable in-repo entry point instead of relying on optax's
    evolving contrib API.
    """
    import optax

    contrib = getattr(optax, "contrib", None)
    scale_by_muon_fn = getattr(contrib, "scale_by_muon", None) if contrib is not None else None
    if scale_by_muon_fn is None:
        raise ValueError("Muon requires optax.contrib.scale_by_muon (upgrade optax).")

    dim_nums_cls = getattr(contrib, "MuonDimensionNumbers", None) if contrib is not None else None
    if weight_dimension_numbers is None and dim_nums_cls is not None:
        # Newer optax versions require a weight-dimension spec callback.
        def weight_dimension_numbers(updates):  # noqa: ANN001
            import jax

            return jax.tree_util.tree_map(lambda _x: dim_nums_cls(), updates)

    kwargs = {
        "nesterov": bool(nesterov),
        "ns_steps": int(ns_steps),
        "eps": float(eps),
    }
    if weight_dimension_numbers is not None:
        kwargs["weight_dimension_numbers"] = weight_dimension_numbers
    if mu_dtype is not None:
        kwargs["mu_dtype"] = mu_dtype

    try:
        return scale_by_muon_fn(beta=float(momentum), **kwargs)
    except TypeError:
        # Older optax versions may not accept `mu_dtype` and/or
        # `weight_dimension_numbers`. Retry with the minimal signature.
        kwargs.pop("mu_dtype", None)
        kwargs.pop("weight_dimension_numbers", None)
        return scale_by_muon_fn(beta=float(momentum), **kwargs)


def build_tx(*, training_steps: int, cfg: OptimizerConfig, params: Any | None = None):
    """Build an Optax optimizer transformation.

    This is intended to be passed to `training2.get_state(..., tx=...)`.
    """
    import jax
    import jax.numpy as jnp
    import optax

    normalized_cfg = normalize_optimizer_config(cfg)
    lr_schedule = build_lr_schedule(training_steps=training_steps, cfg=normalized_cfg.lr_schedule)
    name = normalized_cfg.name
    optimizer_kwargs = normalized_cfg.kwargs
    weight_decay = float(optimizer_kwargs["weight_decay"])

    if name == "lion":
        base = optax.lion(lr_schedule, weight_decay=weight_decay)
    elif name == "adamw":
        base = optax.adamw(lr_schedule, weight_decay=weight_decay)
    elif name == "sgd":
        base = optax.sgd(lr_schedule)
        if weight_decay != 0.0:
            base = optax.chain(optax.add_decayed_weights(weight_decay), base)
    elif name == "muon":
        if params is None:
            raise ValueError("optimizer.name='muon' requires passing `params=` to build the parameter mask.")

        muon_max_dim = int(optimizer_kwargs["max_dim"])
        if muon_max_dim <= 0:
            raise ValueError(f"muon_max_dim must be > 0, got {muon_max_dim}")

        aux_lr_schedule_kwargs = dict(normalized_cfg.lr_schedule.kwargs)
        aux_lr_schedule_kwargs["peak_value"] = float(optimizer_kwargs["aux_lr"])
        aux_lr_cfg = LRScheduleConfig(name=normalized_cfg.lr_schedule.name, kwargs=aux_lr_schedule_kwargs)
        aux_lr_schedule = build_lr_schedule(training_steps=training_steps, cfg=aux_lr_cfg)

        # NOTE: We use a Muon + auxiliary AdamW split via `optax.multi_transform`
        # so we can tune the auxiliary LR separately.
        contrib = getattr(optax, "contrib", None)
        scale_by_muon_fn = getattr(contrib, "scale_by_muon", None) if contrib is not None else None
        dim_nums_cls = getattr(contrib, "MuonDimensionNumbers", None) if contrib is not None else None
        if scale_by_muon_fn is None or dim_nums_cls is None:
            raise ValueError("optimizer.name='muon' requires optax.contrib.scale_by_muon (upgrade optax).")

        # Optax's `scale_by_muon` expects an explicit weight-dimension spec tree
        # (or callable) matching the (masked) updates structure. We provide the
        # default 2D spec everywhere; masked subtrees remain MaskedNode().
        def weight_dimension_numbers(updates):
            return jax.tree_util.tree_map(lambda _x: dim_nums_cls(), updates)

        muon_scale_kwargs = {
            "nesterov": bool(optimizer_kwargs["nesterov"]),
            "ns_steps": int(optimizer_kwargs["ns_steps"]),
            "eps": float(optimizer_kwargs["eps"]),
            "mu_dtype": jnp.float32,
            "weight_dimension_numbers": weight_dimension_numbers,
        }
        try:
            scale_by_muon = scale_by_muon_fn(beta=float(optimizer_kwargs["momentum"]), **muon_scale_kwargs)
        except TypeError:
            muon_scale_kwargs.pop("mu_dtype", None)
            scale_by_muon = scale_by_muon_fn(beta=float(optimizer_kwargs["momentum"]), **muon_scale_kwargs)

        muon_chain = [scale_by_muon]
        if weight_decay != 0.0:
            # Decoupled weight decay: do NOT include it inside the orthogonalization.
            muon_chain.append(optax.add_decayed_weights(weight_decay))
        muon_chain.extend([optax.scale_by_schedule(lr_schedule), optax.scale(-1.0)])
        muon_tx = optax.chain(*muon_chain)
        aux_tx = optax.adamw(aux_lr_schedule, weight_decay=weight_decay)

        def label(p):
            if getattr(p, "ndim", 0) != 2:
                return "adamw"
            m = int(p.shape[0])
            n = int(p.shape[1])
            if max(m, n) > int(muon_max_dim):
                return "adamw"
            return "muon"

        param_labels = jax.tree_util.tree_map(label, params)
        base = optax.multi_transform({"muon": muon_tx, "adamw": aux_tx}, param_labels)
    else:
        raise ValueError(f"Unsupported optimizer.name={cfg.name!r} (expected lion|adamw|sgd|muon)")

    clip_norm = float(optimizer_kwargs["clip_norm"])
    if clip_norm <= 0.0:
        return base
    return optax.chain(optax.clip_by_global_norm(clip_norm), base)


__all__ = [
    "LRScheduleConfig",
    "OptimizerConfig",
    "_scale_by_muon",
    "build_lr_schedule",
    "build_tx",
    "normalize_lr_schedule_config",
    "normalize_lr_schedule_name",
    "normalize_optimizer_config",
    "normalize_optimizer_name",
    "sanitize_optimizer_config_for_logging",
]
