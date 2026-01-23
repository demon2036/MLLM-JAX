from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LRScheduleConfig:
    """Learning-rate schedule config (Optax).

    Defaults match the hardcoded schedule in `training2.get_state`.
    """

    # Supported: warmup_cosine, constant
    type: str = "warmup_cosine"

    init_value: float = 0.0
    peak_value: float = 1e-6
    end_value: float = 0.0

    # If warmup_steps is None, derive: round(training_steps * warmup_ratio).
    warmup_ratio: float = 0.05
    warmup_steps: int | None = None


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer config (Optax).

    Defaults match the hardcoded optimizer in `training2.get_state`.
    """

    # Supported: lion, adamw, sgd
    name: str = "lion"

    clip_norm: float = 1.0
    weight_decay: float = 1e-8

    lr_schedule: LRScheduleConfig = field(default_factory=LRScheduleConfig)


def build_lr_schedule(*, training_steps: int, cfg: LRScheduleConfig):
    import optax

    steps = int(training_steps)
    if steps <= 0:
        raise ValueError(f"training_steps must be > 0, got {steps}")

    schedule_type = str(cfg.type).strip().lower()
    if schedule_type in {"warmup_cosine", "warmup_cosine_decay"}:
        warmup_steps = cfg.warmup_steps
        if warmup_steps is None:
            warmup_steps = int(round(float(steps) * float(cfg.warmup_ratio)))
        warmup_steps = int(warmup_steps)
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        return optax.warmup_cosine_decay_schedule(
            init_value=float(cfg.init_value),
            peak_value=float(cfg.peak_value),
            warmup_steps=warmup_steps,
            decay_steps=steps,
            end_value=float(cfg.end_value),
        )

    if schedule_type in {"constant", "const"}:
        return optax.constant_schedule(float(cfg.peak_value))

    raise ValueError(f"Unsupported lr_schedule.type={cfg.type!r} (expected warmup_cosine|constant)")


def build_tx(*, training_steps: int, cfg: OptimizerConfig):
    """Build an Optax optimizer transformation.

    This is intended to be passed to `training2.get_state(..., tx=...)`.
    """
    import optax

    lr_schedule = build_lr_schedule(training_steps=training_steps, cfg=cfg.lr_schedule)
    name = str(cfg.name).strip().lower()
    weight_decay = float(cfg.weight_decay)

    if name == "lion":
        base = optax.lion(lr_schedule, weight_decay=weight_decay)
    elif name == "adamw":
        base = optax.adamw(lr_schedule, weight_decay=weight_decay)
    elif name == "sgd":
        base = optax.sgd(lr_schedule)
        if weight_decay != 0.0:
            base = optax.chain(optax.add_decayed_weights(weight_decay), base)
    else:
        raise ValueError(f"Unsupported optimizer.name={cfg.name!r} (expected lion|adamw|sgd)")

    clip_norm = float(cfg.clip_norm)
    if clip_norm <= 0.0:
        return base
    return optax.chain(optax.clip_by_global_norm(clip_norm), base)


__all__ = ["LRScheduleConfig", "OptimizerConfig", "build_lr_schedule", "build_tx"]

