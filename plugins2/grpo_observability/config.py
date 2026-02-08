from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml

from plugins.training.core.optim.optimizer import LRScheduleConfig, OptimizerConfig


@dataclass(frozen=True)
class Plugins2RolloutConfig:
    k: int = 8
    global_length: int = 512
    max_length_sample: int = 96


@dataclass(frozen=True)
class Plugins2TrainConfig:
    training_steps: int = 64
    grad_accum_steps: int = 1
    beta: float = 0.0
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass(frozen=True)
class Plugins2WandbConfig:
    project: str = "plugins2-grpo-observability"
    mode: str = "online"
    name: str | None = None


@dataclass(frozen=True)
class Plugins2WebConfig:
    host: str = "0.0.0.0"
    port: int = 8080


@dataclass(frozen=True)
class Plugins2ObservabilityConfig:
    config_path: str
    model_path: str
    mesh_shape: str = "auto"
    reward_weights: tuple[float, float, float] = (1.0, 0.5, 0.5)
    rollout: Plugins2RolloutConfig = field(default_factory=Plugins2RolloutConfig)
    train: Plugins2TrainConfig = field(default_factory=Plugins2TrainConfig)
    wandb: Plugins2WandbConfig = field(default_factory=Plugins2WandbConfig)
    web: Plugins2WebConfig = field(default_factory=Plugins2WebConfig)



def _parse_lr_schedule(raw: Any) -> LRScheduleConfig:
    if raw is None:
        return LRScheduleConfig()
    if not isinstance(raw, dict):
        raise ValueError("train.optimizer.lr_schedule must be a dict")
    return LRScheduleConfig(
        type=str(raw.get("type", "warmup_cosine")),
        init_value=float(raw.get("init_value", 0.0)),
        peak_value=float(raw.get("peak_value", 1e-6)),
        end_value=float(raw.get("end_value", 0.0)),
        warmup_ratio=float(raw.get("warmup_ratio", 0.05)),
        warmup_steps=None if raw.get("warmup_steps") is None else int(raw.get("warmup_steps")),
    )


def _parse_optimizer(raw: Any) -> OptimizerConfig:
    if raw is None:
        return OptimizerConfig()
    if isinstance(raw, str):
        return OptimizerConfig(name=str(raw))
    if not isinstance(raw, dict):
        raise ValueError("train.optimizer must be a dict or string")

    lr_schedule_cfg = _parse_lr_schedule(raw.get("lr_schedule"))
    return OptimizerConfig(
        name=str(raw.get("name", "lion")),
        clip_norm=float(raw.get("clip_norm", 1.0)),
        weight_decay=float(raw.get("weight_decay", 1e-8)),
        lr_schedule=lr_schedule_cfg,
        muon_aux_lr=float(raw.get("muon_aux_lr", 3e-4)),
        muon_momentum=float(raw.get("muon_momentum", 0.95)),
        muon_nesterov=bool(raw.get("muon_nesterov", True)),
        muon_ns_steps=int(raw.get("muon_ns_steps", 5)),
        muon_eps=float(raw.get("muon_eps", 1e-7)),
        muon_max_dim=int(raw.get("muon_max_dim", 10_000)),
    )


def _normalize_wandb_mode(raw: Any) -> str:
    if raw is None or str(raw).strip() == "":
        return "online" if os.environ.get("WANDB_API_KEY") else "disabled"
    mode = str(raw).strip().lower()
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError("wandb.mode must be one of: online/offline/disabled")
    return mode


def load_plugins2_config(path: str) -> Plugins2ObservabilityConfig:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    model_path = str(payload.get("model_path") or "Qwen/Qwen2.5-0.5B-Instruct")
    mesh_shape = str(payload.get("mesh_shape") or "auto")

    reward_weights_raw = payload.get("reward_weights") or (1.0, 0.5, 0.5)
    if not isinstance(reward_weights_raw, (list, tuple)) or len(reward_weights_raw) != 3:
        raise ValueError("reward_weights must be a list/tuple of 3 floats")
    reward_weights = tuple(float(x) for x in reward_weights_raw)

    rollout_raw = payload.get("rollout") or {}
    rollout = Plugins2RolloutConfig(
        k=int(rollout_raw.get("k", 8)),
        global_length=int(rollout_raw.get("global_length", 512)),
        max_length_sample=int(rollout_raw.get("max_length_sample", 96)),
    )

    train_raw = payload.get("train") or {}
    train = Plugins2TrainConfig(
        training_steps=int(train_raw.get("training_steps", 64)),
        grad_accum_steps=int(train_raw.get("grad_accum_steps", 1)),
        beta=float(train_raw.get("beta", 0.0)),
        optimizer=_parse_optimizer(train_raw.get("optimizer")),
    )

    wandb_raw = payload.get("wandb") or {}
    wandb = Plugins2WandbConfig(
        project=str(wandb_raw.get("project") or "plugins2-grpo-observability"),
        mode=_normalize_wandb_mode(wandb_raw.get("mode")),
        name=None if wandb_raw.get("name") in (None, "") else str(wandb_raw.get("name")),
    )

    web_raw = payload.get("web") or {}
    web = Plugins2WebConfig(
        host=str(web_raw.get("host") or "0.0.0.0"),
        port=int(web_raw.get("port", 8080)),
    )

    return Plugins2ObservabilityConfig(
        config_path=str(path),
        model_path=model_path,
        mesh_shape=mesh_shape,
        reward_weights=reward_weights,
        rollout=rollout,
        train=train,
        wandb=wandb,
        web=web,
    )


__all__ = [
    "Plugins2ObservabilityConfig",
    "Plugins2RolloutConfig",
    "Plugins2TrainConfig",
    "Plugins2WandbConfig",
    "Plugins2WebConfig",
    "load_plugins2_config",
]
