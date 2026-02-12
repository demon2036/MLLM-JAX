from __future__ import annotations

import os
import subprocess
import sys
import time
from argparse import ArgumentParser
from typing import Any

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from plugins.training.core.runtime.env import load_dotenv_if_present
from plugins.training.rl.config import load_config
from plugins.training.core.optim.optimizer import LRScheduleConfig, OptimizerConfig, normalize_optimizer_config
from plugins.training.rl.algorithms import (
    AlgoConfig,
    PluginConfig,
    normalize_algo_config,
)
from projects.gsm8k_grpo.config_schema import (
    GRPOCheckpointConfig,
    GRPODynamicSamplingConfig,
    GRPOGsm8kConfig,
    GRPORolloutConfig,
    GRPOTrainConfig,
)


def _maybe_git_short_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None
    return out or None


def _get_by_path(cfg: dict[str, Any], key_path: str) -> Any:
    keys = [k for k in key_path.split(".") if k]
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _get_int_from_aliases(
    cfg: dict[str, Any],
    *,
    label: str,
    paths: list[str] | None = None,
    keys: list[str] | None = None,
) -> int | None:
    paths = paths or []
    keys = keys or []

    found: list[tuple[str, Any]] = []
    for path in paths:
        value = _get_by_path(cfg, path)
        if value is not None:
            found.append((path, value))
    for key in keys:
        value = cfg.get(key)
        if value is not None:
            found.append((key, value))

    if not found:
        return None

    parsed: list[tuple[str, int]] = []
    for src, value in found:
        try:
            parsed.append((src, int(value)))
        except Exception as e:  # pragma: no cover
            raise ValueError(f"{label} must be an int, got {src}={value!r}") from e

    unique_values = {v for _src, v in parsed}
    if len(unique_values) > 1:
        details = ", ".join(f"{src}={v}" for src, v in parsed)
        raise ValueError(f"Conflicting {label} values: {details}")
    return parsed[0][1]


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


def _strict_legacy_key_guard(cfg: dict[str, Any]) -> None:
    # Intentionally breaking migration: old shape keys are rejected.
    legacy_paths = {
        "algo.name": _get_by_path(cfg, "algo.name"),
        "algo.estimator.eps": _get_by_path(cfg, "algo.estimator.eps"),
        "algo.estimator.clip_range": _get_by_path(cfg, "algo.estimator.clip_range"),
        "algo.estimator.dapo_alpha": _get_by_path(cfg, "algo.estimator.dapo_alpha"),
        "algo.estimator.rloo_whiten": _get_by_path(cfg, "algo.estimator.rloo_whiten"),
        "algo.estimator.gae_gamma": _get_by_path(cfg, "algo.estimator.gae_gamma"),
        "algo.estimator.gae_lambda": _get_by_path(cfg, "algo.estimator.gae_lambda"),
        "algo.estimator.gae_normalize": _get_by_path(cfg, "algo.estimator.gae_normalize"),
        "algo.estimator.common": _get_by_path(cfg, "algo.estimator.common"),
        "algo.estimator.grpo": _get_by_path(cfg, "algo.estimator.grpo"),
        "algo.estimator.maxrl": _get_by_path(cfg, "algo.estimator.maxrl"),
        "algo.estimator.reinforce": _get_by_path(cfg, "algo.estimator.reinforce"),
        "algo.estimator.reinforcepp": _get_by_path(cfg, "algo.estimator.reinforcepp"),
        "algo.estimator.dapo": _get_by_path(cfg, "algo.estimator.dapo"),
        "algo.estimator.rloo": _get_by_path(cfg, "algo.estimator.rloo"),
        "algo.estimator.gae": _get_by_path(cfg, "algo.estimator.gae"),
        "algo.update.policy_gradient": _get_by_path(cfg, "algo.update.policy_gradient"),
        "algo.update.ppo": _get_by_path(cfg, "algo.update.ppo"),
        "algo.update.value_coef": _get_by_path(cfg, "algo.update.value_coef"),
        "algo.update.value_clip_range": _get_by_path(cfg, "algo.update.value_clip_range"),
        "algo.update.entropy_coef": _get_by_path(cfg, "algo.update.entropy_coef"),
        "algo.ppo_value_coef": _get_by_path(cfg, "algo.ppo_value_coef"),
        "algo.ppo_value_clip_range": _get_by_path(cfg, "algo.ppo_value_clip_range"),
        "algo.ppo_entropy_coef": _get_by_path(cfg, "algo.ppo_entropy_coef"),
        "algo.ppo_gamma": _get_by_path(cfg, "algo.ppo_gamma"),
        "algo.ppo_gae_lambda": _get_by_path(cfg, "algo.ppo_gae_lambda"),
        "algo.ppo_advantage_norm": _get_by_path(cfg, "algo.ppo_advantage_norm"),
        "train.optimizer.clip_norm": _get_by_path(cfg, "train.optimizer.clip_norm"),
        "train.optimizer.weight_decay": _get_by_path(cfg, "train.optimizer.weight_decay"),
        "train.optimizer.muon_aux_lr": _get_by_path(cfg, "train.optimizer.muon_aux_lr"),
        "train.optimizer.muon_momentum": _get_by_path(cfg, "train.optimizer.muon_momentum"),
        "train.optimizer.muon_nesterov": _get_by_path(cfg, "train.optimizer.muon_nesterov"),
        "train.optimizer.muon_ns_steps": _get_by_path(cfg, "train.optimizer.muon_ns_steps"),
        "train.optimizer.muon_eps": _get_by_path(cfg, "train.optimizer.muon_eps"),
        "train.optimizer.muon_max_dim": _get_by_path(cfg, "train.optimizer.muon_max_dim"),
        "train.optimizer.lr_schedule.type": _get_by_path(cfg, "train.optimizer.lr_schedule.type"),
        "train.optimizer.lr_schedule.init_value": _get_by_path(cfg, "train.optimizer.lr_schedule.init_value"),
        "train.optimizer.lr_schedule.peak_value": _get_by_path(cfg, "train.optimizer.lr_schedule.peak_value"),
        "train.optimizer.lr_schedule.end_value": _get_by_path(cfg, "train.optimizer.lr_schedule.end_value"),
        "train.optimizer.lr_schedule.warmup_ratio": _get_by_path(cfg, "train.optimizer.lr_schedule.warmup_ratio"),
        "train.optimizer.lr_schedule.warmup_steps": _get_by_path(cfg, "train.optimizer.lr_schedule.warmup_steps"),
    }
    found = {k: v for k, v in legacy_paths.items() if v is not None}
    if not found:
        return

    details = ", ".join(f"{k}={v!r}" for k, v in sorted(found.items()))
    raise ValueError(
        "Detected deprecated RL config keys (breaking schema v3). "
        "Use plugin schema: algo.estimator={name,kwargs}, algo.update={name,kwargs}, "
        "train.optimizer={name,kwargs,lr_schedule:{name,kwargs}}. "
        f"Got: {details}"
    )


def _parse_plugin_config(raw: Any, *, defaults: PluginConfig, label: str) -> PluginConfig:
    if raw is None:
        return defaults
    if isinstance(raw, str):
        return PluginConfig(name=str(raw), kwargs={})
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a dict or string when provided")

    name_raw = raw.get("name")
    plugin_name = str(name_raw) if name_raw is not None else defaults.name
    kwargs_raw = raw.get("kwargs")
    if kwargs_raw is None:
        kwargs = dict(defaults.kwargs)
    elif isinstance(kwargs_raw, dict):
        kwargs = dict(kwargs_raw)
    else:
        raise ValueError(f"{label}.kwargs must be a dict when provided")
    return PluginConfig(name=plugin_name, kwargs=kwargs)


def _cfg_from_dict(cfg: dict[str, Any], *, config_path: str) -> GRPOGsm8kConfig:
    model_path = str(cfg.get("model_path") or "Qwen/Qwen2.5-3B-Instruct")
    steps = int(cfg.get("steps") or 100)

    rollout_n = _get_int_from_aliases(
        cfg,
        label="rollout.n",
        paths=["rollout.n", "rollout.num_pre_q"],
        keys=["rollout_n", "rollout_num_pre_q", "n", "num_pre_q"],
    )
    rollout_n = int(rollout_n or 8)

    rollout_batch_size = _get_int_from_aliases(
        cfg,
        label="rollout.batch_size",
        paths=["rollout.batch_size"],
        keys=["rollout_batch_size", "batch_size"],
    )
    rollout_batch_size = int(rollout_batch_size or 32)

    rollout_micro_batch_size = _get_int_from_aliases(
        cfg,
        label="rollout.micro_batch_size",
        paths=["rollout.micro_batch_size"],
        keys=["rollout_micro_batch_size"],
    )
    rollout_micro_batch_size_per_device = _get_int_from_aliases(
        cfg,
        label="rollout.micro_batch_size_per_device",
        paths=["rollout.micro_batch_size_per_device", "rollout.per_device_micro_batch_size"],
        keys=["rollout_micro_batch_size_per_device", "rollout_per_device_micro_batch_size"],
    )
    if rollout_micro_batch_size_per_device is None:
        raise ValueError(
            "rollout.micro_batch_size_per_device is required (sequences per device per rollout pass). "
            "Set it explicitly in YAML."
        )
    rollout_micro_batch_size_per_device = int(rollout_micro_batch_size_per_device)
    if rollout_micro_batch_size_per_device <= 0:
        raise ValueError("rollout.micro_batch_size_per_device must be > 0")

    if rollout_micro_batch_size is not None:
        rollout_micro_batch_size = int(rollout_micro_batch_size)
        if rollout_micro_batch_size <= 0:
            raise ValueError("rollout.micro_batch_size must be > 0 when set")

    deprecated_rollout_keys = {
        "rollout.batch_size_per_process": _get_by_path(cfg, "rollout.batch_size_per_process"),
        "rollout.batch_size_per_device": _get_by_path(cfg, "rollout.batch_size_per_device"),
        "rollout.prompts_per_pass_per_process": _get_by_path(cfg, "rollout.prompts_per_pass_per_process"),
        "rollout.prompts_per_pass_per_device": _get_by_path(cfg, "rollout.prompts_per_pass_per_device"),
        "rollout.global_prompt_batch_size": _get_by_path(cfg, "rollout.global_prompt_batch_size"),
        "rollout.global_sequence_batch_size": _get_by_path(cfg, "rollout.global_sequence_batch_size"),
        "rollout.global_batch_size": _get_by_path(cfg, "rollout.global_batch_size"),
        "rollout.prompt_batch_size": _get_by_path(cfg, "rollout.prompt_batch_size"),
        "rollout.prompt_batch_size_per_process": _get_by_path(cfg, "rollout.prompt_batch_size_per_process"),
        "rollout.prompt_batch_size_per_device": _get_by_path(cfg, "rollout.prompt_batch_size_per_device"),
        "rollout.per_device_batch_size": _get_by_path(cfg, "rollout.per_device_batch_size"),
        "prompt_batch_size": cfg.get("prompt_batch_size"),
        "rollout_prompt_batch_size": cfg.get("rollout_prompt_batch_size"),
    }
    deprecated_rollout_keys = {k: v for k, v in deprecated_rollout_keys.items() if v is not None}
    if deprecated_rollout_keys:
        details = ", ".join(f"{k}={v!r}" for k, v in deprecated_rollout_keys.items())
        raise ValueError(
            "Deprecated rollout batch size keys are no longer supported. "
            "Use `rollout.batch_size`, `rollout.n`, and explicit rollout micro-batch keys: "
            "`rollout.micro_batch_size_per_device` (required), optionally `rollout.micro_batch_size`. "
            f"Got: {details}"
        )

    global_length = _get_by_path(cfg, "rollout.global_length")
    if global_length is None:
        global_length = cfg.get("global_length")
    global_length = int(global_length or 512)

    max_length_sample = _get_by_path(cfg, "rollout.max_length_sample")
    if max_length_sample is None:
        max_length_sample = cfg.get("max_length_sample")
    max_length_sample = int(max_length_sample or 64)

    rollout_backend_raw = _get_by_path(cfg, "rollout.backend")
    if rollout_backend_raw is None:
        rollout_backend_raw = cfg.get("rollout_backend")
    rollout_backend = str(rollout_backend_raw or "naive")

    dynamic_sampling_raw = _get_by_path(cfg, "rollout.dynamic_sampling")
    if dynamic_sampling_raw is None:
        dynamic_sampling_cfg = GRPODynamicSamplingConfig()
    elif not isinstance(dynamic_sampling_raw, dict):
        raise ValueError("rollout.dynamic_sampling must be a dict when provided")
    else:
        enabled_raw = dynamic_sampling_raw.get("enabled")
        enabled = False if enabled_raw is None else _as_bool(enabled_raw, label="rollout.dynamic_sampling.enabled")

        trigger = str(dynamic_sampling_raw.get("trigger") or "homogeneous_group").strip().lower()
        if trigger != "homogeneous_group":
            raise ValueError("rollout.dynamic_sampling.trigger must be 'homogeneous_group'")

        metric = str(dynamic_sampling_raw.get("metric") or "acc").strip().lower()
        if metric not in {"acc", "seq_reward"}:
            raise ValueError("rollout.dynamic_sampling.metric must be one of: acc, seq_reward")

        homogeneity_threshold_raw = dynamic_sampling_raw.get("homogeneity_threshold")
        homogeneity_threshold = 1.0 if homogeneity_threshold_raw is None else float(homogeneity_threshold_raw)
        if not (0.0 < homogeneity_threshold <= 1.0):
            raise ValueError("rollout.dynamic_sampling.homogeneity_threshold must be in (0, 1]")

        min_unique_reward_values_raw = dynamic_sampling_raw.get("min_unique_reward_values")
        min_unique_reward_values = (
            2 if min_unique_reward_values_raw is None else int(min_unique_reward_values_raw)
        )
        if min_unique_reward_values < 1:
            raise ValueError("rollout.dynamic_sampling.min_unique_reward_values must be >= 1")

        max_extra_roll_rounds_raw = dynamic_sampling_raw.get("max_extra_roll_rounds")
        max_extra_roll_rounds = 10 if max_extra_roll_rounds_raw is None else int(max_extra_roll_rounds_raw)
        if max_extra_roll_rounds < 0:
            raise ValueError("rollout.dynamic_sampling.max_extra_roll_rounds must be >= 0")

        target_valid_groups_raw = dynamic_sampling_raw.get("target_valid_groups")
        if target_valid_groups_raw is None:
            target_valid_groups = None
        else:
            target_valid_groups = int(target_valid_groups_raw)
            if target_valid_groups <= 0:
                raise ValueError("rollout.dynamic_sampling.target_valid_groups must be > 0 when set")

        fallback_policy = str(dynamic_sampling_raw.get("fallback_policy") or "keep_last").strip().lower()
        if fallback_policy not in {"keep_last", "drop_group"}:
            raise ValueError("rollout.dynamic_sampling.fallback_policy must be one of: keep_last, drop_group")

        dynamic_sampling_cfg = GRPODynamicSamplingConfig(
            enabled=enabled,
            trigger=trigger,
            metric=metric,
            homogeneity_threshold=homogeneity_threshold,
            min_unique_reward_values=min_unique_reward_values,
            max_extra_roll_rounds=max_extra_roll_rounds,
            target_valid_groups=target_valid_groups,
            fallback_policy=fallback_policy,
        )

    train_micro_batch_size = _get_int_from_aliases(
        cfg,
        label="train.micro_batch_size",
        paths=["train.micro_batch_size"],
        keys=["train_micro_batch_size"],
    )
    train_micro_batch_size_per_device = _get_int_from_aliases(
        cfg,
        label="train.micro_batch_size_per_device",
        paths=["train.micro_batch_size_per_device", "train.per_device_micro_batch_size"],
        keys=["train_micro_batch_size_per_device", "train_per_device_micro_batch_size"],
    )

    deprecated_train_keys = {
        "train.global_micro_batch_size": _get_by_path(cfg, "train.global_micro_batch_size"),
        "train.micro_batch_size_per_process": _get_by_path(cfg, "train.micro_batch_size_per_process"),
        "train_global_micro_batch_size": cfg.get("train_global_micro_batch_size"),
        "train_micro_batch_size_per_process": cfg.get("train_micro_batch_size_per_process"),
    }
    deprecated_train_keys = {k: v for k, v in deprecated_train_keys.items() if v is not None}
    if deprecated_train_keys:
        details = ", ".join(f"{k}={v!r}" for k, v in deprecated_train_keys.items())
        raise ValueError(
            "Deprecated train micro-batch keys are no longer supported. "
            "Use `train.micro_batch_size` (sequences per process per micro-step) and/or "
            "`train.micro_batch_size_per_device` only. "
            f"Got: {details}"
        )

    max_length_total_raw = _get_by_path(cfg, "train.max_length_total")
    if max_length_total_raw is None:
        max_length_total_raw = cfg.get("max_length_total")
    max_length_total = int(max_length_total_raw) if max_length_total_raw is not None else max_length_sample + 128

    ppo_epochs = _get_by_path(cfg, "train.ppo_epochs")
    if ppo_epochs is None:
        ppo_epochs = cfg.get("ppo_epochs")
    ppo_epochs = int(ppo_epochs or 1)

    grad_accum_steps = _get_by_path(cfg, "train.grad_accum_steps")
    if grad_accum_steps is None:
        grad_accum_steps = cfg.get("grad_accum_steps")
    grad_accum_steps = int(grad_accum_steps or 1)

    beta = _get_by_path(cfg, "train.beta")
    if beta is None:
        beta = cfg.get("beta")
    beta = float(beta or 0.0)

    gradient_checkpointing_raw = _get_by_path(cfg, "train.gradient_checkpointing")
    if gradient_checkpointing_raw is None:
        gradient_checkpointing_raw = cfg.get("train_gradient_checkpointing")
    if gradient_checkpointing_raw is None:
        gradient_checkpointing = True
    else:
        gradient_checkpointing = _as_bool(gradient_checkpointing_raw, label="train.gradient_checkpointing")

    mesh_shape = str(cfg.get("mesh_shape") or "1,-1,1")

    optimizer_raw = _get_by_path(cfg, "train.optimizer")
    if optimizer_raw is None:
        optimizer_cfg = OptimizerConfig()
    elif isinstance(optimizer_raw, str):
        optimizer_cfg = OptimizerConfig(name=str(optimizer_raw), kwargs={}, lr_schedule=LRScheduleConfig())
    elif isinstance(optimizer_raw, dict):
        name_raw = optimizer_raw.get("name")
        kwargs_raw = optimizer_raw.get("kwargs")
        if kwargs_raw is None:
            kwargs = {}
        elif isinstance(kwargs_raw, dict):
            kwargs = dict(kwargs_raw)
        else:
            raise ValueError("train.optimizer.kwargs must be a dict when provided")

        lr_raw = optimizer_raw.get("lr_schedule")
        if lr_raw is None:
            lr_cfg = LRScheduleConfig()
        elif isinstance(lr_raw, str):
            lr_cfg = LRScheduleConfig(name=str(lr_raw), kwargs={})
        elif isinstance(lr_raw, dict):
            lr_name_raw = lr_raw.get("name")
            lr_kwargs_raw = lr_raw.get("kwargs")
            if lr_kwargs_raw is None:
                lr_kwargs = {}
            elif isinstance(lr_kwargs_raw, dict):
                lr_kwargs = dict(lr_kwargs_raw)
            else:
                raise ValueError("train.optimizer.lr_schedule.kwargs must be a dict when provided")
            lr_cfg = LRScheduleConfig(name=str(lr_name_raw) if lr_name_raw is not None else "warmup_cosine", kwargs=lr_kwargs)
        else:
            raise ValueError("train.optimizer.lr_schedule must be a dict or string when provided")

        optimizer_cfg = OptimizerConfig(
            name=str(name_raw) if name_raw is not None else "lion",
            kwargs=kwargs,
            lr_schedule=lr_cfg,
        )
    else:
        raise ValueError(f"train.optimizer must be a dict or string, got {type(optimizer_raw).__name__}")

    optimizer_cfg = normalize_optimizer_config(optimizer_cfg)

    wandb_project = str(cfg.get("wandb_project") or "mllm-jax-grpo-gsm8k")

    wandb_mode_raw = cfg.get("wandb_mode")
    if wandb_mode_raw is None or str(wandb_mode_raw).strip() == "":
        wandb_mode = "online" if os.environ.get("WANDB_API_KEY") else "disabled"
    else:
        wandb_mode = str(wandb_mode_raw).strip().lower()
    if wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError("wandb_mode must be one of: online, offline, disabled")

    wandb_name = cfg.get("wandb_name")
    if wandb_name is None or str(wandb_name).strip() == "":
        ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        tag = os.path.basename(str(config_path))
        if tag.endswith(".yaml"):
            tag = tag[: -len(".yaml")]
        tag = tag or "grpo_gsm8k"
        sha = _maybe_git_short_sha()
        wandb_name = f"{tag}_{sha}_{ts}" if sha else f"{tag}_{ts}"
    wandb_name = str(wandb_name)

    reward_weights_raw = cfg.get("reward_weights") or (1.0, 0.5, 0.5)
    if isinstance(reward_weights_raw, (list, tuple)) and len(reward_weights_raw) == 3:
        reward_weights = tuple(float(x) for x in reward_weights_raw)
    else:
        raise ValueError("reward_weights must be a list/tuple of 3 floats")

    algo_raw = cfg.get("algo")
    defaults = AlgoConfig()
    if algo_raw is None:
        algo_cfg = defaults
    elif isinstance(algo_raw, str):
        algo_cfg = AlgoConfig(estimator=PluginConfig(name=str(algo_raw), kwargs={}), update=defaults.update)
    elif isinstance(algo_raw, dict):
        estimator_cfg = _parse_plugin_config(algo_raw.get("estimator"), defaults=defaults.estimator, label="algo.estimator")
        update_cfg = _parse_plugin_config(algo_raw.get("update"), defaults=defaults.update, label="algo.update")
        algo_cfg = AlgoConfig(
            name=defaults.name,
            estimator=estimator_cfg,
            update=update_cfg,
        )
    else:
        raise ValueError(f"algo must be a dict, got {type(algo_raw).__name__}")

    # strict normalization + cross-field validation
    algo_cfg, _algo_name, _estimator_name, _update_name = normalize_algo_config(algo_cfg)

    eval_every_steps = int(cfg.get("eval_every_steps") or 0)
    eval_batches_per_process = _get_int_from_aliases(
        cfg,
        label="eval_batches_per_process",
        paths=["eval_batches_per_process", "eval_batches"],
        keys=[],
    )
    eval_batches_per_process = int(eval_batches_per_process or 1)
    eval_split = str(cfg.get("eval_split") or "test")

    eval_rollout_n_raw = cfg.get("eval_rollout_n")
    eval_rollout_n = int(eval_rollout_n_raw) if eval_rollout_n_raw is not None else 1
    if eval_rollout_n < 1:
        raise ValueError("eval_rollout_n must be >= 1")

    eval_full_every_steps_raw = cfg.get("eval_full_every_steps")
    eval_full_every_steps = int(eval_full_every_steps_raw) if eval_full_every_steps_raw is not None else 0
    if eval_full_every_steps < 0:
        raise ValueError("eval_full_every_steps must be >= 0")

    eval_full_sweep_raw = cfg.get("eval_full_sweep")
    eval_full_sweep = _as_bool(eval_full_sweep_raw, label="eval_full_sweep") if eval_full_sweep_raw is not None else False

    checkpoint_raw = cfg.get("checkpoint")
    if checkpoint_raw is None:
        checkpoint_cfg = GRPOCheckpointConfig()
    elif not isinstance(checkpoint_raw, dict):
        raise ValueError("checkpoint must be a dict when provided")
    else:
        dir_raw = checkpoint_raw.get("dir")
        if dir_raw is None:
            dir_raw = checkpoint_raw.get("path")
        ckpt_dir = str(dir_raw or "").strip()

        save_every_raw = checkpoint_raw.get("save_every_steps")
        if save_every_raw is None:
            save_every_raw = checkpoint_raw.get("every_steps")
        save_every_steps = int(save_every_raw or 0)
        if save_every_steps < 0:
            raise ValueError("checkpoint.save_every_steps must be >= 0")

        max_to_keep_raw = checkpoint_raw.get("max_to_keep")
        if max_to_keep_raw is None:
            max_to_keep = 3
        else:
            if isinstance(max_to_keep_raw, str) and max_to_keep_raw.strip().lower() in {"none", "null"}:
                max_to_keep = None
            elif max_to_keep_raw is None:
                max_to_keep = None
            else:
                max_to_keep = int(max_to_keep_raw)
                if max_to_keep <= 0:
                    raise ValueError("checkpoint.max_to_keep must be > 0 when set")

        resume_raw = checkpoint_raw.get("resume")
        resume = True if resume_raw is None else _as_bool(resume_raw, label="checkpoint.resume")

        if ckpt_dir != "" and save_every_steps == 0:
            raise ValueError("checkpoint.dir is set but checkpoint.save_every_steps=0 (disabled). Set it to >= 1.")

        checkpoint_cfg = GRPOCheckpointConfig(
            dir=ckpt_dir,
            save_every_steps=save_every_steps,
            max_to_keep=max_to_keep,
            resume=resume,
        )

    return GRPOGsm8kConfig(
        config_path=str(config_path),
        model_path=model_path,
        steps=steps,
        rollout=GRPORolloutConfig(
            backend=rollout_backend,
            batch_size=rollout_batch_size,
            n=rollout_n,
            global_length=global_length,
            max_length_sample=max_length_sample,
            micro_batch_size=rollout_micro_batch_size,
            micro_batch_size_per_device=rollout_micro_batch_size_per_device,
            dynamic_sampling=dynamic_sampling_cfg,
        ),
        train=GRPOTrainConfig(
            micro_batch_size_per_device=train_micro_batch_size_per_device,
            micro_batch_size=train_micro_batch_size,
            max_length_total=max_length_total,
            ppo_epochs=ppo_epochs,
            grad_accum_steps=grad_accum_steps,
            beta=beta,
            gradient_checkpointing=gradient_checkpointing,
            optimizer=optimizer_cfg,
        ),
        mesh_shape=mesh_shape,
        wandb_project=wandb_project,
        wandb_mode=wandb_mode,
        wandb_name=wandb_name,
        algo=algo_cfg,
        reward_weights=reward_weights,
        eval_every_steps=eval_every_steps,
        eval_batches_per_process=eval_batches_per_process,
        eval_split=eval_split,
        eval_rollout_n=eval_rollout_n,
        eval_full_every_steps=eval_full_every_steps,
        eval_full_sweep=eval_full_sweep,
        checkpoint=checkpoint_cfg,
    )


def _sanitize_config_for_display(cfg: GRPOGsm8kConfig) -> dict[str, object]:
    out = cfg.to_logging_dict()
    out["rollout"]["sequences_global_per_step"] = int(cfg.rollout.batch_size) * int(cfg.rollout.n)
    return out


def main() -> None:
    parser = ArgumentParser(description="Run GRPO/GSM8K training (projects/gsm8k_grpo).")
    parser.add_argument(
        "--config",
        default="projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_batch128_roll8_literal_v6e8.yaml",
        help="YAML config path.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config entries (repeatable), e.g. --set steps=20",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved config (YAML) and exit (no JAX required).",
    )
    args = parser.parse_args()

    load_dotenv_if_present(repo_root=REPO_ROOT)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    deprecated_env_overrides = [
        "MODEL_PATH",
        "STEPS",
        "ROLLOUT_BACKEND",
        "ROLLOUT_BATCH_SIZE",
        "BATCH_SIZE",
        "ROLLOUT_PROMPT_BATCH_SIZE",
        "ROLLOUT_N",
        "NUM_PRE_Q",
        "GLOBAL_LENGTH",
        "MAX_LENGTH_SAMPLE",
        "TRAIN_GLOBAL_MICRO_BATCH_SIZE",
        "TRAIN_MICRO_BATCH_SIZE_PER_PROCESS",
        "TRAIN_MICRO_BATCH_SIZE",
        "TRAIN_MICRO_BATCH_SIZE_PER_DEVICE",
        "TRAIN_PER_DEVICE_MICRO_BATCH_SIZE",
        "MAX_LENGTH_TOTAL",
        "PPO_EPOCHS",
        "GRAD_ACCUM_STEPS",
        "BETA",
        "MESH_SHAPE_FSDP",
        "WANDB_PROJECT",
        "WANDB_NAME",
        "WANDB_MODE",
        "EVAL_EVERY_STEPS",
        "EVAL_BATCHES",
        "EVAL_SPLIT",
    ]
    set_deprecated_env = [k for k in deprecated_env_overrides if str(os.environ.get(k, "")).strip() != ""]
    if set_deprecated_env:
        details = ", ".join(f"{k}={os.environ.get(k)!r}" for k in set_deprecated_env)
        print(f"WARNING: ignoring deprecated env var overrides (use YAML instead): {details}")

    config_path = str(args.config or "")
    cfg_dict = load_config(config_path if config_path else None, args.set)
    _strict_legacy_key_guard(cfg_dict)
    cfg = _cfg_from_dict(cfg_dict, config_path=config_path or "<default>")

    if args.print_config:
        print(yaml.safe_dump(_sanitize_config_for_display(cfg), sort_keys=False))
        return

    print(yaml.safe_dump(_sanitize_config_for_display(cfg), sort_keys=False))

    from projects.gsm8k_grpo.jax.train import run_grpo_gsm8k

    run_grpo_gsm8k(cfg)


if __name__ == "__main__":
    main()
