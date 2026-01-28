from __future__ import annotations

import os
import subprocess
import sys
import time
from argparse import ArgumentParser
from dataclasses import asdict
from typing import Any

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from plugins.common.env import load_dotenv_if_present
from plugins.training.config import load_config
from plugins.training.algorithms import AlgoConfig, EstimatorConfig, UpdateConfig


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


def _set_by_path(cfg: dict[str, Any], key_path: str, value: Any) -> None:
    keys = [k for k in key_path.split(".") if k]
    if not keys:
        raise ValueError("Empty config key path")
    cur: dict[str, Any] = cfg
    for k in keys[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[keys[-1]] = value


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
    """Read an int value from multiple possible config locations.

    If multiple aliases are set, they must agree (otherwise raises ValueError).
    """
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


def _cfg_from_dict(cfg: dict[str, Any], *, config_path: str) -> GRPOGsm8kConfig:
    from plugins.training.runner import GRPOGsm8kConfig, GRPORolloutConfig, GRPOTrainConfig

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

    deprecated_rollout_keys = {
        "rollout.batch_size_per_process": _get_by_path(cfg, "rollout.batch_size_per_process"),
        "rollout.batch_size_per_device": _get_by_path(cfg, "rollout.batch_size_per_device"),
        "rollout.prompts_per_pass_per_process": _get_by_path(cfg, "rollout.prompts_per_pass_per_process"),
        "rollout.prompts_per_pass_per_device": _get_by_path(cfg, "rollout.prompts_per_pass_per_device"),
        "rollout.global_prompt_batch_size": _get_by_path(cfg, "rollout.global_prompt_batch_size"),
        "rollout.global_sequence_batch_size": _get_by_path(cfg, "rollout.global_sequence_batch_size"),
        "rollout.global_batch_size": _get_by_path(cfg, "rollout.global_batch_size"),
        # Legacy aliases we intentionally drop to avoid ambiguity.
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
            "Use `rollout.batch_size` (global prompts per training step) and `rollout.n` only. "
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
        # Legacy flat keys we intentionally drop to avoid ambiguity.
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
    mesh_shape = str(cfg.get("mesh_shape") or "1,-1,1")

    policy_loss_impl_raw = _get_by_path(cfg, "train.policy_loss_impl")
    if policy_loss_impl_raw is None:
        policy_loss_impl_raw = cfg.get("policy_loss_impl")
    policy_loss_impl = str(policy_loss_impl_raw or "jax").strip().lower()

    from plugins.training.update.optimizer import LRScheduleConfig, OptimizerConfig

    optimizer_raw = _get_by_path(cfg, "train.optimizer")
    if optimizer_raw is None:
        optimizer_cfg = OptimizerConfig()
    elif isinstance(optimizer_raw, str):
        optimizer_cfg = OptimizerConfig(name=str(optimizer_raw))
    elif isinstance(optimizer_raw, dict):
        lr_raw = optimizer_raw.get("lr_schedule")
        if lr_raw is None:
            lr_raw = {}
        if not isinstance(lr_raw, dict):
            raise ValueError("train.optimizer.lr_schedule must be a dict")

        warmup_steps_raw = lr_raw.get("warmup_steps")
        warmup_steps = int(warmup_steps_raw) if warmup_steps_raw is not None else None

        init_value_raw = lr_raw.get("init_value")
        peak_value_raw = lr_raw.get("peak_value")
        end_value_raw = lr_raw.get("end_value")
        warmup_ratio_raw = lr_raw.get("warmup_ratio")

        lr_cfg = LRScheduleConfig(
            type=str(lr_raw.get("type") or "warmup_cosine"),
            init_value=float(init_value_raw) if init_value_raw is not None else 0.0,
            peak_value=float(peak_value_raw) if peak_value_raw is not None else 1e-6,
            end_value=float(end_value_raw) if end_value_raw is not None else 0.0,
            warmup_ratio=float(warmup_ratio_raw) if warmup_ratio_raw is not None else 0.05,
            warmup_steps=warmup_steps,
        )

        name_raw = optimizer_raw.get("name")
        clip_norm_raw = optimizer_raw.get("clip_norm")
        weight_decay_raw = optimizer_raw.get("weight_decay")
        optimizer_cfg = OptimizerConfig(
            name=str(name_raw) if name_raw is not None else "lion",
            clip_norm=float(clip_norm_raw) if clip_norm_raw is not None else 1.0,
            weight_decay=float(weight_decay_raw) if weight_decay_raw is not None else 1e-8,
            lr_schedule=lr_cfg,
        )
    else:
        raise ValueError(f"train.optimizer must be a dict or string, got {type(optimizer_raw).__name__}")

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
    if algo_raw is None:
        algo_cfg = AlgoConfig()
    elif isinstance(algo_raw, str):
        algo_cfg = AlgoConfig(name=str(algo_raw))
    elif isinstance(algo_raw, dict):
        name_raw = algo_raw.get("name")

        defaults = AlgoConfig()
        estimator_defaults = defaults.estimator
        update_defaults = defaults.update

        estimator_raw = algo_raw.get("estimator")
        estimator_data: dict[str, Any] = {}
        estimator_name_raw = None
        if isinstance(estimator_raw, str):
            estimator_name_raw = estimator_raw
        elif isinstance(estimator_raw, dict):
            estimator_data = estimator_raw
            estimator_name_raw = estimator_raw.get("name")
        elif estimator_raw is not None:
            raise ValueError("algo.estimator must be a dict or string when provided")

        if estimator_name_raw is None:
            estimator_name_raw = algo_raw.get("estimator_name")
        if estimator_name_raw is None:
            estimator_name_raw = algo_raw.get("ppo_advantage_estimator")

        def _estimator_value(key: str, legacy_key: str | None = None) -> Any:
            if key in estimator_data:
                return estimator_data[key]
            if legacy_key is not None and legacy_key in estimator_data:
                return estimator_data[legacy_key]
            return algo_raw.get(legacy_key or key)

        eps_raw = _estimator_value("eps")
        clip_range_raw = _estimator_value("clip_range")
        rloo_whiten_raw = _estimator_value("rloo_whiten")
        dapo_alpha_raw = _estimator_value("dapo_alpha")
        gae_gamma_raw = _estimator_value("gae_gamma", "ppo_gamma")
        if "gamma" in estimator_data and "gae_gamma" not in estimator_data:
            gae_gamma_raw = estimator_data["gamma"]
        gae_lambda_raw = _estimator_value("gae_lambda", "ppo_gae_lambda")
        if "lambda" in estimator_data and "gae_lambda" not in estimator_data:
            gae_lambda_raw = estimator_data["lambda"]
        gae_normalize_raw = _estimator_value("gae_normalize", "ppo_advantage_norm")
        if "normalize" in estimator_data and "gae_normalize" not in estimator_data:
            gae_normalize_raw = estimator_data["normalize"]

        clip_range = float(clip_range_raw) if clip_range_raw is not None else None
        estimator_cfg = EstimatorConfig(
            name=str(estimator_name_raw) if estimator_name_raw is not None else estimator_defaults.name,
            eps=float(eps_raw) if eps_raw is not None else estimator_defaults.eps,
            clip_range=clip_range,
            rloo_whiten=bool(rloo_whiten_raw) if rloo_whiten_raw is not None else estimator_defaults.rloo_whiten,
            dapo_alpha=float(dapo_alpha_raw) if dapo_alpha_raw is not None else estimator_defaults.dapo_alpha,
            gae_gamma=float(gae_gamma_raw) if gae_gamma_raw is not None else estimator_defaults.gae_gamma,
            gae_lambda=float(gae_lambda_raw) if gae_lambda_raw is not None else estimator_defaults.gae_lambda,
            gae_normalize=bool(gae_normalize_raw)
            if gae_normalize_raw is not None
            else estimator_defaults.gae_normalize,
        )

        update_raw = algo_raw.get("update")
        update_data: dict[str, Any] = {}
        update_name_raw = None
        if isinstance(update_raw, str):
            update_name_raw = update_raw
        elif isinstance(update_raw, dict):
            update_data = update_raw
            update_name_raw = update_raw.get("name")
        elif update_raw is not None:
            raise ValueError("algo.update must be a dict or string when provided")

        if update_name_raw is None:
            update_name_raw = algo_raw.get("update_name")

        def _update_value(key: str, legacy_key: str | None = None) -> Any:
            if key in update_data:
                return update_data[key]
            return algo_raw.get(legacy_key or key)

        value_coef_raw = _update_value("value_coef", "ppo_value_coef")
        value_clip_range_raw = _update_value("value_clip_range", "ppo_value_clip_range")
        entropy_coef_raw = _update_value("entropy_coef", "ppo_entropy_coef")

        value_clip_range = float(value_clip_range_raw) if value_clip_range_raw is not None else None
        update_cfg = UpdateConfig(
            name=str(update_name_raw) if update_name_raw is not None else update_defaults.name,
            value_coef=float(value_coef_raw) if value_coef_raw is not None else update_defaults.value_coef,
            value_clip_range=value_clip_range,
            entropy_coef=float(entropy_coef_raw) if entropy_coef_raw is not None else update_defaults.entropy_coef,
        )

        algo_cfg = AlgoConfig(
            name=str(name_raw) if name_raw is not None else defaults.name,
            estimator=estimator_cfg,
            update=update_cfg,
        )
    else:
        raise ValueError(f"algo must be a dict or string, got {type(algo_raw).__name__}")

    eval_every_steps = int(cfg.get("eval_every_steps") or 0)
    eval_batches_per_process = _get_int_from_aliases(
        cfg,
        label="eval_batches_per_process",
        paths=["eval_batches_per_process", "eval_batches"],
        keys=[],
    )
    eval_batches_per_process = int(eval_batches_per_process or 1)
    eval_split = str(cfg.get("eval_split") or "test")

    train_steps_per_epoch_raw = cfg.get("train_steps_per_epoch")
    train_steps_per_epoch = int(train_steps_per_epoch_raw) if train_steps_per_epoch_raw is not None else 0

    eval_full_every_epochs_raw = cfg.get("eval_full_every_epochs")
    eval_full_every_epochs = int(eval_full_every_epochs_raw) if eval_full_every_epochs_raw is not None else 0

    eval_full_num_pre_q_raw = cfg.get("eval_full_num_pre_q")
    eval_full_num_pre_q = int(eval_full_num_pre_q_raw) if eval_full_num_pre_q_raw is not None else 1

    eval_full_greedy_raw = cfg.get("eval_full_greedy")
    eval_full_greedy = bool(eval_full_greedy_raw) if eval_full_greedy_raw is not None else False

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
        ),
        train=GRPOTrainConfig(
            micro_batch_size_per_device=train_micro_batch_size_per_device,
            micro_batch_size=train_micro_batch_size,
            max_length_total=max_length_total,
            ppo_epochs=ppo_epochs,
            grad_accum_steps=grad_accum_steps,
            beta=beta,
            policy_loss_impl=policy_loss_impl,
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
        train_steps_per_epoch=train_steps_per_epoch,
        eval_full_every_epochs=eval_full_every_epochs,
        eval_full_num_pre_q=eval_full_num_pre_q,
        eval_full_greedy=eval_full_greedy,
    )


def main() -> None:
    parser = ArgumentParser(description="Run GRPO/GSM8K training via plugins/training runner.")
    parser.add_argument(
        "--config",
        default="plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml",
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
    if args.print_config:
        rollout = cfg_dict.get("rollout")
        if not isinstance(rollout, dict):
            rollout = {}
            cfg_dict["rollout"] = rollout
        batch_size = int(rollout.get("batch_size") or 0)
        n = int(rollout.get("n") or 0)
        rollout["sequences_global_per_step"] = batch_size * n
        print(yaml.safe_dump(cfg_dict, sort_keys=False))
        return

    cfg = _cfg_from_dict(cfg_dict, config_path=config_path or "<default>")
    cfg_out = asdict(cfg)
    cfg_out["rollout"]["sequences_global_per_step"] = int(cfg.rollout.batch_size) * int(cfg.rollout.n)
    print(yaml.safe_dump(cfg_out, sort_keys=False))

    from plugins.training.runner import run_grpo_gsm8k
    run_grpo_gsm8k(cfg)


if __name__ == "__main__":
    main()
