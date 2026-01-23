from __future__ import annotations

import os
import sys
import time
from argparse import ArgumentParser
from dataclasses import asdict
from typing import Any

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from plugins.training.config import load_config
from plugins.training.runner import GRPOGsm8kConfig, GRPORolloutConfig, GRPOTrainConfig, run_grpo_gsm8k


def _load_dotenv_if_present() -> None:
    candidates = [
        os.path.join(REPO_ROOT, ".env"),
        "/root/.env",
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                existing = os.environ.get(k)
                if existing is None or str(existing).strip() == "":
                    os.environ[k] = v
        return


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


def _maybe_override_from_env(cfg: dict[str, Any], *, env: str, key_path: str, cast) -> None:
    value = os.environ.get(env)
    if value is None:
        return
    _set_by_path(cfg, key_path, cast(value))


def _apply_env_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    """Allow existing env-var based runs to keep working.

    The TPU SOPs in this repo historically exported env vars like `STEPS=20`.
    YAML configs are now supported, but env vars remain optional overrides.
    """
    cfg = dict(cfg)
    _maybe_override_from_env(cfg, env="MODEL_PATH", key_path="model_path", cast=str)
    _maybe_override_from_env(cfg, env="STEPS", key_path="steps", cast=int)

    # Rollout (generation)
    _maybe_override_from_env(cfg, env="ROLLOUT_BACKEND", key_path="rollout.backend", cast=str)
    # Prompt batch size per training step (global, across all processes).
    _maybe_override_from_env(cfg, env="ROLLOUT_BATCH_SIZE", key_path="rollout.batch_size", cast=int)
    _maybe_override_from_env(cfg, env="BATCH_SIZE", key_path="rollout.batch_size", cast=int)
    _maybe_override_from_env(cfg, env="ROLLOUT_PROMPT_BATCH_SIZE", key_path="rollout.batch_size", cast=int)

    _maybe_override_from_env(cfg, env="ROLLOUT_N", key_path="rollout.n", cast=int)
    _maybe_override_from_env(cfg, env="NUM_PRE_Q", key_path="rollout.n", cast=int)
    _maybe_override_from_env(cfg, env="GLOBAL_LENGTH", key_path="rollout.global_length", cast=int)
    _maybe_override_from_env(cfg, env="MAX_LENGTH_SAMPLE", key_path="rollout.max_length_sample", cast=int)

    # Train (update)
    _maybe_override_from_env(cfg, env="TRAIN_GLOBAL_MICRO_BATCH_SIZE", key_path="train.micro_batch_size", cast=int)
    _maybe_override_from_env(cfg, env="TRAIN_MICRO_BATCH_SIZE_PER_PROCESS", key_path="train.micro_batch_size", cast=int)
    _maybe_override_from_env(cfg, env="TRAIN_MICRO_BATCH_SIZE", key_path="train.micro_batch_size", cast=int)
    _maybe_override_from_env(
        cfg, env="TRAIN_MICRO_BATCH_SIZE_PER_DEVICE", key_path="train.micro_batch_size_per_device", cast=int
    )
    # Backward-compatible alias.
    _maybe_override_from_env(
        cfg, env="TRAIN_PER_DEVICE_MICRO_BATCH_SIZE", key_path="train.micro_batch_size_per_device", cast=int
    )
    _maybe_override_from_env(cfg, env="MAX_LENGTH_TOTAL", key_path="train.max_length_total", cast=int)
    _maybe_override_from_env(cfg, env="PPO_EPOCHS", key_path="train.ppo_epochs", cast=int)
    _maybe_override_from_env(cfg, env="GRAD_ACCUM_STEPS", key_path="train.grad_accum_steps", cast=int)
    _maybe_override_from_env(cfg, env="BETA", key_path="train.beta", cast=float)

    # Infra / logging
    _maybe_override_from_env(cfg, env="MESH_SHAPE_FSDP", key_path="mesh_shape", cast=str)
    _maybe_override_from_env(cfg, env="WANDB_PROJECT", key_path="wandb_project", cast=str)
    _maybe_override_from_env(cfg, env="WANDB_NAME", key_path="wandb_name", cast=str)
    _maybe_override_from_env(cfg, env="EVAL_EVERY_STEPS", key_path="eval_every_steps", cast=int)
    _maybe_override_from_env(cfg, env="EVAL_BATCHES", key_path="eval_batches_per_process", cast=int)
    _maybe_override_from_env(cfg, env="EVAL_SPLIT", key_path="eval_split", cast=str)
    return cfg


def _cfg_from_dict(cfg: dict[str, Any]) -> GRPOGsm8kConfig:
    model_path = str(cfg.get("model_path") or "Qwen/Qwen2.5-7B-Instruct")
    steps = int(cfg.get("steps") or 20)

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

    wandb_project = str(cfg.get("wandb_project") or "mllm-jax-grpo-gsm8k")
    wandb_name = cfg.get("wandb_name")
    if wandb_name is None or str(wandb_name).strip() == "":
        wandb_name = f"grpo_gsm8k_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_steps{steps}"
    wandb_name = str(wandb_name)

    reward_weights_raw = cfg.get("reward_weights") or (1.0, 0.5, 0.5)
    if isinstance(reward_weights_raw, (list, tuple)) and len(reward_weights_raw) == 3:
        reward_weights = tuple(float(x) for x in reward_weights_raw)
    else:
        raise ValueError("reward_weights must be a list/tuple of 3 floats")

    eval_every_steps = int(cfg.get("eval_every_steps") or 0)
    eval_batches_per_process = _get_int_from_aliases(
        cfg,
        label="eval_batches_per_process",
        paths=["eval_batches_per_process", "eval_batches"],
        keys=[],
    )
    eval_batches_per_process = int(eval_batches_per_process or 1)
    eval_split = str(cfg.get("eval_split") or "test")

    return GRPOGsm8kConfig(
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
        ),
        mesh_shape=mesh_shape,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        reward_weights=reward_weights,
        eval_every_steps=eval_every_steps,
        eval_batches_per_process=eval_batches_per_process,
        eval_split=eval_split,
    )


def main() -> None:
    parser = ArgumentParser(description="Run GRPO/GSM8K training via plugins/training runner.")
    parser.add_argument(
        "--config",
        default="plugins/training/configs/grpo_gsm8k_default.yaml",
        help="YAML config path (optional).",
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

    _load_dotenv_if_present()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if os.environ.get("WANDB_MODE") is None and os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_MODE"] = "online"

    cfg_dict = load_config(args.config if args.config else None, args.set)
    cfg_dict = _apply_env_overrides(cfg_dict)
    cfg = _cfg_from_dict(cfg_dict)

    cfg_out = asdict(cfg)
    cfg_out["rollout"]["sequences_global_per_step"] = int(cfg.rollout.batch_size) * int(cfg.rollout.n)
    print(yaml.safe_dump(cfg_out, sort_keys=False))
    if args.print_config:
        return
    run_grpo_gsm8k(cfg)


if __name__ == "__main__":
    main()
