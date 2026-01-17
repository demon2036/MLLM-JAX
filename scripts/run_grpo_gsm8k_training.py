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
from plugins.training.runner import GRPOGsm8kConfig, run_grpo_gsm8k


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
                os.environ.setdefault(k, v)
        return


def _maybe_override_from_env(cfg: dict[str, Any], *, env: str, key: str, cast) -> None:
    value = os.environ.get(env)
    if value is None:
        return
    cfg[key] = cast(value)


def _apply_env_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    """Allow existing env-var based runs to keep working.

    The TPU SOPs in this repo historically exported env vars like `STEPS=20`.
    YAML configs are now supported, but env vars remain optional overrides.
    """
    cfg = dict(cfg)
    _maybe_override_from_env(cfg, env="MODEL_PATH", key="model_path", cast=str)
    _maybe_override_from_env(cfg, env="STEPS", key="steps", cast=int)
    _maybe_override_from_env(cfg, env="BATCH_SIZE", key="batch_size", cast=int)
    _maybe_override_from_env(cfg, env="NUM_PRE_Q", key="num_pre_q", cast=int)
    _maybe_override_from_env(cfg, env="GLOBAL_LENGTH", key="global_length", cast=int)
    _maybe_override_from_env(cfg, env="MAX_LENGTH_SAMPLE", key="max_length_sample", cast=int)
    _maybe_override_from_env(cfg, env="MAX_LENGTH_TOTAL", key="max_length_total", cast=int)
    _maybe_override_from_env(cfg, env="PPO_EPOCHS", key="ppo_epochs", cast=int)
    _maybe_override_from_env(cfg, env="GRAD_ACCUM_STEPS", key="grad_accum_steps", cast=int)
    _maybe_override_from_env(cfg, env="BETA", key="beta", cast=float)
    _maybe_override_from_env(cfg, env="MESH_SHAPE_FSDP", key="mesh_shape", cast=str)
    _maybe_override_from_env(cfg, env="WANDB_PROJECT", key="wandb_project", cast=str)
    _maybe_override_from_env(cfg, env="WANDB_NAME", key="wandb_name", cast=str)
    _maybe_override_from_env(cfg, env="EVAL_EVERY_STEPS", key="eval_every_steps", cast=int)
    _maybe_override_from_env(cfg, env="EVAL_BATCHES", key="eval_batches", cast=int)
    _maybe_override_from_env(cfg, env="EVAL_SPLIT", key="eval_split", cast=str)
    return cfg


def _cfg_from_dict(cfg: dict[str, Any]) -> GRPOGsm8kConfig:
    model_path = str(cfg.get("model_path") or "Qwen/Qwen2.5-7B-Instruct")
    steps = int(cfg.get("steps") or 20)
    batch_size = int(cfg.get("batch_size") or 1)
    num_pre_q = int(cfg.get("num_pre_q") or 8)
    global_length = int(cfg.get("global_length") or 512)
    max_length_sample = int(cfg.get("max_length_sample") or 64)
    max_length_total_raw = cfg.get("max_length_total")
    max_length_total = int(max_length_total_raw) if max_length_total_raw is not None else max_length_sample + 128
    ppo_epochs = int(cfg.get("ppo_epochs") or 1)
    grad_accum_steps = int(cfg.get("grad_accum_steps") or 1)
    beta = float(cfg.get("beta") or 0.0)
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
    eval_batches = int(cfg.get("eval_batches") or 1)
    eval_split = str(cfg.get("eval_split") or "test")

    return GRPOGsm8kConfig(
        model_path=model_path,
        steps=steps,
        batch_size=batch_size,
        num_pre_q=num_pre_q,
        global_length=global_length,
        max_length_sample=max_length_sample,
        max_length_total=max_length_total,
        ppo_epochs=ppo_epochs,
        grad_accum_steps=grad_accum_steps,
        beta=beta,
        mesh_shape=mesh_shape,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        reward_weights=reward_weights,
        eval_every_steps=eval_every_steps,
        eval_batches=eval_batches,
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

    print(yaml.safe_dump(asdict(cfg), sort_keys=False))
    if args.print_config:
        return
    run_grpo_gsm8k(cfg)


if __name__ == "__main__":
    main()
