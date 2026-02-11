from __future__ import annotations

from typing import Any

from plugins.training.core.config.loader import load_config as _load_config


DEFAULT_CONFIG: dict[str, Any] = {
    # Model / loop
    "model_path": "Qwen/Qwen2.5-3B-Instruct",
    "steps": 100,
    # Rollout (generation) vs Train (update) are separated, AReaL-style.
    "rollout": {
        # Rollout backend selector (swappable generation engine).
        # - "naive": in-process sampler (current default)
        # - Future: other engines (e.g. vllm)
        "backend": "naive",
        # Prompt batch size per training step (global, across all processes).
        #
        # Each prompt is expanded to `n` sampled completions, so the global
        # sequence batch is: `batch_size * n`.
        "batch_size": 16,
        # Number of samples per prompt (GRPO group size, a.k.a. K / num_pre_q).
        "n": 8,
        "global_length": 512,
        "max_length_sample": 1024,
        # Explicit rollout micro-batch controls (sequences per rollout pass).
        # `projects/gsm8k_grpo` requires `micro_batch_size_per_device` to be set
        # in YAML (no hidden rollout auto-split heuristics).
        "micro_batch_size": None,
        "micro_batch_size_per_device": None,
        "dynamic_sampling": {
            "enabled": False,
            "trigger": "homogeneous_group",
            "metric": "acc",
            "homogeneity_threshold": 1.0,
            "min_unique_reward_values": 2,
            "max_extra_roll_rounds": 10,
            "target_valid_groups": None,
            "fallback_policy": "keep_last",
        },
    },
    "train": {
        # Optional: sequences per process per micro-step.
        "micro_batch_size": None,
        # Optional: sequences per device per micro-step.
        "micro_batch_size_per_device": 4,
        "ppo_epochs": 1,
        "beta": 0.0,
        "gradient_checkpointing": True,
        # Optimizer (pluggable; defaults match `training2.get_state`).
        "optimizer": {
            "name": "lion",
            "kwargs": {
                "clip_norm": 1.0,
                "weight_decay": 1e-8,
            },
            "lr_schedule": {
                "name": "warmup_cosine",
                "kwargs": {
                    "init_value": 0.0,
                    "peak_value": 1e-6,
                    "end_value": 0.0,
                    "warmup_ratio": 0.05,
                    "warmup_steps": None,
                },
            },
        },
    },
    # Algorithm (advantage estimator + update wiring).
    #
    # Note: `train.ppo_epochs` remains the knob for PPO-style multi-epoch updates.
    "algo": {
        "estimator": {
            "name": "grpo",
            "kwargs": {
                "eps": 1e-4,
                "clip_range": None,
            },
        },
        "update": {
            "name": "policy_gradient",
            "kwargs": {},
        },
    },
    # Mesh
    "mesh_shape": "auto",
    # Logging
    "wandb_project": "mllm-jax-grpo-gsm8k",
    "wandb_mode": "online",
    "wandb_name": None,
    # Rewards
    "reward_weights": [1.0, 0.5, 0.5],
    # Eval (optional)
    # Run a lightweight eval rollout+reward every N steps (0 disables).
    "eval_every_steps": 10,
    "eval_batches_per_process": 1,
    "eval_split": "test",
    # Eval rollout count per prompt (kept independent from train rollout.n).
    "eval_rollout_n": 1,
    # Whether to run a full-split eval sweep once at the end.
    "eval_full_sweep": False,
}


def load_config(config_path: str | None, overrides: list[str] | None = None) -> dict[str, Any]:
    return _load_config(DEFAULT_CONFIG, config_path, overrides=overrides)


__all__ = ["DEFAULT_CONFIG", "load_config"]
