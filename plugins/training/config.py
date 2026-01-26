from __future__ import annotations

from typing import Any

from plugins.common.config_loader import load_config as _load_config


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
    },
    "train": {
        # Optional: sequences per process per micro-step.
        "micro_batch_size": None,
        # Optional: sequences per device per micro-step.
        "micro_batch_size_per_device": 4,
        "ppo_epochs": 1,
        "beta": 0.0,
        # Optimizer (pluggable; defaults match `training2.get_state`).
        "optimizer": {
            "name": "lion",
            "clip_norm": 1.0,
            "weight_decay": 1e-8,
            "lr_schedule": {
                "type": "warmup_cosine",
                "init_value": 0.0,
                "peak_value": 1e-6,
                "end_value": 0.0,
                "warmup_ratio": 0.05,
                "warmup_steps": None,
            },
        },
        # Optional: enable the GRPO Pallas loss kernel (logits-level).
        "grpo_kernel": {
            "enabled": False,
            # Kernel tiling knobs.
            "kernel": {
                "block_size": 2048,
                "time_block": 8,
                # Backward implementation ("jax" is typically more fusible on TPU).
                "bwd_impl": "jax",
            },
            # shard_map wrapper knobs (only relevant for multi-device).
            "sharding": {
                "batch_axes": ["dp", "fsdp"],
                # Set to "tp" only if the vocab/logits dimension is sharded across tp.
                "vocab_axis": None,
            },
        },
    },
    # Algorithm (advantage estimator + update wiring).
    #
    # Note: `train.ppo_epochs` remains the knob for PPO-style multi-epoch updates.
    "algo": {
        "name": "grpo",
        "estimator": {
            "name": "grpo",
            "eps": 1e-4,
            "clip_range": None,
            "rloo_whiten": True,
            "dapo_alpha": 0.2,
            "gae_gamma": 1.0,
            "gae_lambda": 0.95,
            "gae_normalize": True,
        },
        "update": {
            "name": "policy_gradient",
            "value_coef": 0.5,
            "value_clip_range": 0.2,
            "entropy_coef": 0.0,
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
}


def load_config(config_path: str | None, overrides: list[str] | None = None) -> dict[str, Any]:
    return _load_config(DEFAULT_CONFIG, config_path, overrides=overrides)


__all__ = ["DEFAULT_CONFIG", "load_config"]
