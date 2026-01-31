from __future__ import annotations

from typing import Any

from plugins.training.core.config.loader import load_config as _load_config


DEFAULT_CONFIG: dict[str, Any] = {
    "base_model": "Qwen/Qwen2.5-1.5B",
    "output_dir": "runs/minionerec_rl",
    "seed": 42,
    "device": "tpu",
    # Optional: initialize params from a msgpack checkpoint (SFT or RL).
    "resume_from_checkpoint": None,
    "data": {
        "category": "Industrial_and_Scientific",
        "train_file": "workdir/MiniOneRec/data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv",
        "eval_file": "workdir/MiniOneRec/data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv",
        "test_file": "workdir/MiniOneRec/data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv",
        "info_file": "workdir/MiniOneRec/data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt",
        "sid_index_path": "workdir/MiniOneRec/data/Amazon/index/Industrial_and_Scientific.index.json",
        "max_len": 512,
        "sample_train": -1,
        "sample_eval": -1,
        "sample_test": -1,
    },
    "jax": {
        "mesh_shape": "auto",
        "param_dtype": "bfloat16",
        "compute_dtype": "bfloat16",
        "max_cache_length": 512,
    },
    "rollout": {
        "prompt_batch_size": 16,
        "num_generations": 16,
        "prompt_pad_len": 256,
        "global_length": 512,
    },
    "train": {
        "num_train_epochs": 0.0,
        "max_steps": 100,
        "grad_accum_steps": 1,
        "ppo_steps": 1,
        "beta": 1e-3,
        "logging_steps": 10,
        "save_last": True,
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
    },
    "eval": {
        "enabled": True,
        "every_steps": 0,
        "batch_size": 4,
        "num_beams": 50,
        "topk": [1, 3, 5, 10, 20, 50],
        "save_predictions_json": True,
    },
    "wandb": {
        "project": "minionerec-sid-rl",
        "mode": "online",
        "name": None,
    },
}


def load_config(config_path: str | None, overrides: list[str] | None = None) -> dict[str, Any]:
    return _load_config(DEFAULT_CONFIG, config_path, overrides=overrides)


__all__ = ["DEFAULT_CONFIG", "load_config"]

