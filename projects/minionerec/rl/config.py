from __future__ import annotations

from typing import Any

from plugins.common.config_loader import load_config as _load_config


DEFAULT_CONFIG: dict[str, Any] = {
    "base_model": "Qwen/Qwen2.5-1.5B",
    "output_dir": "runs/minionerec_rl",
    "seed": 42,
    # Execution backend:
    # - "tpu": JAX/Flax on TPU
    # - "jax": alias for "tpu"
    "device": "tpu",
    "jax": {
        # Mesh dims for ("dp", "fsdp", "tp"). "auto" uses dp across processes and fsdp within host.
        "mesh_shape": "auto",
        # Dtypes: "float32" | "bfloat16" | "float16"
        "param_dtype": "bfloat16",
        "compute_dtype": "bfloat16",
        "max_cache_length": 512,
    },
    "data": {
        "category": "Industrial_and_Scientific",
        "train_file": "workdir/MiniOneRec/data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv",
        "eval_file": "workdir/MiniOneRec/data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv",
        "test_file": "workdir/MiniOneRec/data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv",
        "info_file": "workdir/MiniOneRec/data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt",
        "sid_index_path": "workdir/MiniOneRec/data/Amazon/index/Industrial_and_Scientific.index.json",
        "item_meta_path": "workdir/MiniOneRec/data/Amazon/index/Industrial_and_Scientific.item.json",
        "max_len": 512,
        "sample_train": -1,
        "sample_eval": -1,
        "sample_test": -1,
        # Upstream defaults: seq-title RL uses a 10k subset; alignment uses full.
        "sample_seq_title": 10000,
        "sample_item_alignment": -1,
    },
    "tasks": {
        "sid_next_item": True,
        "title2sid": True,
        "description2sid": True,
        "seq_title2sid": True,
    },
    "rollout": {
        # Prompts per update (global).
        "prompt_batch_size": 32,
        "num_generations": 16,
        # Fixed training padding (no length bucketing).
        "prompt_pad_len": 256,
        "global_length": 512,
    },
    "train": {
        "num_train_epochs": 1.0,
        "max_steps": -1,
        "grad_accum_steps": 2,
        "ppo_steps": 1,
        # Upstream rl.py default: beta=0.04.
        "beta": 0.04,
        "logging_steps": 10,
        "save_last": True,
        "optimizer": {
            "name": "adamw",
            # Upstream rl.py: max_grad_norm=0.3.
            "clip_norm": 0.3,
            "weight_decay": 0.0,
            "lr_schedule": {
                "type": "warmup_cosine",
                "init_value": 0.0,
                "peak_value": 1e-6,
                "end_value": 0.0,
                "warmup_ratio": 0.03,
                "warmup_steps": None,
            },
        },
    },
    "eval": {
        "enabled": True,
        # 0 disables in-training eval; eval is typically run separately via --run-mode eval.
        "every_steps": 0,
        "batch_size": 1,
        "num_beams": 50,
        "topk": [1, 3, 5, 10, 20, 50],
        "save_predictions_json": True,
    },
    "wandb": {
        "project": "minionerec-sid-rl",
        "mode": "online",
        "name": None,
    },
    # If set, loads params from a `sft_state_*.msgpack` checkpoint for train and/or eval.
    "resume_from_checkpoint": None,
}


def load_config(config_path: str | None, overrides: list[str] | None = None) -> dict[str, Any]:
    return _load_config(DEFAULT_CONFIG, config_path, overrides=overrides)


__all__ = ["DEFAULT_CONFIG", "load_config"]

