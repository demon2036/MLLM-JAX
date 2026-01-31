from __future__ import annotations

from typing import Any

from plugins.training.core.config.loader import load_config as _load_config


# Default config aims to mirror upstream `MiniOneRec/rl.sh` and `rl.py` defaults
# (GRPO, ranking reward, 2 epochs, beta=1e-3, lr=1e-5).
DEFAULT_CONFIG: dict[str, Any] = {
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "output_dir": "runs/minionerec_sid_rl",
    "seed": 42,
    "device": "tpu",
    "jax": {
        "mesh_shape": "auto",
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
        # Official RL runs from an SFT model path; in this repo we load a JAX
        # msgpack checkpoint produced by `projects/sid_sft` (sft_state_*.msgpack).
        "init_from_sft_checkpoint": None,
    },
    "tasks": {
        # Upstream RL mixes multiple datasets; keep the same defaults.
        "next_item": True,
        "title2sid": True,
        "description2sid": True,
        "seqtitle2sid": True,
        # Upstream `rl.py` hardcodes RLSeqTitle2SidDataset(sample=10000).
        "seqtitle2sid_sample": 10000,
    },
    "rollout": {
        # Upstream: `train_batch_size` (prompts) and `num_generations=16`.
        "prompt_batch_size": 64,
        "num_generations": 16,
        # Fixed prompt padding (no length bucketing).
        "prompt_pad_len": 256,
        # Fixed training padding.
        "global_length": 512,
    },
    "train": {
        "num_train_epochs": 2.0,
        "max_steps": -1,
        "grad_accum_steps": 2,
        "beta": 1e-3,
        "logging_steps": 10,
        "save_last": True,
        "ema": {
            "enabled": False,
            "decay": 0.9998,
            "use_for_eval": True,
        },
        "optimizer": {
            "name": "adamw",
            "clip_norm": 0.3,
            "weight_decay": 1e-8,
            "lr_schedule": {
                "type": "warmup_cosine",
                "init_value": 0.0,
                "peak_value": 1e-5,
                "end_value": 0.0,
                "warmup_ratio": 0.03,
                "warmup_steps": None,
            },
        },
    },
    "eval": {
        "enabled": True,
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
}


def load_config(config_path: str | None, overrides: list[str] | None = None) -> dict[str, Any]:
    return _load_config(DEFAULT_CONFIG, config_path, overrides=overrides)


__all__ = ["DEFAULT_CONFIG", "load_config"]

