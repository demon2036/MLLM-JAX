from __future__ import annotations

from typing import Any

from plugins.training.core.config.loader import load_config as _load_config


DEFAULT_CONFIG: dict[str, Any] = {
    "openonerec_root": "workdir/OpenOneRec",
    "base_model": "OpenOneRec/OneRec-1.7B",
    "output_dir": "runs/openonerec_eval",
    "seed": 42,
    "eval": {
        "task_types": ["video", "ad", "product", "label_cond", "interactive"],
        "splits": ["test"],
        "sample_size": 16,
        "overwrite": False,
        "enable_thinking": False,
        "run_official_evaluator": True,
        # Optional evaluator override for deterministic smoke fixtures.
        # Keep None to preserve official evaluator defaults.
        "evaluation_mode": None,
    },
    "generation": {
        # "jax": run JAX model generation
        # "replay": read generations from replay JSON
        "mode": "jax",
        "batch_size": 4,
        "num_beams": 16,
        "num_return_sequences": 32,
        "max_new_tokens": 3,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 50,
        "prompt_token": "<|sid_begin|>",
        # Optional msgpack checkpoint path produced by openonerec_train
        # (`plugins/training/core/checkpoint/msgpack.py`).
        # When set, eval loads checkpoint params instead of HF base weights.
        "params_checkpoint_path": None,
        # Optional template for replay inputs, e.g.:
        # memory/.../smoke_replay/{task}_{split}_generated.json
        "replay_path_template": None,
    },
    "jax": {
        "mesh_shape": "1,-1,1",
        "max_cache_length": 512,
        "param_dtype": "float32",
    },
    "data": {
        "benchmark_data_dir": "workdir/OpenOneRec/benchmarks/data",
    },
    "wandb": {
        "project": "openonerec-eval",
        "mode": "disabled",
        "name": None,
    },
}


def load_config(config_path: str | None, overrides: list[str] | None = None) -> dict[str, Any]:
    return _load_config(DEFAULT_CONFIG, config_path, overrides=overrides)


__all__ = ["DEFAULT_CONFIG", "load_config"]
