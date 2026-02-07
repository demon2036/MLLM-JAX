from __future__ import annotations

from typing import Any

from plugins.training.core.config.loader import load_config as _load_config


DEFAULT_CONFIG: dict[str, Any] = {
    "openonerec_root": "workdir/OpenOneRec",
    "base_model": "OpenOneRec/OneRec-1.7B",
    "output_dir": "runs/openonerec_train",
    "seed": 42,
    "data": {
        "benchmark_data_dir": "workdir/OpenOneRec/benchmarks/data",
        "task_types": ["video", "ad", "product", "label_cond", "interactive"],
        "split": "test",
        "sample_size": 32,
        "max_len": 512,
    },
    "jax": {
        "mesh_shape": "1,-1,1",
        "param_dtype": "float32",
        "compute_dtype": "bfloat16",
        "max_cache_length": 2048,
    },
    "train": {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_steps": 20,
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "optimizer": "adamw",
        "logging_steps": 10,
        "warmup_steps": 0,
        "shuffle": True,
        "dataloader_drop_last": True,
        "padding_side": "right",
        "save_last": True,
    },
    "wandb": {
        "project": "openonerec-train",
        "mode": "online",
        "name": None,
    },
}


def load_config(config_path: str | None, overrides: list[str] | None = None) -> dict[str, Any]:
    return _load_config(DEFAULT_CONFIG, config_path, overrides=overrides)


__all__ = ["DEFAULT_CONFIG", "load_config"]
