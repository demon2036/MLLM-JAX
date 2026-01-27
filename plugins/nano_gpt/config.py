from __future__ import annotations

from typing import Any

from plugins.common.config_loader import load_config as _load_config


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 1337,
    "output_dir": "runs/nano_gpt",
    "data": {
        "name": "tinyshakespeare_char",
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "cache_dir": "workdir/nano_gpt_data",
        "train_ratio": 0.9,
    },
    "model": {
        "block_size": 256,
        "n_layer": 6,
        "n_head": 6,
        "n_embd": 384,
        "dropout": 0.2,
        "bias": True,
        "param_dtype": "float32",
        "compute_dtype": "float32",
    },
    "train": {
        "max_steps": 200,
        "global_batch_size": 64,
        "learning_rate": 3e-4,
        "min_lr": 3e-5,
        "warmup_steps": 20,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.95,
        "grad_clip_norm": 1.0,
        "log_every": 10,
        "eval_every": 50,
        "eval_iters": 20,
        "sample_every": 50,
        "sample_tokens": 256,
        "temperature": 1.0,
        "top_k": 50,
        "ckpt_every": 100,
        "keep_ckpts": 2,
    },
    "wandb": {
        "project": "nano-gpt-jax",
        "mode": "disabled",
        "name": None,
    },
}


def load_config(config_path: str | None, overrides: list[str] | None = None) -> dict[str, Any]:
    return _load_config(DEFAULT_CONFIG, config_path, overrides=overrides)


__all__ = ["DEFAULT_CONFIG", "load_config"]

