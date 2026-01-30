from __future__ import annotations

from typing import Any

from plugins.training.core.config.loader import load_config as _load_config


DEFAULT_CONFIG: dict[str, Any] = {
    # Execution backend:
    # - "jax": JAX/Flax training + eval (TPU-friendly)
    # - "tpu": alias for "jax"
    "backend": "jax",
    "base_model": "Qwen/Qwen2.5-0.5B",
    "output_dir": "runs/sid_sft",
    "seed": 42,
    "jax": {
        # Mesh dims for ("dp", "fsdp", "tp"). Accepts either "1,-1,1" or named dims like "dp:1,fsdp:-1,tp:1".
        "mesh_shape": "1,-1,1",
        # Dtypes: "float32" | "bfloat16" | "float16"
        "param_dtype": "float32",
        "compute_dtype": "bfloat16",
        # Cache length used for constrained generation eval (prompt_len + max_new_tokens must fit).
        "max_cache_length": 2048,
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
    },
    "tasks": {
        "sid_next_item": True,
        "sid_item_alignment": True,
        "fusion_seq_rec": True,
    },
    "train": {
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        # If > 0, overrides `gradient_accumulation_steps` to match this global
        # effective batch size: micro * (dp*fsdp) * grad_accum.
        "global_batch_size": 0,
        "gradient_accumulation_steps": 1,
        "learning_rate": 3e-4,
        # Optimizer: "adamw" | "lion" | "muon"
        "optimizer": "adamw",
        "muon": {
            # Auxiliary AdamW LR for non-matrix params; Muon LR is `train.learning_rate`.
            "aux_learning_rate": 3e-4,
            "momentum": 0.95,
            "nesterov": True,
            "ns_steps": 5,
            "eps": 1e-7,
            "max_dim": 10_000,
        },
        "ema": {
            # If enabled, maintain an exponential moving average of params and
            # (by default) use it for eval.
            "enabled": False,
            "decay": 0.9998,
            "use_for_eval": True,
        },
        "weight_decay": 0.0,
        "num_train_epochs": 1,
        # If > 0, overrides num_train_epochs.
        "max_steps": -1,
        "warmup_steps": 0,
        "logging_steps": 10,
        "eval_steps": 200,
        "save_steps": 200,
        "save_total_limit": 1,
        # If false, the JAX backend won't write `sft_state_last.msgpack`.
        "save_last": True,
        "group_by_length": False,
        "freeze_LLM": False,
        "train_from_scratch": False,
        "resume_from_checkpoint": None,
        "early_stopping_patience": 3,
        # Dtype flags (Trainer will pick the appropriate one).
        "bf16": False,
        "fp16": False,
    },
    "eval": {
        "enabled": True,
        "batch_size": 4,
        "num_beams": 50,
        "max_new_tokens": 64,
        "length_penalty": 0.0,
        "topk": [1, 3, 5, 10, 20, 50],
        "constrained": True,
        "save_predictions_json": True,
        # Constrained decoding prefill strategy:
        # - "bucket": sampler-style prefill buckets (default; can compile per bucket)
        # - "fixed": single fixed prefill length across the whole dataset (compile once)
        "prefill_mode": "bucket",
        # Only used when prefill_mode="fixed". If null, derive from dataset.
        "fixed_prefill_len": None,
    },
    "wandb": {
        "project": "minionerec-sid-sft",
        "mode": "online",
        "name": None,
    },
}


def load_config(config_path: str | None, overrides: list[str] | None = None) -> dict[str, Any]:
    return _load_config(DEFAULT_CONFIG, config_path, overrides=overrides)


__all__ = ["DEFAULT_CONFIG", "load_config"]
