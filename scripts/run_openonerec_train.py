from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser
from dataclasses import asdict
from typing import Any

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from plugins.training.core.runtime.env import load_dotenv_if_present
from projects.openonerec_train.config import DEFAULT_CONFIG, load_config
from projects.openonerec_train.runner import (
    OpenOneRecTrainConfig,
    OpenOneRecTrainDataConfig,
    OpenOneRecTrainJaxConfig,
    OpenOneRecTrainTrainConfig,
    OpenOneRecTrainWandbConfig,
    run_openonerec_train,
)


def _get_by_path(cfg: dict[str, Any], key_path: str) -> Any:
    cur: Any = cfg
    for key in [k for k in key_path.split(".") if k]:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _get_or_default(cfg: dict[str, Any], key_path: str, default: Any) -> Any:
    value = _get_by_path(cfg, key_path)
    return default if value is None else value


def _normalize_sample_size(value: Any) -> int | str | None:
    if value is None:
        return None
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"", "none", "null"}:
            return None
        if norm == "full":
            return "full"
        return int(value)
    return int(value)


def _cfg_from_dict(cfg: dict[str, Any], *, config_path: str) -> OpenOneRecTrainConfig:
    return OpenOneRecTrainConfig(
        config_path=config_path,
        openonerec_root=str(_get_or_default(cfg, "openonerec_root", DEFAULT_CONFIG["openonerec_root"])),
        base_model=str(_get_or_default(cfg, "base_model", DEFAULT_CONFIG["base_model"])),
        output_dir=str(_get_or_default(cfg, "output_dir", DEFAULT_CONFIG["output_dir"])),
        seed=int(_get_or_default(cfg, "seed", DEFAULT_CONFIG["seed"])),
        data=OpenOneRecTrainDataConfig(
            benchmark_data_dir=str(_get_or_default(cfg, "data.benchmark_data_dir", DEFAULT_CONFIG["data"]["benchmark_data_dir"])),
            task_types=tuple(_get_or_default(cfg, "data.task_types", DEFAULT_CONFIG["data"]["task_types"])),
            split=str(_get_or_default(cfg, "data.split", DEFAULT_CONFIG["data"]["split"])),
            sample_size=_normalize_sample_size(_get_or_default(cfg, "data.sample_size", DEFAULT_CONFIG["data"]["sample_size"])),
            max_len=int(_get_or_default(cfg, "data.max_len", DEFAULT_CONFIG["data"]["max_len"])),
        ),
        jax=OpenOneRecTrainJaxConfig(
            mesh_shape=str(_get_or_default(cfg, "jax.mesh_shape", DEFAULT_CONFIG["jax"]["mesh_shape"])),
            param_dtype=str(_get_or_default(cfg, "jax.param_dtype", DEFAULT_CONFIG["jax"]["param_dtype"])),
            compute_dtype=str(_get_or_default(cfg, "jax.compute_dtype", DEFAULT_CONFIG["jax"]["compute_dtype"])),
            max_cache_length=int(_get_or_default(cfg, "jax.max_cache_length", DEFAULT_CONFIG["jax"]["max_cache_length"])),
        ),
        train=OpenOneRecTrainTrainConfig(
            per_device_train_batch_size=int(
                _get_or_default(cfg, "train.per_device_train_batch_size", DEFAULT_CONFIG["train"]["per_device_train_batch_size"])
            ),
            gradient_accumulation_steps=int(
                _get_or_default(cfg, "train.gradient_accumulation_steps", DEFAULT_CONFIG["train"]["gradient_accumulation_steps"])
            ),
            max_steps=int(_get_or_default(cfg, "train.max_steps", DEFAULT_CONFIG["train"]["max_steps"])),
            learning_rate=float(_get_or_default(cfg, "train.learning_rate", DEFAULT_CONFIG["train"]["learning_rate"])),
            weight_decay=float(_get_or_default(cfg, "train.weight_decay", DEFAULT_CONFIG["train"]["weight_decay"])),
            optimizer=str(_get_or_default(cfg, "train.optimizer", DEFAULT_CONFIG["train"]["optimizer"])),
            logging_steps=int(_get_or_default(cfg, "train.logging_steps", DEFAULT_CONFIG["train"]["logging_steps"])),
            warmup_steps=int(_get_or_default(cfg, "train.warmup_steps", DEFAULT_CONFIG["train"]["warmup_steps"])),
            shuffle=bool(_get_or_default(cfg, "train.shuffle", DEFAULT_CONFIG["train"]["shuffle"])),
            dataloader_drop_last=bool(
                _get_or_default(cfg, "train.dataloader_drop_last", DEFAULT_CONFIG["train"]["dataloader_drop_last"])
            ),
            padding_side=str(_get_or_default(cfg, "train.padding_side", DEFAULT_CONFIG["train"]["padding_side"])),
            save_last=bool(_get_or_default(cfg, "train.save_last", DEFAULT_CONFIG["train"]["save_last"])),
        ),
        wandb=OpenOneRecTrainWandbConfig(
            project=str(_get_or_default(cfg, "wandb.project", DEFAULT_CONFIG["wandb"]["project"])),
            mode=str(_get_or_default(cfg, "wandb.mode", DEFAULT_CONFIG["wandb"]["mode"])),
            name=_get_or_default(cfg, "wandb.name", DEFAULT_CONFIG["wandb"]["name"]),
        ),
    )


def main() -> None:
    load_dotenv_if_present(repo_root=REPO_ROOT)

    parser = ArgumentParser(description="Run OpenOneRec SFT training (projects/openonerec_train).")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    parser.add_argument("--run-mode", type=str, default="train", help="Only 'train' is supported.")
    parser.add_argument("--print-config", action="store_true", help="Print resolved config and exit.")
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    cfg = _cfg_from_dict(cfg_dict, config_path=str(args.config or "<default>"))

    if args.print_config:
        print(yaml.safe_dump(json.loads(json.dumps(asdict(cfg))), sort_keys=False))
        return

    run_mode = str(args.run_mode).strip().lower()
    if run_mode != "train":
        raise ValueError(f"Unsupported --run-mode={args.run_mode!r}; only 'train' is supported.")

    os.makedirs(cfg.output_dir, exist_ok=True)
    result = run_openonerec_train(cfg, run_mode=run_mode)

    summary_path = os.path.join(cfg.output_dir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"summary_json={summary_path}")


if __name__ == "__main__":
    main()
