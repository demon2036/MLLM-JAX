from __future__ import annotations

import json
import os
import subprocess
import sys
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime
from typing import Any

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from plugins.common.env import load_dotenv_if_present
from plugins.sft.config import DEFAULT_CONFIG, load_config
from plugins.sft.runner.sid_sft import (
    SidSftConfig,
    SidSftDataConfig,
    SidSftEvalConfig,
    SidSftJaxConfig,
    SidSftTasksConfig,
    SidSftTrainConfig,
    SidSftWandbConfig,
    run_sid_sft,
)


def _maybe_git_short_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None
    return out or None


def _get_by_path(cfg: dict[str, Any], key_path: str) -> Any:
    keys = [k for k in key_path.split(".") if k]
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _get_or_default(cfg: dict[str, Any], key_path: str, default: Any) -> Any:
    value = _get_by_path(cfg, key_path)
    return default if value is None else value


def _cfg_from_dict(cfg: dict[str, Any], *, config_path: str) -> SidSftConfig:
    backend = str(cfg.get("backend") if cfg.get("backend") is not None else DEFAULT_CONFIG.get("backend") or "jax")
    base_model = str(cfg.get("base_model") if cfg.get("base_model") is not None else DEFAULT_CONFIG["base_model"])
    output_dir = str(cfg.get("output_dir") if cfg.get("output_dir") is not None else DEFAULT_CONFIG["output_dir"])
    seed = int(cfg.get("seed") if cfg.get("seed") is not None else DEFAULT_CONFIG["seed"])
    device = str(cfg.get("device") if cfg.get("device") is not None else "cpu")

    jax_cfg = SidSftJaxConfig(
        mesh_shape=str(_get_or_default(cfg, "jax.mesh_shape", DEFAULT_CONFIG.get("jax", {}).get("mesh_shape") or "1,-1,1")),
        param_dtype=str(_get_or_default(cfg, "jax.param_dtype", DEFAULT_CONFIG.get("jax", {}).get("param_dtype") or "float32")),
        compute_dtype=str(_get_or_default(cfg, "jax.compute_dtype", DEFAULT_CONFIG.get("jax", {}).get("compute_dtype") or "bfloat16")),
        max_cache_length=int(_get_or_default(cfg, "jax.max_cache_length", DEFAULT_CONFIG.get("jax", {}).get("max_cache_length") or 2048)),
    )

    data = SidSftDataConfig(
        category=str(_get_or_default(cfg, "data.category", DEFAULT_CONFIG["data"]["category"])),
        train_file=str(_get_or_default(cfg, "data.train_file", DEFAULT_CONFIG["data"]["train_file"])),
        eval_file=str(_get_or_default(cfg, "data.eval_file", DEFAULT_CONFIG["data"]["eval_file"])),
        test_file=str(_get_or_default(cfg, "data.test_file", DEFAULT_CONFIG["data"]["test_file"])),
        info_file=str(_get_or_default(cfg, "data.info_file", DEFAULT_CONFIG["data"]["info_file"])),
        sid_index_path=str(_get_or_default(cfg, "data.sid_index_path", DEFAULT_CONFIG["data"]["sid_index_path"])),
        item_meta_path=str(_get_or_default(cfg, "data.item_meta_path", DEFAULT_CONFIG["data"]["item_meta_path"])),
        max_len=int(_get_or_default(cfg, "data.max_len", DEFAULT_CONFIG["data"]["max_len"])),
        sample_train=int(_get_or_default(cfg, "data.sample_train", DEFAULT_CONFIG["data"]["sample_train"])),
        sample_eval=int(_get_or_default(cfg, "data.sample_eval", DEFAULT_CONFIG["data"]["sample_eval"])),
        sample_test=int(_get_or_default(cfg, "data.sample_test", DEFAULT_CONFIG["data"]["sample_test"])),
    )

    tasks = SidSftTasksConfig(
        sid_next_item=bool(_get_by_path(cfg, "tasks.sid_next_item") if _get_by_path(cfg, "tasks.sid_next_item") is not None else True),
        sid_item_alignment=bool(
            _get_by_path(cfg, "tasks.sid_item_alignment") if _get_by_path(cfg, "tasks.sid_item_alignment") is not None else True
        ),
        fusion_seq_rec=bool(_get_by_path(cfg, "tasks.fusion_seq_rec") if _get_by_path(cfg, "tasks.fusion_seq_rec") is not None else True),
    )

    train = SidSftTrainConfig(
        per_device_train_batch_size=int(_get_or_default(cfg, "train.per_device_train_batch_size", DEFAULT_CONFIG["train"]["per_device_train_batch_size"])),
        per_device_eval_batch_size=int(_get_or_default(cfg, "train.per_device_eval_batch_size", DEFAULT_CONFIG["train"]["per_device_eval_batch_size"])),
        global_batch_size=int(_get_or_default(cfg, "train.global_batch_size", DEFAULT_CONFIG["train"].get("global_batch_size") or 0)),
        gradient_accumulation_steps=int(_get_or_default(cfg, "train.gradient_accumulation_steps", DEFAULT_CONFIG["train"]["gradient_accumulation_steps"])),
        learning_rate=float(_get_or_default(cfg, "train.learning_rate", DEFAULT_CONFIG["train"]["learning_rate"])),
        optimizer=str(_get_or_default(cfg, "train.optimizer", DEFAULT_CONFIG["train"].get("optimizer") or "adamw")),
        weight_decay=float(_get_or_default(cfg, "train.weight_decay", DEFAULT_CONFIG["train"].get("weight_decay") or 0.0)),
        num_train_epochs=float(_get_or_default(cfg, "train.num_train_epochs", DEFAULT_CONFIG["train"]["num_train_epochs"])),
        max_steps=int(_get_or_default(cfg, "train.max_steps", DEFAULT_CONFIG["train"]["max_steps"])),
        warmup_steps=int(_get_or_default(cfg, "train.warmup_steps", DEFAULT_CONFIG["train"]["warmup_steps"])),
        logging_steps=int(_get_or_default(cfg, "train.logging_steps", DEFAULT_CONFIG["train"]["logging_steps"])),
        eval_steps=int(_get_or_default(cfg, "train.eval_steps", DEFAULT_CONFIG["train"]["eval_steps"])),
        save_steps=int(_get_or_default(cfg, "train.save_steps", DEFAULT_CONFIG["train"]["save_steps"])),
        save_total_limit=int(_get_or_default(cfg, "train.save_total_limit", DEFAULT_CONFIG["train"]["save_total_limit"])),
        save_last=bool(_get_or_default(cfg, "train.save_last", DEFAULT_CONFIG["train"].get("save_last", True))),
        group_by_length=bool(_get_or_default(cfg, "train.group_by_length", DEFAULT_CONFIG["train"]["group_by_length"])),
        freeze_LLM=bool(_get_or_default(cfg, "train.freeze_LLM", DEFAULT_CONFIG["train"]["freeze_LLM"])),
        train_from_scratch=bool(_get_or_default(cfg, "train.train_from_scratch", DEFAULT_CONFIG["train"]["train_from_scratch"])),
        resume_from_checkpoint=_get_or_default(cfg, "train.resume_from_checkpoint", DEFAULT_CONFIG["train"]["resume_from_checkpoint"]),
        early_stopping_patience=int(_get_or_default(cfg, "train.early_stopping_patience", DEFAULT_CONFIG["train"]["early_stopping_patience"])),
        bf16=bool(_get_or_default(cfg, "train.bf16", DEFAULT_CONFIG["train"]["bf16"])),
        fp16=bool(_get_or_default(cfg, "train.fp16", DEFAULT_CONFIG["train"]["fp16"])),
    )

    eval_cfg = SidSftEvalConfig(
        enabled=bool(_get_by_path(cfg, "eval.enabled") if _get_by_path(cfg, "eval.enabled") is not None else DEFAULT_CONFIG["eval"]["enabled"]),
        batch_size=int(_get_or_default(cfg, "eval.batch_size", DEFAULT_CONFIG["eval"]["batch_size"])),
        num_beams=int(_get_or_default(cfg, "eval.num_beams", DEFAULT_CONFIG["eval"]["num_beams"])),
        max_new_tokens=int(_get_or_default(cfg, "eval.max_new_tokens", DEFAULT_CONFIG["eval"]["max_new_tokens"])),
        length_penalty=float(_get_or_default(cfg, "eval.length_penalty", DEFAULT_CONFIG["eval"]["length_penalty"])),
        topk=tuple(_get_or_default(cfg, "eval.topk", DEFAULT_CONFIG["eval"]["topk"])),
        constrained=bool(_get_by_path(cfg, "eval.constrained") if _get_by_path(cfg, "eval.constrained") is not None else True),
        save_predictions_json=bool(
            _get_by_path(cfg, "eval.save_predictions_json") if _get_by_path(cfg, "eval.save_predictions_json") is not None else True
        ),
        prefill_mode=str(_get_or_default(cfg, "eval.prefill_mode", DEFAULT_CONFIG["eval"].get("prefill_mode") or "bucket")),
        fixed_prefill_len=(
            None
            if _get_or_default(cfg, "eval.fixed_prefill_len", DEFAULT_CONFIG["eval"].get("fixed_prefill_len")) is None
            else int(_get_or_default(cfg, "eval.fixed_prefill_len", DEFAULT_CONFIG["eval"].get("fixed_prefill_len")))
        ),
    )

    wandb_cfg = SidSftWandbConfig(
        project=str(_get_or_default(cfg, "wandb.project", DEFAULT_CONFIG["wandb"]["project"])),
        mode=str(_get_or_default(cfg, "wandb.mode", DEFAULT_CONFIG["wandb"]["mode"])),
        name=_get_or_default(cfg, "wandb.name", DEFAULT_CONFIG["wandb"]["name"]),
    )

    if wandb_cfg.name is None:
        sha = _maybe_git_short_sha()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"{data.category}-sid-sft-{ts}"
        if sha:
            suffix = f"{suffix}-{sha}"
        wandb_cfg = SidSftWandbConfig(project=wandb_cfg.project, mode=wandb_cfg.mode, name=suffix)

    return SidSftConfig(
        config_path=config_path,
        backend=backend,
        base_model=base_model,
        output_dir=output_dir,
        seed=seed,
        device=device,
        jax=jax_cfg,
        data=data,
        tasks=tasks,
        train=train,
        eval=eval_cfg,
        wandb=wandb_cfg,
    )


def main() -> None:
    load_dotenv_if_present(repo_root=REPO_ROOT)

    parser = ArgumentParser(description="Run MiniOneRec SID SFT (plugins/sft).")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    parser.add_argument("--run-mode", type=str, default="train_eval", choices=["train", "eval", "train_eval"])
    parser.add_argument("--print-config", action="store_true", help="Print the resolved config and exit.")
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    config_path = str(args.config or "<default>")
    cfg = _cfg_from_dict(cfg_dict, config_path=config_path)

    if args.print_config:
        print(yaml.safe_dump(json.loads(json.dumps(asdict(cfg))), sort_keys=False))
        return

    os.makedirs(cfg.output_dir, exist_ok=True)
    result = run_sid_sft(cfg, run_mode=args.run_mode)
    summary_path = os.path.join(cfg.output_dir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"summary_json={summary_path}")


if __name__ == "__main__":
    main()
