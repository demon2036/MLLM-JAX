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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from plugins.training.core.runtime.env import load_dotenv_if_present
from projects.minionerec_rl.config import DEFAULT_CONFIG, load_config
from projects.minionerec_rl.runner.minionerec_rl import (
    MiniOneRecRlConfig,
    MiniOneRecRlDataConfig,
    MiniOneRecRlEmaConfig,
    MiniOneRecRlEvalConfig,
    MiniOneRecRlJaxConfig,
    MiniOneRecRlRolloutConfig,
    MiniOneRecRlTasksConfig,
    MiniOneRecRlTrainConfig,
    MiniOneRecRlWandbConfig,
    run_minionerec_rl,
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


def _cfg_from_dict(cfg: dict[str, Any], *, config_path: str) -> MiniOneRecRlConfig:
    base_model = str(cfg.get("base_model") if cfg.get("base_model") is not None else DEFAULT_CONFIG["base_model"])
    output_dir = str(cfg.get("output_dir") if cfg.get("output_dir") is not None else DEFAULT_CONFIG["output_dir"])
    seed = int(cfg.get("seed") if cfg.get("seed") is not None else DEFAULT_CONFIG["seed"])
    device = str(cfg.get("device") if cfg.get("device") is not None else "tpu")

    jax_cfg = MiniOneRecRlJaxConfig(
        mesh_shape=str(_get_or_default(cfg, "jax.mesh_shape", DEFAULT_CONFIG["jax"]["mesh_shape"])),
        param_dtype=str(_get_or_default(cfg, "jax.param_dtype", DEFAULT_CONFIG["jax"]["param_dtype"])),
        compute_dtype=str(_get_or_default(cfg, "jax.compute_dtype", DEFAULT_CONFIG["jax"]["compute_dtype"])),
        max_cache_length=int(_get_or_default(cfg, "jax.max_cache_length", DEFAULT_CONFIG["jax"]["max_cache_length"])),
    )

    data = MiniOneRecRlDataConfig(
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
        init_from_sft_checkpoint=_get_or_default(cfg, "data.init_from_sft_checkpoint", DEFAULT_CONFIG["data"]["init_from_sft_checkpoint"]),
    )

    tasks = MiniOneRecRlTasksConfig(
        next_item=bool(_get_by_path(cfg, "tasks.next_item") if _get_by_path(cfg, "tasks.next_item") is not None else DEFAULT_CONFIG["tasks"]["next_item"]),
        title2sid=bool(_get_by_path(cfg, "tasks.title2sid") if _get_by_path(cfg, "tasks.title2sid") is not None else DEFAULT_CONFIG["tasks"]["title2sid"]),
        description2sid=bool(
            _get_by_path(cfg, "tasks.description2sid")
            if _get_by_path(cfg, "tasks.description2sid") is not None
            else DEFAULT_CONFIG["tasks"]["description2sid"]
        ),
        seqtitle2sid=bool(
            _get_by_path(cfg, "tasks.seqtitle2sid") if _get_by_path(cfg, "tasks.seqtitle2sid") is not None else DEFAULT_CONFIG["tasks"]["seqtitle2sid"]
        ),
        seqtitle2sid_sample=int(_get_or_default(cfg, "tasks.seqtitle2sid_sample", DEFAULT_CONFIG["tasks"]["seqtitle2sid_sample"])),
    )

    rollout = MiniOneRecRlRolloutConfig(
        prompt_batch_size=int(_get_or_default(cfg, "rollout.prompt_batch_size", DEFAULT_CONFIG["rollout"]["prompt_batch_size"])),
        num_generations=int(_get_or_default(cfg, "rollout.num_generations", DEFAULT_CONFIG["rollout"]["num_generations"])),
        prompt_pad_len=int(_get_or_default(cfg, "rollout.prompt_pad_len", DEFAULT_CONFIG["rollout"]["prompt_pad_len"])),
        global_length=int(_get_or_default(cfg, "rollout.global_length", DEFAULT_CONFIG["rollout"]["global_length"])),
    )

    ema_cfg = MiniOneRecRlEmaConfig(
        enabled=bool(_get_or_default(cfg, "train.ema.enabled", DEFAULT_CONFIG["train"]["ema"]["enabled"])),
        decay=float(_get_or_default(cfg, "train.ema.decay", DEFAULT_CONFIG["train"]["ema"]["decay"])),
        use_for_eval=bool(_get_or_default(cfg, "train.ema.use_for_eval", DEFAULT_CONFIG["train"]["ema"]["use_for_eval"])),
    )

    from plugins.training.core.optim.optimizer import LRScheduleConfig, OptimizerConfig

    optimizer_raw = _get_by_path(cfg, "train.optimizer")
    if optimizer_raw is None:
        optimizer_cfg = OptimizerConfig()
    elif isinstance(optimizer_raw, str):
        optimizer_cfg = OptimizerConfig(name=str(optimizer_raw))
    elif isinstance(optimizer_raw, dict):
        lr_raw = optimizer_raw.get("lr_schedule")
        if lr_raw is None:
            lr_cfg = LRScheduleConfig()
        elif isinstance(lr_raw, dict):
            lr_cfg = LRScheduleConfig(
                type=str(lr_raw.get("type", DEFAULT_CONFIG["train"]["optimizer"]["lr_schedule"]["type"])),
                init_value=float(lr_raw.get("init_value", DEFAULT_CONFIG["train"]["optimizer"]["lr_schedule"]["init_value"])),
                peak_value=float(lr_raw.get("peak_value", DEFAULT_CONFIG["train"]["optimizer"]["lr_schedule"]["peak_value"])),
                end_value=float(lr_raw.get("end_value", DEFAULT_CONFIG["train"]["optimizer"]["lr_schedule"]["end_value"])),
                warmup_ratio=float(lr_raw.get("warmup_ratio", DEFAULT_CONFIG["train"]["optimizer"]["lr_schedule"]["warmup_ratio"])),
                warmup_steps=lr_raw.get("warmup_steps", DEFAULT_CONFIG["train"]["optimizer"]["lr_schedule"]["warmup_steps"]),
            )
        else:
            raise ValueError("train.optimizer.lr_schedule must be a dict when provided")

        optimizer_cfg = OptimizerConfig(
            name=str(optimizer_raw.get("name", DEFAULT_CONFIG["train"]["optimizer"]["name"])),
            clip_norm=float(optimizer_raw.get("clip_norm", DEFAULT_CONFIG["train"]["optimizer"]["clip_norm"])),
            weight_decay=float(optimizer_raw.get("weight_decay", DEFAULT_CONFIG["train"]["optimizer"]["weight_decay"])),
            lr_schedule=lr_cfg,
            muon_aux_lr=float(optimizer_raw.get("muon_aux_lr", OptimizerConfig().muon_aux_lr)),
            muon_momentum=float(optimizer_raw.get("muon_momentum", OptimizerConfig().muon_momentum)),
            muon_nesterov=bool(optimizer_raw.get("muon_nesterov", OptimizerConfig().muon_nesterov)),
            muon_ns_steps=int(optimizer_raw.get("muon_ns_steps", OptimizerConfig().muon_ns_steps)),
            muon_eps=float(optimizer_raw.get("muon_eps", OptimizerConfig().muon_eps)),
            muon_max_dim=int(optimizer_raw.get("muon_max_dim", OptimizerConfig().muon_max_dim)),
        )
    else:
        raise ValueError("train.optimizer must be a dict or string when provided")

    train = MiniOneRecRlTrainConfig(
        num_train_epochs=float(_get_or_default(cfg, "train.num_train_epochs", DEFAULT_CONFIG["train"]["num_train_epochs"])),
        max_steps=int(_get_or_default(cfg, "train.max_steps", DEFAULT_CONFIG["train"]["max_steps"])),
        grad_accum_steps=int(_get_or_default(cfg, "train.grad_accum_steps", DEFAULT_CONFIG["train"]["grad_accum_steps"])),
        beta=float(_get_or_default(cfg, "train.beta", DEFAULT_CONFIG["train"]["beta"])),
        logging_steps=int(_get_or_default(cfg, "train.logging_steps", DEFAULT_CONFIG["train"]["logging_steps"])),
        save_last=bool(_get_or_default(cfg, "train.save_last", DEFAULT_CONFIG["train"]["save_last"])),
        ema=ema_cfg,
        optimizer=optimizer_cfg,
    )

    eval_cfg = MiniOneRecRlEvalConfig(
        enabled=bool(_get_or_default(cfg, "eval.enabled", DEFAULT_CONFIG["eval"]["enabled"])),
        every_steps=int(_get_or_default(cfg, "eval.every_steps", DEFAULT_CONFIG["eval"]["every_steps"])),
        batch_size=int(_get_or_default(cfg, "eval.batch_size", DEFAULT_CONFIG["eval"]["batch_size"])),
        num_beams=int(_get_or_default(cfg, "eval.num_beams", DEFAULT_CONFIG["eval"]["num_beams"])),
        topk=tuple(_get_or_default(cfg, "eval.topk", DEFAULT_CONFIG["eval"]["topk"])),
        save_predictions_json=bool(_get_or_default(cfg, "eval.save_predictions_json", DEFAULT_CONFIG["eval"]["save_predictions_json"])),
    )

    wandb_cfg = MiniOneRecRlWandbConfig(
        project=str(_get_or_default(cfg, "wandb.project", DEFAULT_CONFIG["wandb"]["project"])),
        mode=str(_get_or_default(cfg, "wandb.mode", DEFAULT_CONFIG["wandb"]["mode"])),
        name=_get_or_default(cfg, "wandb.name", DEFAULT_CONFIG["wandb"]["name"]),
    )

    if wandb_cfg.name is None:
        sha = _maybe_git_short_sha()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"{data.category}-sid-rl-{ts}"
        if sha:
            suffix = f"{suffix}-{sha}"
        wandb_cfg = MiniOneRecRlWandbConfig(project=wandb_cfg.project, mode=wandb_cfg.mode, name=suffix)

    return MiniOneRecRlConfig(
        config_path=config_path,
        base_model=base_model,
        output_dir=output_dir,
        seed=seed,
        device=device,
        data=data,
        tasks=tasks,
        jax=jax_cfg,
        rollout=rollout,
        train=train,
        eval=eval_cfg,
        wandb=wandb_cfg,
    )


def main() -> None:
    load_dotenv_if_present(repo_root=REPO_ROOT)

    parser = ArgumentParser(description="Run MiniOneRec RL (GRPO-style, JAX/TPU).")
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
    out = run_minionerec_rl(cfg, run_mode=args.run_mode)
    with open(os.path.join(cfg.output_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=False)


if __name__ == "__main__":
    main()

