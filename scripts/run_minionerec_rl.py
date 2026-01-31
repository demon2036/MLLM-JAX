from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime
from typing import Any

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from plugins.training.core.runtime.env import load_dotenv_if_present
from plugins.training.core.optim.optimizer import LRScheduleConfig, OptimizerConfig
from projects.minionerec_rl.config import DEFAULT_CONFIG, load_config
from projects.minionerec_rl.runner import (
    MiniOneRecRlConfig,
    MiniOneRecRlDataConfig,
    MiniOneRecRlEvalConfig,
    MiniOneRecRlJaxConfig,
    MiniOneRecRlRolloutConfig,
    MiniOneRecRlTrainConfig,
    MiniOneRecRlWandbConfig,
    run_minionerec_rl,
)


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


def _default_wandb_name(category: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"minionerec-rl-{category}-{ts}"


def _optimizer_from_dict(opt_raw: dict[str, Any], *, default_raw: dict[str, Any]) -> OptimizerConfig:
    opt_raw = dict(default_raw) | dict(opt_raw or {})

    lr_raw_default = dict(default_raw.get("lr_schedule") or {})
    lr_raw = dict(lr_raw_default) | dict(opt_raw.get("lr_schedule") or {})

    lr_cfg = LRScheduleConfig(
        type=str(lr_raw.get("type") or "warmup_cosine"),
        init_value=float(lr_raw.get("init_value") if lr_raw.get("init_value") is not None else 0.0),
        peak_value=float(lr_raw.get("peak_value") if lr_raw.get("peak_value") is not None else 1e-6),
        end_value=float(lr_raw.get("end_value") if lr_raw.get("end_value") is not None else 0.0),
        warmup_ratio=float(lr_raw.get("warmup_ratio") if lr_raw.get("warmup_ratio") is not None else 0.05),
        warmup_steps=None if lr_raw.get("warmup_steps") is None else int(lr_raw.get("warmup_steps")),
    )

    return OptimizerConfig(
        name=str(opt_raw.get("name") or "lion"),
        clip_norm=float(opt_raw.get("clip_norm") if opt_raw.get("clip_norm") is not None else 1.0),
        weight_decay=float(opt_raw.get("weight_decay") if opt_raw.get("weight_decay") is not None else 1e-8),
        lr_schedule=lr_cfg,
        muon_aux_lr=float(opt_raw.get("muon_aux_lr") if opt_raw.get("muon_aux_lr") is not None else 3e-4),
        muon_momentum=float(opt_raw.get("muon_momentum") if opt_raw.get("muon_momentum") is not None else 0.95),
        muon_nesterov=bool(opt_raw.get("muon_nesterov") if opt_raw.get("muon_nesterov") is not None else True),
        muon_ns_steps=int(opt_raw.get("muon_ns_steps") if opt_raw.get("muon_ns_steps") is not None else 5),
        muon_eps=float(opt_raw.get("muon_eps") if opt_raw.get("muon_eps") is not None else 1e-7),
        muon_max_dim=int(opt_raw.get("muon_max_dim") if opt_raw.get("muon_max_dim") is not None else 10_000),
    )


def _cfg_from_dict(cfg: dict[str, Any], *, config_path: str) -> MiniOneRecRlConfig:
    base_model = str(_get_or_default(cfg, "base_model", DEFAULT_CONFIG["base_model"]))
    output_dir = str(_get_or_default(cfg, "output_dir", DEFAULT_CONFIG["output_dir"]))
    seed = int(_get_or_default(cfg, "seed", DEFAULT_CONFIG["seed"]))
    device = str(_get_or_default(cfg, "device", DEFAULT_CONFIG["device"]))

    data = MiniOneRecRlDataConfig(
        category=str(_get_or_default(cfg, "data.category", DEFAULT_CONFIG["data"]["category"])),
        train_file=str(_get_or_default(cfg, "data.train_file", DEFAULT_CONFIG["data"]["train_file"])),
        eval_file=str(_get_or_default(cfg, "data.eval_file", DEFAULT_CONFIG["data"]["eval_file"])),
        test_file=str(_get_or_default(cfg, "data.test_file", DEFAULT_CONFIG["data"]["test_file"])),
        info_file=str(_get_or_default(cfg, "data.info_file", DEFAULT_CONFIG["data"]["info_file"])),
        sid_index_path=str(_get_or_default(cfg, "data.sid_index_path", DEFAULT_CONFIG["data"]["sid_index_path"])),
        max_len=int(_get_or_default(cfg, "data.max_len", DEFAULT_CONFIG["data"]["max_len"])),
        sample_train=int(_get_or_default(cfg, "data.sample_train", DEFAULT_CONFIG["data"]["sample_train"])),
        sample_eval=int(_get_or_default(cfg, "data.sample_eval", DEFAULT_CONFIG["data"]["sample_eval"])),
        sample_test=int(_get_or_default(cfg, "data.sample_test", DEFAULT_CONFIG["data"]["sample_test"])),
    )

    jax_cfg = MiniOneRecRlJaxConfig(
        mesh_shape=str(_get_or_default(cfg, "jax.mesh_shape", DEFAULT_CONFIG["jax"]["mesh_shape"])),
        param_dtype=str(_get_or_default(cfg, "jax.param_dtype", DEFAULT_CONFIG["jax"]["param_dtype"])),
        compute_dtype=str(_get_or_default(cfg, "jax.compute_dtype", DEFAULT_CONFIG["jax"]["compute_dtype"])),
        max_cache_length=int(_get_or_default(cfg, "jax.max_cache_length", DEFAULT_CONFIG["jax"]["max_cache_length"])),
    )

    rollout = MiniOneRecRlRolloutConfig(
        prompt_batch_size=int(_get_or_default(cfg, "rollout.prompt_batch_size", DEFAULT_CONFIG["rollout"]["prompt_batch_size"])),
        num_generations=int(_get_or_default(cfg, "rollout.num_generations", DEFAULT_CONFIG["rollout"]["num_generations"])),
        prompt_pad_len=int(_get_or_default(cfg, "rollout.prompt_pad_len", DEFAULT_CONFIG["rollout"]["prompt_pad_len"])),
        global_length=int(_get_or_default(cfg, "rollout.global_length", DEFAULT_CONFIG["rollout"]["global_length"])),
    )

    train_raw = dict(_get_or_default(cfg, "train", DEFAULT_CONFIG["train"]) or {})
    opt_cfg = _optimizer_from_dict(
        dict(train_raw.get("optimizer") or {}),
        default_raw=dict(DEFAULT_CONFIG["train"]["optimizer"]),
    )
    train = MiniOneRecRlTrainConfig(
        num_train_epochs=float(_get_or_default(cfg, "train.num_train_epochs", DEFAULT_CONFIG["train"]["num_train_epochs"])),
        max_steps=int(_get_or_default(cfg, "train.max_steps", DEFAULT_CONFIG["train"]["max_steps"])),
        grad_accum_steps=int(_get_or_default(cfg, "train.grad_accum_steps", DEFAULT_CONFIG["train"]["grad_accum_steps"])),
        ppo_steps=int(_get_or_default(cfg, "train.ppo_steps", DEFAULT_CONFIG["train"]["ppo_steps"])),
        beta=float(_get_or_default(cfg, "train.beta", DEFAULT_CONFIG["train"]["beta"])),
        logging_steps=int(_get_or_default(cfg, "train.logging_steps", DEFAULT_CONFIG["train"]["logging_steps"])),
        save_last=bool(_get_or_default(cfg, "train.save_last", DEFAULT_CONFIG["train"]["save_last"])),
        optimizer=opt_cfg,
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
        wandb_cfg = MiniOneRecRlWandbConfig(project=wandb_cfg.project, mode=wandb_cfg.mode, name=_default_wandb_name(data.category))

    resume_from_checkpoint = _get_or_default(cfg, "resume_from_checkpoint", DEFAULT_CONFIG.get("resume_from_checkpoint"))
    if resume_from_checkpoint is not None:
        resume_from_checkpoint = str(resume_from_checkpoint)

    return MiniOneRecRlConfig(
        config_path=config_path,
        base_model=base_model,
        output_dir=output_dir,
        seed=seed,
        device=device,
        data=data,
        jax=jax_cfg,
        rollout=rollout,
        train=train,
        eval=eval_cfg,
        wandb=wandb_cfg,
        resume_from_checkpoint=resume_from_checkpoint,
    )


def main() -> None:
    load_dotenv_if_present(repo_root=REPO_ROOT)

    parser = ArgumentParser(description="Run MiniOneRec SID RL (projects/minionerec_rl).")
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
    result = run_minionerec_rl(cfg, run_mode=args.run_mode)
    summary_path = os.path.join(cfg.output_dir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"summary_json={summary_path}")


if __name__ == "__main__":
    main()

