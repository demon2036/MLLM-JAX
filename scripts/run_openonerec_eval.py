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
from projects.openonerec_eval.config import DEFAULT_CONFIG, load_config
from projects.openonerec_eval.runner import (
    OpenOneRecEvalConfig,
    OpenOneRecEvalDataConfig,
    OpenOneRecEvalEvalConfig,
    OpenOneRecEvalGenerationConfig,
    OpenOneRecEvalJaxConfig,
    OpenOneRecEvalWandbConfig,
    run_openonerec_eval,
)


def _get_by_path(cfg: dict[str, Any], key_path: str) -> Any:
    cur: Any = cfg
    for key in [k for k in key_path.split(".") if k]:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _get_or_default(cfg: dict[str, Any], key_path: str, default: Any) -> Any:
    val = _get_by_path(cfg, key_path)
    return default if val is None else val


def _cfg_from_dict(cfg: dict[str, Any], *, config_path: str) -> OpenOneRecEvalConfig:
    eval_sample_size = _get_or_default(cfg, "eval.sample_size", DEFAULT_CONFIG["eval"]["sample_size"])
    if isinstance(eval_sample_size, str) and eval_sample_size.strip().lower() == "none":
        eval_sample_size = None

    return OpenOneRecEvalConfig(
        config_path=config_path,
        openonerec_root=str(_get_or_default(cfg, "openonerec_root", DEFAULT_CONFIG["openonerec_root"])),
        base_model=str(_get_or_default(cfg, "base_model", DEFAULT_CONFIG["base_model"])),
        output_dir=str(_get_or_default(cfg, "output_dir", DEFAULT_CONFIG["output_dir"])),
        seed=int(_get_or_default(cfg, "seed", DEFAULT_CONFIG["seed"])),
        eval=OpenOneRecEvalEvalConfig(
            task_types=tuple(_get_or_default(cfg, "eval.task_types", DEFAULT_CONFIG["eval"]["task_types"])),
            splits=tuple(_get_or_default(cfg, "eval.splits", DEFAULT_CONFIG["eval"]["splits"])),
            sample_size=eval_sample_size,
            overwrite=bool(_get_or_default(cfg, "eval.overwrite", DEFAULT_CONFIG["eval"]["overwrite"])),
            enable_thinking=bool(_get_or_default(cfg, "eval.enable_thinking", DEFAULT_CONFIG["eval"]["enable_thinking"])),
            run_official_evaluator=bool(
                _get_or_default(cfg, "eval.run_official_evaluator", DEFAULT_CONFIG["eval"]["run_official_evaluator"])
            ),
            evaluation_mode=(
                None
                if _get_by_path(cfg, "eval.evaluation_mode") is None
                else str(_get_or_default(cfg, "eval.evaluation_mode", DEFAULT_CONFIG["eval"].get("evaluation_mode")))
            ),
        ),
        generation=OpenOneRecEvalGenerationConfig(
            mode=str(_get_or_default(cfg, "generation.mode", DEFAULT_CONFIG["generation"]["mode"])),
            batch_size=int(_get_or_default(cfg, "generation.batch_size", DEFAULT_CONFIG["generation"]["batch_size"])),
            num_beams=int(_get_or_default(cfg, "generation.num_beams", DEFAULT_CONFIG["generation"]["num_beams"])),
            num_return_sequences=int(
                _get_or_default(cfg, "generation.num_return_sequences", DEFAULT_CONFIG["generation"]["num_return_sequences"])
            ),
            max_new_tokens=int(_get_or_default(cfg, "generation.max_new_tokens", DEFAULT_CONFIG["generation"]["max_new_tokens"])),
            temperature=float(_get_or_default(cfg, "generation.temperature", DEFAULT_CONFIG["generation"]["temperature"])),
            top_p=float(_get_or_default(cfg, "generation.top_p", DEFAULT_CONFIG["generation"]["top_p"])),
            top_k=int(_get_or_default(cfg, "generation.top_k", DEFAULT_CONFIG["generation"]["top_k"])),
            prompt_token=str(_get_or_default(cfg, "generation.prompt_token", DEFAULT_CONFIG["generation"]["prompt_token"])),
            params_checkpoint_path=_get_or_default(
                cfg,
                "generation.params_checkpoint_path",
                DEFAULT_CONFIG["generation"].get("params_checkpoint_path"),
            ),
            replay_path_template=_get_or_default(
                cfg,
                "generation.replay_path_template",
                DEFAULT_CONFIG["generation"].get("replay_path_template"),
            ),
        ),
        jax=OpenOneRecEvalJaxConfig(
            mesh_shape=str(_get_or_default(cfg, "jax.mesh_shape", DEFAULT_CONFIG["jax"]["mesh_shape"])),
            max_cache_length=int(_get_or_default(cfg, "jax.max_cache_length", DEFAULT_CONFIG["jax"]["max_cache_length"])),
            param_dtype=str(_get_or_default(cfg, "jax.param_dtype", DEFAULT_CONFIG["jax"]["param_dtype"])),
        ),
        data=OpenOneRecEvalDataConfig(
            benchmark_data_dir=str(
                _get_or_default(cfg, "data.benchmark_data_dir", DEFAULT_CONFIG["data"]["benchmark_data_dir"])
            ),
        ),
        wandb=OpenOneRecEvalWandbConfig(
            project=str(_get_or_default(cfg, "wandb.project", DEFAULT_CONFIG["wandb"]["project"])),
            mode=str(_get_or_default(cfg, "wandb.mode", DEFAULT_CONFIG["wandb"]["mode"])),
            name=_get_or_default(cfg, "wandb.name", DEFAULT_CONFIG["wandb"]["name"]),
        ),
    )


def main() -> None:
    load_dotenv_if_present(repo_root=REPO_ROOT)

    parser = ArgumentParser(description="Run OpenOneRec recommendation evaluation (projects/openonerec_eval).")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    parser.add_argument("--run-mode", type=str, default="eval", choices=["eval"])
    parser.add_argument("--print-config", action="store_true", help="Print resolved config and exit.")
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    cfg = _cfg_from_dict(cfg_dict, config_path=str(args.config or "<default>"))

    if args.print_config:
        print(yaml.safe_dump(json.loads(json.dumps(asdict(cfg))), sort_keys=False))
        return

    os.makedirs(cfg.output_dir, exist_ok=True)
    result = run_openonerec_eval(cfg, run_mode=args.run_mode)

    summary_path = os.path.join(cfg.output_dir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"summary_json={summary_path}")


if __name__ == "__main__":
    main()
