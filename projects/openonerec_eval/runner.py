from __future__ import annotations

import copy
import importlib
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from plugins.training.core.logging.wandb import maybe_init_wandb
from projects.openonerec_eval.jax_generator import OpenOneRecJaxGenerator

PAPER_RECALL32_TARGETS: dict[str, float] = {
    "video": 0.0272,
    "ad": 0.0707,
    "product": 0.0360,
    "label_cond": 0.0184,
    "interactive": 0.1941,
}


@dataclass(frozen=True)
class OpenOneRecEvalEvalConfig:
    task_types: tuple[str, ...] = ("video", "ad", "product", "label_cond", "interactive")
    splits: tuple[str, ...] = ("test",)
    sample_size: int | str | None = None
    overwrite: bool = False
    enable_thinking: bool = False
    run_official_evaluator: bool = True
    # Optional evaluator override (official evaluator still computes metrics).
    evaluation_mode: str | None = None


@dataclass(frozen=True)
class OpenOneRecEvalGenerationConfig:
    mode: str = "jax"  # jax | replay
    batch_size: int = 4
    num_beams: int = 32
    num_return_sequences: int = 32
    max_new_tokens: int = 3
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 50
    prompt_token: str = "<|sid_begin|>"
    replay_path_template: str | None = None
    params_checkpoint_path: str | None = None


@dataclass(frozen=True)
class OpenOneRecEvalJaxConfig:
    mesh_shape: str = "1,-1,1"
    max_cache_length: int = 512
    param_dtype: str = "float32"


@dataclass(frozen=True)
class OpenOneRecEvalDataConfig:
    benchmark_data_dir: str = "workdir/OpenOneRec/benchmarks/data"


@dataclass(frozen=True)
class OpenOneRecEvalWandbConfig:
    project: str = "openonerec-eval"
    mode: str = "disabled"
    name: str | None = None


@dataclass(frozen=True)
class OpenOneRecEvalConfig:
    config_path: str
    openonerec_root: str
    base_model: str
    output_dir: str
    seed: int
    eval: OpenOneRecEvalEvalConfig = field(default_factory=OpenOneRecEvalEvalConfig)
    generation: OpenOneRecEvalGenerationConfig = field(default_factory=OpenOneRecEvalGenerationConfig)
    jax: OpenOneRecEvalJaxConfig = field(default_factory=OpenOneRecEvalJaxConfig)
    data: OpenOneRecEvalDataConfig = field(default_factory=OpenOneRecEvalDataConfig)
    wandb: OpenOneRecEvalWandbConfig = field(default_factory=OpenOneRecEvalWandbConfig)


def _set_seed(seed: int) -> None:
    seed_i = int(seed)
    random.seed(seed_i)
    np.random.seed(seed_i)


def _resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _normalize_model_name(base_model: str) -> str:
    return str(base_model).strip().replace("/", "__")


def _import_openonerec_registry(openonerec_root: str):
    benchmarks_dir = _resolve_path(str(Path(openonerec_root) / "benchmarks"))
    if not benchmarks_dir.exists():
        raise FileNotFoundError(f"OpenOneRec benchmarks directory not found: {benchmarks_dir}")

    bench_str = str(benchmarks_dir)
    if bench_str not in sys.path:
        sys.path.insert(0, bench_str)

    importlib.invalidate_caches()
    from benchmark.tasks.v1_0.registry import get_evaluator, get_loader, get_task_config

    return get_loader, get_evaluator, get_task_config


def _build_generation_payload(
    *,
    model_name: str,
    task_name: str,
    split: str,
    samples: dict[str, dict[str, Any]],
    generations: dict[str, list[str]],
    logprobs: dict[str, list[float]],
    total_time: float,
) -> dict[str, Any]:
    sample_payload: dict[str, dict[str, Any]] = {}
    for sample_id, sample in samples.items():
        payload_item: dict[str, Any] = {
            "prompt": sample.get("prompt", ""),
            "generations": list(generations.get(sample_id, [])),
            "ground_truth": sample.get("ground_truth", ""),
            "metadata": sample.get("metadata", {}),
        }
        if sample_id in logprobs and logprobs[sample_id]:
            payload_item["logprobs"] = [float(x) for x in logprobs[sample_id]]
        sample_payload[sample_id] = payload_item

    sample_count = len(sample_payload)
    avg_time = float(total_time) / float(sample_count) if sample_count > 0 else 0.0
    return {
        "model_name": model_name,
        "task_name": task_name,
        "split": split,
        "total_time": float(total_time),
        "avg_time_per_sample": float(avg_time),
        "samples": sample_payload,
    }


def _read_replay_samples(replay_path: Path) -> dict[str, Any]:
    with replay_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "samples" in data and isinstance(data["samples"], dict):
        return data

    raise ValueError(f"Replay file missing required top-level 'samples' map: {replay_path}")


def _extract_replay_generations(
    *,
    loader_samples: dict[str, dict[str, Any]],
    replay_data: dict[str, Any],
) -> tuple[dict[str, list[str]], dict[str, list[float]], float]:
    replay_samples = replay_data.get("samples", {})
    generations: dict[str, list[str]] = {}
    logprobs: dict[str, list[float]] = {}

    for sample_id in loader_samples:
        replay_item = replay_samples.get(sample_id, {})
        gens = replay_item.get("generations", [])
        lps = replay_item.get("logprobs", [])

        generations[sample_id] = [str(x) for x in gens]
        if isinstance(lps, list):
            logprobs[sample_id] = [float(x) for x in lps]

    replay_total_time = float(replay_data.get("total_time", 0.0) or 0.0)
    return generations, logprobs, replay_total_time


def _resolve_replay_path(template: str | None, task_name: str, split: str) -> Path:
    if not template:
        raise ValueError("generation.mode='replay' requires generation.replay_path_template in config.")

    replay_path = _resolve_path(str(template).format(task=task_name, split=split))
    if not replay_path.exists():
        raise FileNotFoundError(f"Replay file not found: {replay_path}")

    return replay_path


def _build_paper_alignment(*, model_name: str, model_results: dict[str, Any], splits: tuple[str, ...]) -> dict[str, Any]:
    split_order = [str(x) for x in splits]
    per_task: dict[str, Any] = {}

    for task_name, target in PAPER_RECALL32_TARGETS.items():
        actual = None
        task_results = model_results.get(task_name, {}) if isinstance(model_results, dict) else {}

        for split in split_order:
            split_metrics = task_results.get(split, {}) if isinstance(task_results, dict) else {}
            if "recall@32" in split_metrics:
                actual = float(split_metrics["recall@32"])
                break

        if actual is None:
            per_task[task_name] = {
                "metric": "recall@32",
                "value": None,
                "target": float(target),
                "delta": None,
                "available": False,
            }
        else:
            per_task[task_name] = {
                "metric": "recall@32",
                "value": float(actual),
                "target": float(target),
                "delta": float(actual - target),
                "available": True,
            }

    return {
        "model_name": model_name,
        "paper": "OneRec-1.7B",
        "task_metric": "recall@32",
        "targets": PAPER_RECALL32_TARGETS,
        "tasks": per_task,
    }


def run_openonerec_eval(cfg: OpenOneRecEvalConfig, *, run_mode: str = "eval") -> dict[str, Any]:
    _set_seed(cfg.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    run_mode_norm = str(run_mode).strip().lower()
    if run_mode_norm != "eval":
        raise ValueError("run_mode must be 'eval' for projects/openonerec_eval")

    output_dir = _resolve_path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    get_loader, get_evaluator, get_task_config = _import_openonerec_registry(cfg.openonerec_root)

    from transformers import AutoTokenizer

    model_name = _normalize_model_name(cfg.base_model)
    generation_root = output_dir / model_name

    mode_norm = str(cfg.generation.mode or "jax").strip().lower()
    if mode_norm not in {"jax", "replay"}:
        raise ValueError(f"Unsupported generation.mode={cfg.generation.mode!r} (expected jax|replay)")

    if mode_norm == "jax":
        generator = OpenOneRecJaxGenerator(base_model=cfg.base_model, generation_cfg=cfg.generation, jax_cfg=cfg.jax, benchmark_data_dir=str(cfg.data.benchmark_data_dir))
        tokenizer = generator.tokenizer
    else:
        generator = None
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)

    wandb = maybe_init_wandb(
        cfg=cfg,
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        mode=cfg.wandb.mode,
        process_index=0,
    )

    eval_results: dict[str, Any] = {model_name: {}}
    total_time = 0.0

    for task_name in cfg.eval.task_types:
        task_result: dict[str, Any] = {}
        task_cfg = copy.deepcopy(get_task_config(task_name=task_name))
        if cfg.eval.evaluation_mode:
            task_cfg.setdefault("evaluation_config", {})["evaluation_mode"] = str(cfg.eval.evaluation_mode)

        loader = get_loader(
            task_name=task_name,
            data_dir=str(cfg.data.benchmark_data_dir),
            tokenizer=tokenizer,
            enable_thinking=bool(cfg.eval.enable_thinking),
        )
        evaluator_class = get_evaluator(task_name=task_name)

        for split in cfg.eval.splits:
            samples = loader.load_data(split=str(split), sample_size=cfg.eval.sample_size)

            if mode_norm == "jax":
                generations, logprobs, generation_time = generator.generate_samples(task_name=task_name, samples=samples)
            else:
                replay_path = _resolve_replay_path(cfg.generation.replay_path_template, task_name=task_name, split=str(split))
                replay_data = _read_replay_samples(replay_path)
                generations, logprobs, generation_time = _extract_replay_generations(
                    loader_samples=samples,
                    replay_data=replay_data,
                )

            payload = _build_generation_payload(
                model_name=model_name,
                task_name=task_name,
                split=str(split),
                samples=samples,
                generations=generations,
                logprobs=logprobs,
                total_time=float(generation_time),
            )

            generation_dir = generation_root / task_name
            generation_dir.mkdir(parents=True, exist_ok=True)
            generation_file = generation_dir / f"{split}_generated.json"
            with generation_file.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
                f.write("\n")

            metrics: dict[str, Any] = {}
            if cfg.eval.run_official_evaluator:
                evaluator = evaluator_class(
                    samples=payload["samples"],
                    task_name=task_name,
                    predictions_dir=str(generation_dir),
                    debug=False,
                    task_config=task_cfg,
                    data_dir=str(cfg.data.benchmark_data_dir),
                    overwrite=bool(cfg.eval.overwrite),
                    cached_metrics={},
                )
                metrics, per_sample_metrics = evaluator.evaluate()
                for sample_id, sample_metric in per_sample_metrics.items():
                    if sample_id in payload["samples"]:
                        payload["samples"][sample_id].update(sample_metric)
                with generation_file.open("w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                    f.write("\n")

            metric_entry = {
                **metrics,
                "total_time": float(payload["total_time"]),
                "avg_time_per_sample": float(payload["avg_time_per_sample"]),
                "generation_file": str(generation_file),
            }
            task_result[str(split)] = metric_entry
            total_time += float(payload["total_time"])

            if wandb is not None:
                log_payload = {
                    f"eval/{task_name}/{split}/{k}": v
                    for k, v in metric_entry.items()
                    if isinstance(v, (float, int))
                }
                if log_payload:
                    wandb.log(log_payload)

        eval_results[model_name][task_name] = task_result

    eval_results[model_name]["_total_time"] = float(total_time)

    eval_results_path = output_dir / "eval_results.json"
    with eval_results_path.open("w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
        f.write("\n")

    paper_alignment = _build_paper_alignment(model_name=model_name, model_results=eval_results[model_name], splits=cfg.eval.splits)
    paper_alignment_path = output_dir / "paper_alignment.json"
    with paper_alignment_path.open("w", encoding="utf-8") as f:
        json.dump(paper_alignment, f, indent=2, ensure_ascii=False)
        f.write("\n")

    if wandb is not None:
        wandb.finish()

    return {
        "config": asdict(cfg),
        "generation_root": str(generation_root),
        "eval_results_path": str(eval_results_path),
        "paper_alignment_path": str(paper_alignment_path),
        "eval_results": eval_results,
    }


__all__ = [
    "OpenOneRecEvalConfig",
    "OpenOneRecEvalDataConfig",
    "OpenOneRecEvalEvalConfig",
    "OpenOneRecEvalGenerationConfig",
    "OpenOneRecEvalJaxConfig",
    "OpenOneRecEvalWandbConfig",
    "PAPER_RECALL32_TARGETS",
    "run_openonerec_eval",
]
