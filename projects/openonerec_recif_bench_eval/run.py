from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from plugins.training.core.runtime.env import load_dotenv_if_present
from projects.openonerec_recif_bench_eval.config import load_config


def _unset_socks_proxies() -> None:
    for key in ("ALL_PROXY", "all_proxy"):
        value = os.environ.get(key)
        if value and value.strip().lower().startswith("socks://"):
            os.environ.pop(key, None)


def _run(cmd: list[str], *, cwd: str | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def _require_generated_file(
    output_dir: str,
    *,
    model_repo_id: str,
    task_name: str,
    split: str,
) -> str:
    model_key = os.path.basename(model_repo_id.rstrip("/"))
    path = os.path.join(output_dir, model_key, task_name, f"{split}_generated.json")
    if not os.path.isfile(path):
        raise RuntimeError(f"Generation failed for task={task_name!r} split={split!r}; missing file: {path}")
    if os.path.getsize(path) <= 0:
        raise RuntimeError(f"Generation failed for task={task_name!r} split={split!r}; empty file: {path}")
    return path


def _ensure_openonerec_checkout(cfg: dict[str, Any]) -> str:
    upstream_cfg = cfg.get("upstream", {}) or {}
    repo_url = str(upstream_cfg.get("repo_url") or "https://github.com/Kuaishou-OneRec/OpenOneRec.git")
    checkout_dir = str(upstream_cfg.get("checkout_dir") or os.path.join(REPO_ROOT, "workdirs", "OpenOneRec"))

    checkout_path = Path(checkout_dir)
    if (checkout_path / ".git").is_dir():
        return str(checkout_path)

    checkout_path.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", repo_url, str(checkout_path)])
    return str(checkout_path)


def _ensure_recif_bench_data(cfg: dict[str, Any]) -> str:
    data_cfg = cfg.get("data", {}) or {}
    hf_dataset = str(data_cfg.get("hf_dataset") or "OpenOneRec/OpenOneRec-RecIF")
    local_dir = str(data_cfg.get("local_dir") or os.path.join(REPO_ROOT, "workdirs", "raw_data", "onerec_data"))
    include = data_cfg.get("include") or ["benchmark_data/**"]

    benchmark_dir = Path(local_dir) / "benchmark_data"
    if benchmark_dir.is_dir():
        return str(benchmark_dir)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_API_TOKEN")
    if not token:
        raise RuntimeError("Missing HF_TOKEN (set it via .env or environment variables).")

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"huggingface_hub is required to download datasets: {e}") from e

    Path(local_dir).mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=hf_dataset,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=list(include),
        token=token,
    )

    if not benchmark_dir.is_dir():
        raise FileNotFoundError(f"benchmark_data dir not found after download: {benchmark_dir}")
    return str(benchmark_dir)


def _patch_gemini_config(openonerec_benchmarks_dir: str, cfg: dict[str, Any]) -> None:
    llm_cfg = cfg.get("llm", {}) or {}
    location = str(llm_cfg.get("gemini_location") or "us-central1")
    project = llm_cfg.get("gemini_project")
    if not project:
        try:
            import google.auth

            _, project = google.auth.default()
        except Exception:
            project = None

    if not project:
        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("CLOUDSDK_CORE_PROJECT")

    if not project:
        raise RuntimeError(
            "Gemini judge requires a GCP project id. Provide llm.gemini_project in YAML or set GOOGLE_CLOUD_PROJECT."
        )

    config_path = Path(openonerec_benchmarks_dir) / "api" / "config" / "llm_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"llm_config.json not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gemini = data.get("gemini") or {}
    gemini["project"] = str(project)
    gemini["location"] = location
    data["gemini"] = gemini

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _run_eval(cfg: dict[str, Any], *, config_path: str) -> dict[str, Any]:
    _unset_socks_proxies()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Enable hf_transfer only when the optional dependency is available.
    # Some environments (local smoke) won't have it installed.
    try:
        import hf_transfer  # noqa: F401
    except Exception:
        os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    else:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    model_cfg = cfg.get("model", {}) or {}
    model_repo_id = str(model_cfg.get("repo_id") or "")
    if not model_repo_id:
        raise ValueError("config.model.repo_id is required")

    torch_dtype = model_cfg.get("torch_dtype") or "bfloat16"
    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))

    openonerec_dir = _ensure_openonerec_checkout(cfg)
    benchmarks_dir = os.path.join(openonerec_dir, "benchmarks")
    if not os.path.isdir(benchmarks_dir):
        raise FileNotFoundError(f"OpenOneRec benchmarks/ dir not found: {benchmarks_dir}")

    # Make upstream benchmarks importable BEFORE importing our generator adapter.
    if benchmarks_dir not in sys.path:
        sys.path.insert(0, benchmarks_dir)

    from benchmark import Benchmark  # type: ignore
    from projects.openonerec_recif_bench_eval.hf_generator import BenchmarkTransformersGenerator

    eval_cfg = cfg.get("eval", {}) or {}
    data_version = str(eval_cfg.get("data_version") or "v1.0")
    enable_thinking = bool(eval_cfg.get("enable_thinking", False))
    overwrite = bool(eval_cfg.get("overwrite", False))
    sample_size = eval_cfg.get("sample_size", None)
    disable_llm_judge = bool(eval_cfg.get("disable_llm_judge", False))

    data_dir = _ensure_recif_bench_data(cfg)
    if not disable_llm_judge:
        _patch_gemini_config(benchmarks_dir, cfg)

    output_root = str(eval_cfg.get("output_root") or os.path.join(REPO_ROOT, "workdirs", "openonerec_recif_eval"))
    run_name = str((cfg.get("project", {}) or {}).get("run_name") or f"recif_{Path(config_path).stem}")
    output_dir = os.path.join(output_root, "results", data_version, f"results_{run_name}")
    os.makedirs(output_dir, exist_ok=True)

    gen_cfg = cfg.get("generator", {}) or {}
    batch_size = int(gen_cfg.get("batch_size", 8))
    max_batch_size = int(gen_cfg.get("max_batch_size", 64))

    generator = BenchmarkTransformersGenerator(
        model_repo_id,
        torch_dtype=str(torch_dtype),
        trust_remote_code=trust_remote_code,
        prefer_tpu=True,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
    )

    # Match upstream eval_script.sh ordering + per-task overrides.
    task_runs: list[tuple[str, dict[str, Any]]] = [
        ("rec_reason", {"worker_batch_size": 5}),
        ("item_understand", {"worker_batch_size": 250}),
        ("ad", {"worker_batch_size": 1875, "num_beams": 32, "num_return_sequences": 32, "num_return_thinking_sequences": 1}),
        ("product", {"worker_batch_size": 1875, "num_beams": 32, "num_return_sequences": 32, "num_return_thinking_sequences": 1}),
        ("label_cond", {"worker_batch_size": 1875, "num_beams": 32, "num_return_sequences": 32, "num_return_thinking_sequences": 1}),
        ("video", {"worker_batch_size": 1875, "num_beams": 32, "num_return_sequences": 32, "num_return_thinking_sequences": 1}),
        ("interactive", {"worker_batch_size": 250, "num_beams": 32, "num_return_sequences": 32, "num_return_thinking_sequences": 1}),
        ("label_pred", {"worker_batch_size": 3200, "max_logprobs": 10000}),
    ]

    task_filter = eval_cfg.get("tasks")
    if task_filter is not None:
        wanted = {str(name) for name in task_filter}
        task_runs = [pair for pair in task_runs if pair[0] in wanted]

    # Optional config-driven per-task overrides (useful for local smoke or for tuning batch sizes
    # without editing code). By default, this is empty so behavior stays aligned with upstream.
    task_overrides = eval_cfg.get("task_overrides") or {}
    if not isinstance(task_overrides, dict):
        raise ValueError(f"eval.task_overrides must be a mapping, got: {type(task_overrides)}")

    if task_overrides:
        merged: list[tuple[str, dict[str, Any]]] = []
        for task_name, overrides in task_runs:
            extra = task_overrides.get(task_name)
            if extra is None:
                merged.append((task_name, overrides))
                continue
            if not isinstance(extra, dict):
                raise ValueError(f"eval.task_overrides[{task_name!r}] must be a mapping, got: {type(extra)}")
            merged.append((task_name, {**overrides, **extra}))
        task_runs = merged

    splits = ["test"]
    for task_name, overrides in task_runs:
        benchmark = Benchmark(
            model_path=model_repo_id,
            task_types=[task_name],
            splits=splits,
            data_dir=data_dir,
            enable_thinking=enable_thinking,
        )
        benchmark.run(
            generator=generator,
            output_dir=output_dir,
            overwrite=overwrite,
            enable_thinking=enable_thinking,
            sample_size=sample_size,
            **overrides,
        )
        for split in splits:
            _require_generated_file(
                output_dir,
                model_repo_id=model_repo_id,
                task_name=task_name,
                split=split,
            )

    eval_results_path = os.path.join(output_dir, "eval_results.json")
    Benchmark.evaluate_dev(
        generation_results_dir=output_dir,
        output_path=eval_results_path,
        data_dir=data_dir,
        overwrite=overwrite,
        wip_enabled=False if disable_llm_judge else None,
        llm_eval_enabled=False if disable_llm_judge else None,
        task_types=[pair[0] for pair in task_runs] if task_filter is not None else None,
    )

    with open(eval_results_path, "r", encoding="utf-8") as f:
        eval_results = json.load(f)

    # W&B logging (optional but required for TPU delivery per repo policy).
    project_cfg = cfg.get("project", {}) or {}
    wandb_project = str(project_cfg.get("wandb_project") or "openonerec-recif-bench")
    wandb_mode = str(project_cfg.get("wandb_mode") or os.environ.get("WANDB_MODE") or "online")

    wandb_required = bool(project_cfg.get("wandb_required", True))
    wandb_url = None
    if wandb_mode != "disabled":
        if wandb_required and not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError("WANDB_API_KEY is required when wandb_mode is not disabled.")

        try:
            import wandb

            run = wandb.init(
                project=wandb_project,
                name=run_name,
                mode=wandb_mode,
                config={"config_path": config_path, **cfg},
            )
            wandb_url = run.get_url()

            model_key = os.path.basename(model_repo_id.rstrip("/"))
            model_results = eval_results.get(model_key) or {}
            flat_metrics: dict[str, float] = {}
            for task, task_payload in model_results.items():
                if task.startswith("_") or not isinstance(task_payload, dict):
                    continue
                split_payload = task_payload.get("test") if isinstance(task_payload, dict) else None
                if not isinstance(split_payload, dict):
                    continue
                for k, v in split_payload.items():
                    if isinstance(v, (int, float)):
                        flat_metrics[f"{task}/{k}"] = float(v)
            if flat_metrics:
                run.log(flat_metrics)
            run.finish()
        except Exception as exc:
            if wandb_required:
                raise
            print(f"[warn] wandb logging failed: {exc}", file=sys.stderr)

    return {
        "model": model_repo_id,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "eval_results_path": eval_results_path,
        "wandb_url": wandb_url,
    }


def main() -> None:
    load_dotenv_if_present(repo_root=REPO_ROOT)

    parser = ArgumentParser(description="Run OpenOneRec RecIF-Bench eval on TPU from a YAML config.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    parser.add_argument("--print-config", action="store_true", help="Print the resolved config and exit.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_path = str(args.config or "<config>")

    if args.print_config:
        print(yaml.safe_dump(cfg, sort_keys=False))
        return

    result = _run_eval(cfg, config_path=config_path)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
