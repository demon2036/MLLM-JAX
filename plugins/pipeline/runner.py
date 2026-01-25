from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass(frozen=True)
class PipelineStage:
    """One stage in a multi-stage pipeline.

    For now, stages are executed as subprocess commands so each stage can own
    its JAX runtime and release memory between stages.
    """

    name: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None


@dataclass(frozen=True)
class PipelineConfig:
    config_path: str
    output_dir: str
    stages: list[PipelineStage]


def _as_command(x: Any) -> list[str]:
    if isinstance(x, list) and all(isinstance(i, str) for i in x):
        return list(x)
    if isinstance(x, str) and x.strip():
        # Run via bash -lc for string commands.
        return ["bash", "-lc", x]
    raise TypeError("stage.command must be a non-empty string or list[str]")


def load_pipeline_config(config_path: str) -> PipelineConfig:
    config_path = str(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError("Pipeline YAML root must be a dict")

    run = data.get("run") or {}
    if not isinstance(run, dict):
        raise TypeError("run must be a dict")
    output_dir = str(run.get("output_dir") or "runs/pipeline")

    stages_raw = data.get("stages") or []
    if not isinstance(stages_raw, list) or not stages_raw:
        raise ValueError("stages must be a non-empty list")

    stages: list[PipelineStage] = []
    for i, s in enumerate(stages_raw):
        if not isinstance(s, dict):
            raise TypeError(f"stages[{i}] must be a dict")
        name = str(s.get("name") or f"stage{i}")
        command = _as_command(s.get("command"))
        env = s.get("env") or {}
        if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
            raise TypeError(f"stages[{i}].env must be a dict[str,str]")
        cwd = s.get("cwd")
        if cwd is not None:
            cwd = str(cwd)
        stages.append(PipelineStage(name=name, command=command, env=dict(env), cwd=cwd))

    return PipelineConfig(config_path=config_path, output_dir=output_dir, stages=stages)


def run_pipeline(cfg: PipelineConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Record the pipeline config path for traceability.
    meta_path = os.path.join(cfg.output_dir, "pipeline_meta.yaml")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(
            yaml.safe_dump(
                {
                    "config_path": cfg.config_path,
                    "output_dir": cfg.output_dir,
                    "stages": [{"name": s.name, "command": s.command, "cwd": s.cwd, "env": s.env} for s in cfg.stages],
                },
                sort_keys=False,
            )
        )

    base_env = dict(os.environ)
    base_env.setdefault("PIPELINE_CONFIG_PATH", os.path.abspath(cfg.config_path))
    base_env.setdefault("PIPELINE_OUTPUT_DIR", os.path.abspath(cfg.output_dir))

    for idx, stage in enumerate(cfg.stages):
        t0 = time.perf_counter()
        print(f"[pipeline] start stage={idx} name={stage.name} cmd={stage.command!r}")
        env = dict(base_env)
        env.update(stage.env)
        env["PIPELINE_STAGE_NAME"] = str(stage.name)
        env["PIPELINE_STAGE_INDEX"] = str(idx)

        proc = subprocess.run(
            stage.command,
            cwd=stage.cwd,
            env=env,
            check=False,
        )
        dt = time.perf_counter() - t0
        print(f"[pipeline] done stage={idx} name={stage.name} rc={proc.returncode} dt={dt:.1f}s")
        if proc.returncode != 0:
            raise RuntimeError(f"Stage failed: {stage.name} (rc={proc.returncode})")


__all__ = ["PipelineConfig", "PipelineStage", "load_pipeline_config", "run_pipeline"]

