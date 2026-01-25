"""Multi-stage pipeline orchestration.

This package is intentionally lightweight: it provides a config-driven way to
run multiple stages (e.g. SFT → RL) sequentially, without forcing SFT/RL code to
live in one monolithic runner.
"""

from plugins.pipeline.runner import PipelineConfig, PipelineStage, run_pipeline

__all__ = ["PipelineConfig", "PipelineStage", "run_pipeline"]

