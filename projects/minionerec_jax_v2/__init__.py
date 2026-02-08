"""MiniOneRec-JAX v2 package."""

from typing import Any

from .config import (
    DEFAULT_CONFIG,
    MiniOneRecCheckpointConfig,
    MiniOneRecDatasetConfig,
    MiniOneRecDecodeConfig,
    MiniOneRecEvalConfig,
    MiniOneRecJaxConfig,
    MiniOneRecJaxV2Config,
    MiniOneRecRuntimeConfig,
    MiniOneRecWandbConfig,
    config_from_dict,
    load_config,
)
from .datasets import OFFICIAL_INSTRUCTION, OfficialEvalSidDataset, build_official_eval_prompt
from .evaluator import SidNextItemJaxEvaluator, evaluate_sid_next_item_jax
from .metrics import RankingMetrics, compute_hr_ndcg, normalize_sid_text


def run_official_eval(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .runner import run_official_eval as _run_official_eval

    return _run_official_eval(*args, **kwargs)


def run_official_eval_once(*args: Any, **kwargs: Any) -> tuple[list[list[str]], Any]:
    from .runner import run_official_eval_once as _run_official_eval_once

    return _run_official_eval_once(*args, **kwargs)

__all__ = [
    "DEFAULT_CONFIG",
    "MiniOneRecCheckpointConfig",
    "MiniOneRecDatasetConfig",
    "MiniOneRecDecodeConfig",
    "MiniOneRecEvalConfig",
    "MiniOneRecJaxConfig",
    "MiniOneRecJaxV2Config",
    "MiniOneRecRuntimeConfig",
    "MiniOneRecWandbConfig",
    "OFFICIAL_INSTRUCTION",
    "OfficialEvalSidDataset",
    "RankingMetrics",
    "SidNextItemJaxEvaluator",
    "build_official_eval_prompt",
    "compute_hr_ndcg",
    "config_from_dict",
    "evaluate_sid_next_item_jax",
    "load_config",
    "normalize_sid_text",
    "run_official_eval",
    "run_official_eval_once",
]
