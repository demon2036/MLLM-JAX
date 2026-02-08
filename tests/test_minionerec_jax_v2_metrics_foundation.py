from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path


def _load_module(module_name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_metrics = _load_module("_minionerec_jax_v2_metrics_under_test", "projects/minionerec_jax_v2/metrics.py")

compute_hr_ndcg = _metrics.compute_hr_ndcg
normalize_sid_text = _metrics.normalize_sid_text


def test_normalize_sid_text_matches_official_cleaning_style() -> None:
    raw = '  "### Response:\n sid-123 \n trailing-text"  '
    assert normalize_sid_text(raw) == "sid-123"


def test_compute_hr_ndcg_matches_calc_style_math_and_invalid_counting() -> None:
    predictions = [
        ["### Response:\n sid-A\n", "sid-X", "sid-B"],
        [" sid-X ", ' "sid-B" ', "sid-C"],
        ["sid-X", "sid-Y", "sid-Z"],
    ]
    targets = ['"sid-A"', "### Response:\n sid-B ", "sid-D"]

    result = compute_hr_ndcg(
        predictions=predictions,
        targets=targets,
        topk=(3, 5, 1, 1, -4),
        valid_items={"sid-A", "sid-B", "sid-C", "sid-D"},
    )

    assert result.topk == [1, 3]
    assert math.isclose(result.hr[1], 1.0 / 3.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(result.hr[3], 2.0 / 3.0, rel_tol=0.0, abs_tol=1e-12)

    expected_ndcg_at_1 = 1.0 / 3.0
    expected_ndcg_at_3 = (1.0 + (1.0 / math.log2(3.0))) / 3.0
    assert math.isclose(result.ndcg[1], expected_ndcg_at_1, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(result.ndcg[3], expected_ndcg_at_3, rel_tol=0.0, abs_tol=1e-12)

    assert result.invalid_prediction_count == 4
    assert result.n_samples == 3
    assert result.n_beams == 3
