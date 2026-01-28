import math

import pytest

from projects.minionerec.sft.metrics import compute_hr_ndcg, normalize_sid_text


def test_normalize_sid_text_strips_common_noise():
    assert normalize_sid_text('  "<a_1><b_2><c_3>"\n') == "<a_1><b_2><c_3>"
    assert normalize_sid_text("### Response:\n<a_1><b_2><c_3>\n") == "<a_1><b_2><c_3>"


def test_compute_hr_ndcg_single_match():
    metrics = compute_hr_ndcg(
        predictions=[["A", "B", "C", "D", "E"]],
        targets=["C"],
        topk=[1, 3, 5],
        valid_items={"A", "B", "C", "D", "E"},
    )
    assert metrics.n_samples == 1
    assert metrics.n_beams == 5
    assert metrics.hr[1] == 0.0
    assert metrics.hr[3] == 1.0
    assert metrics.hr[5] == 1.0
    assert metrics.ndcg[3] == pytest.approx(math.log(2) / math.log(4))


def test_compute_hr_ndcg_no_match():
    metrics = compute_hr_ndcg(
        predictions=[["A", "B", "C"]],
        targets=["Z"],
        topk=[1, 3],
        valid_items={"A", "B", "C"},
    )
    assert metrics.hr[1] == 0.0
    assert metrics.hr[3] == 0.0
    assert metrics.ndcg[1] == 0.0
    assert metrics.ndcg[3] == 0.0


def test_compute_hr_ndcg_counts_invalid_predictions():
    metrics = compute_hr_ndcg(
        predictions=[["X", "C"]],
        targets=["C"],
        topk=[1, 2],
        valid_items={"C"},
    )
    assert metrics.invalid_prediction_count == 1


def test_compute_hr_ndcg_raises_on_shape_mismatch():
    with pytest.raises(ValueError):
        compute_hr_ndcg(predictions=[["A"]], targets=["A", "B"])

