from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


def normalize_sid_text(text: str) -> str:
    value = str(text)
    if "Response:\n" in value:
        value = value.split("Response:\n")[-1]
    value = value.strip().strip('"').strip()
    if "\n" in value:
        value = value.split("\n", 1)[0].strip()
    value = "".join(value.split())
    return value


@dataclass(frozen=True)
class RankingMetrics:
    hr: dict[int, float]
    ndcg: dict[int, float]
    topk: list[int]
    n_samples: int
    n_beams: int
    invalid_prediction_count: int


def compute_hr_ndcg(
    *,
    predictions: list[list[str]],
    targets: list[str],
    topk: Iterable[int] = (3, 5, 10),
    valid_items: set[str] | None = None,
) -> RankingMetrics:
    if len(predictions) != len(targets):
        raise ValueError(f"predictions and targets must have same length, got {len(predictions)} vs {len(targets)}")
    n_samples = len(predictions)
    if n_samples <= 0:
        raise ValueError("Empty predictions")

    n_beams = min(len(items) for items in predictions)
    if n_beams <= 0:
        raise ValueError("predictions must contain at least one beam per sample")

    topk_list = sorted({int(k) for k in topk if int(k) > 0 and int(k) <= int(n_beams)})
    if not topk_list:
        raise ValueError(f"No valid topk <= n_beams={n_beams}: {list(topk)}")

    sum_hr = {k: 0.0 for k in topk_list}
    sum_ndcg = {k: 0.0 for k in topk_list}
    invalid_prediction_count = 0

    for predicted_items, target in zip(predictions, targets, strict=True):
        preds_norm = [normalize_sid_text(item) for item in predicted_items]
        target_norm = normalize_sid_text(target)

        min_index = None
        for idx, candidate in enumerate(preds_norm):
            if valid_items is not None and candidate not in valid_items:
                invalid_prediction_count += 1
            if candidate == target_norm:
                min_index = idx
                break

        if min_index is None:
            continue

        for k in topk_list:
            if min_index < int(k):
                sum_hr[k] += 1.0
                sum_ndcg[k] += 1.0 / math.log(min_index + 2)

    hr = {k: float(sum_hr[k] / n_samples) for k in topk_list}
    ndcg = {k: float((sum_ndcg[k] / n_samples) * math.log(2)) for k in topk_list}
    return RankingMetrics(
        hr=hr,
        ndcg=ndcg,
        topk=topk_list,
        n_samples=int(n_samples),
        n_beams=int(n_beams),
        invalid_prediction_count=int(invalid_prediction_count),
    )


__all__ = ["RankingMetrics", "compute_hr_ndcg", "normalize_sid_text"]
