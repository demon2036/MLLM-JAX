from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


def normalize_sid_text(text: str) -> str:
    s = str(text)
    if "Response:\n" in s:
        s = s.split("Response:\n")[-1]
    s = s.strip().strip('"').strip()
    if "\n" in s:
        s = s.split("\n", 1)[0].strip()
    # Constrained decoding should emit only SID tokens; tolerate whitespace-only
    # differences (some tokenizers may insert spaces between added tokens).
    s = "".join(s.split())
    return s


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
    topk: Iterable[int] = (1, 3, 5, 10, 20, 50),
    valid_items: set[str] | None = None,
) -> RankingMetrics:
    if len(predictions) != len(targets):
        raise ValueError(f"predictions and targets must have same length, got {len(predictions)} vs {len(targets)}")
    n_samples = len(predictions)
    if n_samples == 0:
        raise ValueError("Empty predictions")

    n_beams = min(len(p) for p in predictions) if predictions else 0
    if n_beams <= 0:
        raise ValueError("predictions must contain at least one beam per sample")

    topk_list = sorted({int(k) for k in topk if int(k) > 0 and int(k) <= n_beams})
    if not topk_list:
        raise ValueError(f"No valid topk <= n_beams={n_beams}: {list(topk)}")

    sum_hr = {k: 0.0 for k in topk_list}
    sum_ndcg = {k: 0.0 for k in topk_list}
    invalid_count = 0

    for preds, target in zip(predictions, targets, strict=True):
        preds_norm = [normalize_sid_text(p) for p in preds]
        target_norm = normalize_sid_text(target)

        min_idx = None
        for i, p in enumerate(preds_norm):
            if valid_items is not None and p not in valid_items:
                invalid_count += 1
            if p == target_norm:
                min_idx = i
                break

        if min_idx is None:
            continue

        for k in topk_list:
            if min_idx < k:
                sum_hr[k] += 1.0
                sum_ndcg[k] += 1.0 / math.log(min_idx + 2)

    # Align with MiniOneRec `calc.py`:
    # NDCG = (avg_dcg) / (1/log(2)) = avg_dcg * log(2)
    hr = {k: (sum_hr[k] / n_samples) for k in topk_list}
    ndcg = {k: ((sum_ndcg[k] / n_samples) * math.log(2)) for k in topk_list}
    return RankingMetrics(
        hr=hr,
        ndcg=ndcg,
        topk=topk_list,
        n_samples=n_samples,
        n_beams=n_beams,
        invalid_prediction_count=invalid_count,
    )
