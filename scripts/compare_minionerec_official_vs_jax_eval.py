from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from projects.sid_sft.metrics import compute_hr_ndcg, normalize_sid_text
from projects.sid_sft.sid_utils import load_valid_sids_from_info


def _parse_topk_csv(value: str) -> list[int]:
    parts = [p.strip() for p in str(value).split(",")]
    out: list[int] = []
    for p in parts:
        if not p:
            continue
        k = int(p)
        if k <= 0:
            raise ValueError(f"--topk must contain positive ints, got: {value!r}")
        out.append(k)
    if not out:
        raise ValueError(f"--topk parsed to empty list from: {value!r}")
    return out


def _load_json_records(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        records: list[dict[str, Any]] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSONL at {path}:{line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object per line in {path}:{line_no}, got {type(obj).__name__}")
            records.append(obj)
        payload = records

    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list at {path}, got {type(payload).__name__}")
    out: list[dict[str, Any]] = []
    for i, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"Expected list[dict] at {path}[{i}], got {type(row).__name__}")
        out.append(row)
    return out


def _get_output(row: dict[str, Any], *, path: str, idx: int) -> str:
    if "output" not in row:
        raise ValueError(f"Missing key 'output' in {path}[{idx}]")
    return str(row["output"])


def _get_predict(row: dict[str, Any], *, path: str, idx: int) -> list[str]:
    if "predict" not in row:
        raise ValueError(f"Missing key 'predict' in {path}[{idx}]")
    preds = row["predict"]
    if not isinstance(preds, list):
        raise ValueError(f"Expected 'predict' to be a list in {path}[{idx}], got {type(preds).__name__}")
    return [str(p) for p in preds]


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.fmean(values))


def _summarize_int(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "min": None, "max": None}
    values_sorted = sorted(values)
    p50 = values_sorted[len(values_sorted) // 2]
    return {
        "count": int(len(values_sorted)),
        "mean": float(statistics.fmean(values_sorted)),
        "p50": int(p50),
        "min": int(values_sorted[0]),
        "max": int(values_sorted[-1]),
    }


def _rank_in_preds(*, preds_norm: list[str], target_norm: str) -> int | None:
    for i, p in enumerate(preds_norm):
        if p == target_norm:
            return i
    return None


def _build_report(
    *,
    torch_path: str,
    jax_path: str,
    topk: list[int],
    max_samples: int | None,
    info_file: str | None,
) -> dict[str, Any]:
    torch_rows = _load_json_records(torch_path)
    jax_rows = _load_json_records(jax_path)

    report: dict[str, Any] = {
        "inputs": {
            "torch_json": str(torch_path),
            "jax_json": str(jax_path),
            "info_file": str(info_file) if info_file else None,
            "topk": [int(k) for k in topk],
            "max_samples": int(max_samples) if max_samples is not None else None,
        }
    }

    if len(torch_rows) != len(jax_rows):
        report["alignment"] = {
            "aligned": False,
            "reason": "length_mismatch",
            "torch_len": int(len(torch_rows)),
            "jax_len": int(len(jax_rows)),
        }
        return report

    n_total = len(torch_rows)
    n = n_total if max_samples is None else min(int(max_samples), n_total)
    torch_rows = torch_rows[:n]
    jax_rows = jax_rows[:n]

    output_mismatch: list[int] = []
    for i, (t, j) in enumerate(zip(torch_rows, jax_rows, strict=True)):
        t_out_norm = normalize_sid_text(_get_output(t, path=torch_path, idx=i))
        j_out_norm = normalize_sid_text(_get_output(j, path=jax_path, idx=i))
        if t_out_norm != j_out_norm:
            output_mismatch.append(i)

    report["alignment"] = {
        "aligned": len(output_mismatch) == 0,
        "reason": None if not output_mismatch else "output_mismatch",
        "n_samples": int(n),
        "output_mismatch_indices": output_mismatch,
    }
    if output_mismatch:
        return report

    targets = [_get_output(r, path=torch_path, idx=i) for i, r in enumerate(torch_rows)]
    torch_predictions = [_get_predict(r, path=torch_path, idx=i) for i, r in enumerate(torch_rows)]
    jax_predictions = [_get_predict(r, path=jax_path, idx=i) for i, r in enumerate(jax_rows)]

    beam_len_mismatch: list[int] = []
    for i, (tp, jp) in enumerate(zip(torch_predictions, jax_predictions, strict=True)):
        if len(tp) != len(jp):
            beam_len_mismatch.append(i)

    torch_preds_norm = [[normalize_sid_text(p) for p in preds] for preds in torch_predictions]
    jax_preds_norm = [[normalize_sid_text(p) for p in preds] for preds in jax_predictions]
    targets_norm = [normalize_sid_text(t) for t in targets]

    top1_mismatch: list[int] = []
    full_list_mismatch: list[int] = []
    for i, (tp, jp) in enumerate(zip(torch_preds_norm, jax_preds_norm, strict=True)):
        if not tp or not jp:
            top1_mismatch.append(i)
            full_list_mismatch.append(i)
            continue
        if tp[0] != jp[0]:
            top1_mismatch.append(i)
        if tp != jp:
            full_list_mismatch.append(i)

    overlap_intersection_sizes: list[int] = []
    overlap_only_torch_sizes: list[int] = []
    overlap_only_jax_sizes: list[int] = []
    overlap_jaccard: list[float] = []
    for tp, jp in zip(torch_preds_norm, jax_preds_norm, strict=True):
        t_set = set(tp)
        j_set = set(jp)
        inter = t_set & j_set
        only_t = t_set - j_set
        only_j = j_set - t_set
        overlap_intersection_sizes.append(len(inter))
        overlap_only_torch_sizes.append(len(only_t))
        overlap_only_jax_sizes.append(len(only_j))
        union = t_set | j_set
        overlap_jaccard.append((len(inter) / len(union)) if union else 1.0)

    torch_ranks: list[int | None] = []
    jax_ranks: list[int | None] = []
    for tp, jp, target_norm in zip(torch_preds_norm, jax_preds_norm, targets_norm, strict=True):
        torch_ranks.append(_rank_in_preds(preds_norm=tp, target_norm=target_norm))
        jax_ranks.append(_rank_in_preds(preds_norm=jp, target_norm=target_norm))

    rank_mismatch: list[int] = []
    target_missing_torch: list[int] = []
    target_missing_jax: list[int] = []
    rank_deltas: list[int] = []
    for i, (tr, jr) in enumerate(zip(torch_ranks, jax_ranks, strict=True)):
        if tr is None:
            target_missing_torch.append(i)
        if jr is None:
            target_missing_jax.append(i)
        if tr != jr:
            rank_mismatch.append(i)
        if tr is not None and jr is not None:
            rank_deltas.append(int(jr) - int(tr))

    valid_items: set[str] | None = None
    if info_file:
        valid_items = set(load_valid_sids_from_info(info_file))

    torch_metrics = compute_hr_ndcg(predictions=torch_predictions, targets=targets, topk=topk, valid_items=valid_items)
    jax_metrics = compute_hr_ndcg(predictions=jax_predictions, targets=targets, topk=topk, valid_items=valid_items)
    topk_effective = [int(k) for k in torch_metrics.topk]

    in_topk: dict[int, dict[str, int]] = {}
    for k in topk_effective:
        torch_count = 0
        jax_count = 0
        both_count = 0
        only_torch_count = 0
        only_jax_count = 0
        for tr, jr in zip(torch_ranks, jax_ranks, strict=True):
            t_hit = (tr is not None) and (int(tr) < int(k))
            j_hit = (jr is not None) and (int(jr) < int(k))
            if t_hit:
                torch_count += 1
            if j_hit:
                jax_count += 1
            if t_hit and j_hit:
                both_count += 1
            elif t_hit and not j_hit:
                only_torch_count += 1
            elif j_hit and not t_hit:
                only_jax_count += 1
        in_topk[int(k)] = {
            "torch_count": int(torch_count),
            "jax_count": int(jax_count),
            "both_count": int(both_count),
            "only_torch_count": int(only_torch_count),
            "only_jax_count": int(only_jax_count),
        }

    max_k = max(topk_effective) if topk_effective else None
    rank_deltas_within_maxk: list[int] = []
    if max_k is not None:
        for tr, jr in zip(torch_ranks, jax_ranks, strict=True):
            if tr is None or jr is None:
                continue
            if int(tr) < int(max_k) and int(jr) < int(max_k):
                rank_deltas_within_maxk.append(int(jr) - int(tr))

    report["summary"] = {
        "n_samples": int(n),
        "torch_beams_min": int(min((len(p) for p in torch_predictions), default=0)),
        "jax_beams_min": int(min((len(p) for p in jax_predictions), default=0)),
        "top1_match": {
            "count": int(n - len(top1_mismatch)),
            "rate": (float(n - len(top1_mismatch)) / float(n)) if n else 0.0,
        },
        "full_list_match": {
            "count": int(n - len(full_list_mismatch)),
            "rate": (float(n - len(full_list_mismatch)) / float(n)) if n else 0.0,
        },
        "overlap": {
            "intersection": _summarize_int(overlap_intersection_sizes),
            "only_torch": _summarize_int(overlap_only_torch_sizes),
            "only_jax": _summarize_int(overlap_only_jax_sizes),
            "jaccard_mean": _safe_mean(overlap_jaccard),
        },
        "rank": {
            "both_found_count": int(sum((tr is not None) and (jr is not None) for tr, jr in zip(torch_ranks, jax_ranks, strict=True))),
            "torch_found_count": int(sum(tr is not None for tr in torch_ranks)),
            "jax_found_count": int(sum(jr is not None for jr in jax_ranks)),
            "delta": _summarize_int(rank_deltas),
            "delta_within_maxk": _summarize_int(rank_deltas_within_maxk),
            "in_topk": in_topk,
        },
        "hr_ndcg": {
            "torch": asdict(torch_metrics),
            "jax": asdict(jax_metrics),
        },
    }

    report["mismatch_indices"] = {
        "beam_len": beam_len_mismatch,
        "top1": top1_mismatch,
        "full_list": full_list_mismatch,
        "rank": rank_mismatch,
        "target_missing_torch": target_missing_torch,
        "target_missing_jax": target_missing_jax,
    }
    return report


def _print_summary(report: dict[str, Any]) -> None:
    inputs = report.get("inputs", {})
    alignment = report.get("alignment", {})

    print("MiniOneRec official vs JAX eval diff")
    print(f"- torch_json: {inputs.get('torch_json')}")
    print(f"- jax_json:   {inputs.get('jax_json')}")
    if inputs.get("info_file"):
        print(f"- info_file:  {inputs.get('info_file')}")

    if not alignment.get("aligned"):
        print(f"- aligned:    false ({alignment.get('reason')})")
        if alignment.get("reason") == "length_mismatch":
            print(f"- torch_len:  {alignment.get('torch_len')}")
            print(f"- jax_len:    {alignment.get('jax_len')}")
        if alignment.get("reason") == "output_mismatch":
            mism = alignment.get("output_mismatch_indices") or []
            first = mism[0] if mism else None
            print(f"- output_mismatch: {len(mism)} (first={first})")
        return

    summary = report.get("summary", {})
    mism = report.get("mismatch_indices", {})

    n_samples = int(summary.get("n_samples") or 0)
    print(f"- n_samples:  {n_samples}")
    print(f"- beams_min:  torch={summary.get('torch_beams_min')} jax={summary.get('jax_beams_min')}")

    top1 = summary.get("top1_match", {})
    full = summary.get("full_list_match", {})
    top1_rate = float(top1.get("rate") or 0.0)
    full_rate = float(full.get("rate") or 0.0)
    print(f"- top1:       {top1.get('count')}/{n_samples} ({top1_rate:.4f})")
    print(f"- full_list:  {full.get('count')}/{n_samples} ({full_rate:.4f})")

    overlap = summary.get("overlap", {})
    inter_mean = ((overlap.get("intersection") or {}).get("mean"))
    jacc_mean = overlap.get("jaccard_mean")
    if inter_mean is not None and jacc_mean is not None:
        print(f"- overlap:    intersection_mean={float(inter_mean):.2f} jaccard_mean={float(jacc_mean):.4f}")

    rank = summary.get("rank", {})
    delta_mean = ((rank.get("delta") or {}).get("mean"))
    delta_p50 = ((rank.get("delta") or {}).get("p50")
    )
    if delta_mean is not None:
        print(f"- rank_delta: mean={float(delta_mean):.3f} p50={delta_p50}")

    hr_ndcg = summary.get("hr_ndcg", {})
    torch_m = hr_ndcg.get("torch") or {}
    jax_m = hr_ndcg.get("jax") or {}
    topk_effective = torch_m.get("topk") or []
    if topk_effective:
        k_max = int(max(topk_effective))
        torch_hr = (torch_m.get("hr") or {}).get(k_max)
        torch_ndcg = (torch_m.get("ndcg") or {}).get(k_max)
        jax_hr = (jax_m.get("hr") or {}).get(k_max)
        jax_ndcg = (jax_m.get("ndcg") or {}).get(k_max)
        if torch_hr is not None and jax_hr is not None:
            print(f"- hr@{k_max}:     torch={float(torch_hr):.6f} jax={float(jax_hr):.6f}")
        if torch_ndcg is not None and jax_ndcg is not None:
            print(f"- ndcg@{k_max}:   torch={float(torch_ndcg):.6f} jax={float(jax_ndcg):.6f}")

    top1_mism = mism.get("top1") or []
    full_mism = mism.get("full_list") or []
    if top1_mism:
        print(f"- top1_mismatch_indices: {len(top1_mism)} (first={top1_mism[0]})")
    if full_mism:
        print(f"- full_list_mismatch_indices: {len(full_mism)} (first={full_mism[0]})")


def _write_json(path: str, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically compare MiniOneRec official evaluate.py JSON outputs vs this repo's JAX eval_predictions.json. "
            "Validates row alignment by normalized 'output' and reports ranking diffs + HR/NDCG@K."
        )
    )
    parser.add_argument("--torch-json", required=True, help="Path to MiniOneRec official torch evaluate.py output JSON")
    parser.add_argument("--jax-json", required=True, help="Path to our JAX eval_predictions.json")
    parser.add_argument(
        "--info-file",
        default=None,
        help=(
            "Optional MiniOneRec info .txt path (sid\\t...). When provided, counts invalid predictions via "
            "projects.sid_sft.sid_utils.load_valid_sids_from_info + compute_hr_ndcg(valid_items=...)."
        ),
    )
    parser.add_argument(
        "--topk",
        type=_parse_topk_csv,
        default=_parse_topk_csv("1,3,5,10,20,50"),
        help="Comma-separated list of K values for HR/NDCG + rank stats (default: 1,3,5,10,20,50)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Compare only the first N samples (default: all)",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Optional path to write a machine-readable JSON report",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    max_samples: int | None = None
    if args.max_samples is not None:
        if int(args.max_samples) <= 0:
            raise ValueError("--max-samples must be >= 1")
        max_samples = int(args.max_samples)

    report = _build_report(
        torch_path=str(args.torch_json),
        jax_path=str(args.jax_json),
        topk=list(args.topk),
        max_samples=max_samples,
        info_file=(str(args.info_file) if args.info_file else None),
    )

    _print_summary(report)
    if args.out_json:
        _write_json(str(args.out_json), report)

    alignment = report.get("alignment", {})
    return 0 if alignment.get("aligned") else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(2)
