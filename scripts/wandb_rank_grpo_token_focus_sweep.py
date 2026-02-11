from __future__ import annotations

import argparse
import math
import os
from typing import Any


def _get_nested(d: Any, path: list[str]) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _as_float_or_nan(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        v = float(x)
        if math.isnan(v):
            return float("nan")
        return v
    except Exception:
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank GRPO token-focus sweep runs in W&B by a metric.")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY") or "johntitordemon2036")
    parser.add_argument("--project", default="mllm-jax-grpo-gsm8k-tokenfocus-sweep")
    parser.add_argument("--metric", default="eval/accuracy/pass_at_1")
    parser.add_argument("--filter-config-substr", default="projects/gsm8k_grpo/configs/token_focus_sweep/")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    import wandb  # type: ignore

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}")

    rows: list[dict[str, Any]] = []
    for run in runs:
        cfg = dict(getattr(run, "config", {}) or {})
        config_path = str(cfg.get("config_path") or "")
        if args.filter_config_substr and args.filter_config_substr not in config_path:
            continue

        token_focus = _get_nested(cfg, ["algo", "update", "kwargs", "token_focus"]) or {}
        p = _get_nested(token_focus, ["prob_threshold"])
        k = _get_nested(token_focus, ["max_tokens_per_sequence"])

        summary = dict(getattr(run, "summary", {}) or {})
        metric_value = summary.get(args.metric)

        rows.append(
            {
                "name": getattr(run, "name", ""),
                "id": getattr(run, "id", ""),
                "state": getattr(run, "state", ""),
                "config_path": config_path,
                "p": p,
                "k": k,
                "metric": _as_float_or_nan(metric_value),
                "eval_reward_mean": _as_float_or_nan(summary.get("eval/reward/total/mean")),
                "selected_frac": _as_float_or_nan(summary.get("token_focus/selected_fraction")),
                "eligible_frac": _as_float_or_nan(summary.get("token_focus/eligible_fraction")),
                "empty_seq_frac": _as_float_or_nan(summary.get("token_focus/empty_seq_fraction")),
                "url": getattr(run, "url", ""),
            }
        )

    rows.sort(key=lambda r: (float("-inf") if math.isnan(r["metric"]) else r["metric"]), reverse=True)

    print(f"entity={args.entity} project={args.project} metric={args.metric}")
    print(f"runs_filtered={len(rows)} (config_path contains {args.filter_config_substr!r})")
    print()
    header = [
        "rank",
        "metric",
        "p",
        "k",
        "eval_reward",
        "sel_frac",
        "elig_frac",
        "empty_frac",
        "state",
        "id",
        "name",
    ]
    print("\t".join(header))

    for i, r in enumerate(rows[: int(args.limit)]):
        print(
            "\t".join(
                [
                    str(i + 1),
                    f"{r['metric']:.4f}" if not math.isnan(r["metric"]) else "nan",
                    str(r["p"]),
                    str(r["k"]),
                    f"{r['eval_reward_mean']:.4f}" if not math.isnan(r["eval_reward_mean"]) else "nan",
                    f"{r['selected_frac']:.4f}" if not math.isnan(r["selected_frac"]) else "nan",
                    f"{r['eligible_frac']:.4f}" if not math.isnan(r["eligible_frac"]) else "nan",
                    f"{r['empty_seq_frac']:.4f}" if not math.isnan(r["empty_seq_frac"]) else "nan",
                    str(r["state"]),
                    str(r["id"]),
                    str(r["name"]),
                ]
            )
        )
        if r["url"]:
            print(f"  url={r['url']}")


if __name__ == "__main__":
    main()

