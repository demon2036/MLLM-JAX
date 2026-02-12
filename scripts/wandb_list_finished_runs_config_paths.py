from __future__ import annotations

import argparse
import os
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List config_path values for finished W&B runs (optionally filtered by substring)."
    )
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY") or "johntitordemon2036")
    parser.add_argument("--project", required=True)
    parser.add_argument("--filter-config-substr", default="")
    parser.add_argument("--limit", type=int, default=2000)
    args = parser.parse_args()

    import wandb  # type: ignore

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}")

    out: list[str] = []
    count = 0
    for run in runs:
        if count >= int(args.limit):
            break
        count += 1
        state = str(getattr(run, "state", "") or "")
        if state != "finished":
            continue
        cfg = dict(getattr(run, "config", {}) or {})
        config_path = str(cfg.get("config_path") or "")
        if not config_path:
            continue
        if args.filter_config_substr and args.filter_config_substr not in config_path:
            continue
        out.append(config_path)

    out = sorted(set(out))
    for path in out:
        print(path)


if __name__ == "__main__":
    main()

