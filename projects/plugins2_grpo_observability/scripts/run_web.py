from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

from prompts.prompts import system_prompt as default_system_prompt

from plugins2.grpo_observability import (
    GRPOObservabilityEngine,
    RunRequest,
    load_plugins2_config,
    serve_observability_http,
)



def _default_wandb_name(config_path: str) -> str:
    tag = Path(config_path).stem
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return f"plugins2_{tag}_{timestamp}"



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run plugins2 GRPO observability backend/web on TPU.")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--once", action="store_true", help="Run one request and print JSON response")
    parser.add_argument("--system-prompt", default=default_system_prompt, help="System prompt for --once")
    parser.add_argument(
        "--user-prompt",
        default="If Alice has 3 apples and buys 5 more, how many apples does she have?",
        help="User prompt for --once",
    )
    parser.add_argument("--label", default="8", help="Label/ground-truth answer for --once")
    parser.add_argument("--k", type=int, default=None, help="Override rollout.k for this request")
    parser.add_argument("--host", default=None, help="Override web host")
    parser.add_argument("--port", type=int, default=None, help="Override web port")
    parser.add_argument(
        "--html-template",
        default=str(Path(__file__).resolve().parents[1] / "web" / "index.html"),
        help="Path to frontend HTML template",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    cfg = load_plugins2_config(args.config)
    if cfg.wandb.name is None and cfg.wandb.mode != "disabled":
        cfg = replace(cfg, wandb=replace(cfg.wandb, name=_default_wandb_name(cfg.config_path)))

    engine = GRPOObservabilityEngine(cfg)

    if args.once:
        request = RunRequest(
            system_prompt=str(args.system_prompt),
            user_prompt=str(args.user_prompt),
            label=str(args.label),
            k=args.k,
        )
        output = engine.run_request(request)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    host = args.host or cfg.web.host
    port = int(args.port or cfg.web.port)

    serve_observability_http(
        engine=engine,
        host=host,
        port=port,
        html_template_path=args.html_template,
    )


if __name__ == "__main__":
    main()
