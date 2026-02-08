from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser
from dataclasses import asdict

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from plugins.training.core.runtime.env import load_dotenv_if_present
from projects.minionerec_jax_v2.config import config_from_dict, load_config
from projects.minionerec_jax_v2.runner import run_official_eval


def main() -> None:
    load_dotenv_if_present(repo_root=REPO_ROOT)

    parser = ArgumentParser(description="Run MiniOneRec JAX v2 official eval.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    parser.add_argument("--print-config", action="store_true", help="Print the resolved config and exit.")
    parser.add_argument("--run-tag", type=str, default=None, help="Optional run tag appended to metadata and W&B name.")
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    config_path = str(args.config or "<default>")
    cfg = config_from_dict(cfg_dict, config_path=config_path)

    if args.print_config:
        print(yaml.safe_dump(json.loads(json.dumps(asdict(cfg))), sort_keys=False))
        return

    os.makedirs(cfg.runtime.output_dir, exist_ok=True)
    result = run_official_eval(cfg, run_tag=args.run_tag)
    summary_path = os.path.join(cfg.runtime.output_dir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"summary_json={summary_path}")


if __name__ == "__main__":
    main()
