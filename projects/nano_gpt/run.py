from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from plugins.training.core.runtime.env import load_dotenv_if_present
from projects.nano_gpt.config import load_config
from projects.nano_gpt.runner import run_nano_gpt


def main() -> None:
    load_dotenv_if_present(repo_root=REPO_ROOT)

    parser = ArgumentParser(description="Run nanoGPT baseline (JAX/Flax) from a YAML config.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    parser.add_argument("--print-config", action="store_true", help="Print the resolved config and exit.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_path = str(args.config or "<default>")

    if args.print_config:
        print(yaml.safe_dump(cfg, sort_keys=False))
        return

    result = run_nano_gpt(cfg, config_path=config_path)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
