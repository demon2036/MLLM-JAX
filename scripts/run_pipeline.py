from __future__ import annotations

import os
import sys
from argparse import ArgumentParser

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from plugins.common.env import load_dotenv_if_present
from plugins.pipeline.runner import load_pipeline_config, run_pipeline


def main() -> None:
    load_dotenv_if_present(repo_root=REPO_ROOT)
    os.chdir(REPO_ROOT)

    parser = ArgumentParser(description="Run a multi-stage pipeline (SFT → RL, etc) from a YAML config.")
    parser.add_argument("--config", required=True, help="Pipeline YAML path.")
    args = parser.parse_args()

    cfg = load_pipeline_config(str(args.config))
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
