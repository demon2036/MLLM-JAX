from __future__ import annotations

import os
import sys
import time
from argparse import ArgumentParser
from dataclasses import asdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from plugins.training.runner import GRPOGsm8kConfig, run_grpo_gsm8k


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def _get_env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    return float(value)


def _load_dotenv_if_present() -> None:
    candidates = [
        os.path.join(REPO_ROOT, ".env"),
        "/root/.env",
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                os.environ.setdefault(k, v)
        return


def main() -> None:
    parser = ArgumentParser(description="Run GRPO/GSM8K training via plugins/training runner.")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved config and exit (no JAX required).",
    )
    args = parser.parse_args()

    _load_dotenv_if_present()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if os.environ.get("WANDB_MODE") is None and os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_MODE"] = "online"

    model_path = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
    steps = _get_env_int("STEPS", 20)
    batch_size = _get_env_int("BATCH_SIZE", 1)
    num_pre_q = _get_env_int("NUM_PRE_Q", 8)
    global_length = _get_env_int("GLOBAL_LENGTH", 512)
    max_length_sample = _get_env_int("MAX_LENGTH_SAMPLE", 64)
    max_length_total = _get_env_int("MAX_LENGTH_TOTAL", max_length_sample + 128)
    ppo_epochs = _get_env_int("PPO_EPOCHS", 1)
    grad_accum_steps = _get_env_int("GRAD_ACCUM_STEPS", 1)
    beta = _get_env_float("BETA", 0.0)
    mesh_shape = os.environ.get("MESH_SHAPE_FSDP", "1,-1,1")

    wandb_project = os.environ.get("WANDB_PROJECT", "mllm-jax-grpo-gsm8k")
    wandb_name = os.environ.get(
        "WANDB_NAME",
        f"grpo_gsm8k_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_steps{steps}",
    )

    cfg = GRPOGsm8kConfig(
        model_path=model_path,
        steps=steps,
        batch_size=batch_size,
        num_pre_q=num_pre_q,
        global_length=global_length,
        max_length_sample=max_length_sample,
        max_length_total=max_length_total,
        ppo_epochs=ppo_epochs,
        grad_accum_steps=grad_accum_steps,
        beta=beta,
        mesh_shape=mesh_shape,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )
    print("config=" + str(asdict(cfg)))
    if args.print_config:
        return
    run_grpo_gsm8k(cfg)


if __name__ == "__main__":
    main()
