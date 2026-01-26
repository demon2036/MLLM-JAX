import argparse
import json
import os
import sys
import time
from dataclasses import dataclass

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from plugins.common.env import load_dotenv_if_present
from plugins.common.wandb_utils import maybe_init_wandb
from plugins.training.kernels.grpo_loss_pallas import (
    GRPOKernelConfig,
    grpo_per_token_loss_pallas,
    grpo_per_token_loss_reference,
)


def _jsonable_memory_stats(stats):
    if stats is None:
        return None
    out = {}
    for k, v in stats.items():
        if v is None:
            out[k] = None
            continue
        try:
            out[k] = int(v)
        except Exception:
            out[k] = str(v)
    return out


@dataclass(frozen=True)
class GRPOKernelBenchWandbConfig:
    impl: str
    batch: int
    time: int
    vocab: int
    dtype: str
    seed: int
    iters: int
    warmup: int
    old_logp_noise_scale: float
    epsilon_low: float
    epsilon_high: float
    temperature: float
    block_size: int
    time_block: int


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--impl", choices=["ref", "kernel"], required=True)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--time", type=int, default=2048)
    p.add_argument("--vocab", type=int, default=151643)
    p.add_argument("--dtype", choices=["bf16", "f16", "f32"], default="bf16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--old_logp_noise_scale", type=float, default=0.3)
    p.add_argument("--epsilon_low", type=float, default=0.2)
    p.add_argument("--epsilon_high", type=float, default=0.2)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--block_size", type=int, default=2048)
    p.add_argument("--time_block", type=int, default=8)
    p.add_argument("--wandb_project", type=str, default="mllm-jax-grpo-kernel")
    p.add_argument("--wandb_mode", type=str, default="online")
    p.add_argument("--wandb_name", type=str, default=None)
    args = p.parse_args()

    if args.temperature <= 0:
        raise ValueError("temperature must be > 0")

    load_dotenv_if_present(repo_root=REPO_ROOT)

    import jax
    import jax.numpy as jnp

    try:
        jax.distributed.initialize()
    except Exception:
        pass

    process_index = int(getattr(jax, "process_index", lambda: 0)())

    cfg_wandb = GRPOKernelBenchWandbConfig(
        impl=str(args.impl),
        batch=int(args.batch),
        time=int(args.time),
        vocab=int(args.vocab),
        dtype=str(args.dtype),
        seed=int(args.seed),
        iters=int(args.iters),
        warmup=int(args.warmup),
        old_logp_noise_scale=float(args.old_logp_noise_scale),
        epsilon_low=float(args.epsilon_low),
        epsilon_high=float(args.epsilon_high),
        temperature=float(args.temperature),
        block_size=int(args.block_size),
        time_block=int(args.time_block),
    )

    wandb_name = args.wandb_name
    if wandb_name is None:
        wandb_name = f"grpo_kernel_bench_{cfg_wandb.impl}_b{cfg_wandb.batch}_t{cfg_wandb.time}_v{cfg_wandb.vocab}"

    wandb = maybe_init_wandb(
        cfg=cfg_wandb,
        project=str(args.wandb_project),
        name=str(wandb_name),
        mode=str(args.wandb_mode),
        process_index=process_index,
    )

    dtype = {
        "bf16": jnp.bfloat16,
        "f16": jnp.float16,
        "f32": jnp.float32,
    }[args.dtype]

    device = jax.devices()[0]

    key = jax.random.PRNGKey(int(args.seed))
    key_ids, key_adv, key_noise = jax.random.split(key, 3)

    # Use zeros for logits to avoid random-gen peak inflating HBM stats.
    logits = jnp.zeros((int(args.batch), int(args.time), int(args.vocab)), dtype=dtype)

    chosen_ids = jax.random.randint(
        key_ids,
        (int(args.batch), int(args.time)),
        0,
        int(args.vocab),
        dtype=jnp.int32,
    )
    advantages = jax.random.normal(key_adv, (int(args.batch),), dtype=jnp.float32)

    base_logp = -jnp.log(jnp.asarray(int(args.vocab), dtype=jnp.float32)) / float(args.temperature)
    noise = jax.random.normal(key_noise, chosen_ids.shape, dtype=jnp.float32) * float(args.old_logp_noise_scale)
    old_per_token_logps = base_logp + noise

    kernel_cfg = GRPOKernelConfig(
        block_size=int(args.block_size),
        time_block=int(args.time_block),
        epsilon_low=float(args.epsilon_low),
        epsilon_high=float(args.epsilon_high),
        temperature=float(args.temperature),
    )

    def scalar_loss_ref(l):
        per_loss, _per_logp = grpo_per_token_loss_reference(
            logits=l,
            chosen_ids=chosen_ids,
            old_per_token_logps=old_per_token_logps,
            advantages=advantages,
            epsilon_low=float(args.epsilon_low),
            epsilon_high=float(args.epsilon_high),
            temperature=float(args.temperature),
        )
        return per_loss.mean()

    def scalar_loss_kernel(l):
        per_loss, _per_logp = grpo_per_token_loss_pallas(
            logits=l,
            chosen_ids=chosen_ids,
            old_per_token_logps=old_per_token_logps,
            advantages=advantages,
            cfg=kernel_cfg,
            interpret=False,
            debug=False,
        )
        return per_loss.mean()

    loss_fn = scalar_loss_ref if args.impl == "ref" else scalar_loss_kernel
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    # Baseline HBM after data residency.
    jax.block_until_ready(logits)
    mem_after_alloc = _jsonable_memory_stats(device.memory_stats())

    t0 = time.perf_counter()
    loss, grad = loss_and_grad(logits)
    jax.block_until_ready((loss, grad))
    first_call_s = time.perf_counter() - t0

    for _ in range(int(args.warmup)):
        loss, grad = loss_and_grad(logits)
        jax.block_until_ready((loss, grad))

    times_s = []
    for _ in range(int(args.iters)):
        t0 = time.perf_counter()
        loss, grad = loss_and_grad(logits)
        jax.block_until_ready((loss, grad))
        times_s.append(time.perf_counter() - t0)

    mem_after_run = _jsonable_memory_stats(device.memory_stats())

    out = {
        "impl": args.impl,
        "device": str(device),
        "shape": {"batch": int(args.batch), "time": int(args.time), "vocab": int(args.vocab)},
        "dtype": str(dtype),
        "loss": float(jnp.asarray(loss)),
        "kernel": {"block_size": int(kernel_cfg.block_size), "time_block": int(kernel_cfg.time_block)},
        "grpo": {
            "epsilon_low": float(args.epsilon_low),
            "epsilon_high": float(args.epsilon_high),
            "temperature": float(args.temperature),
        },
        "old_logp_noise_scale": float(args.old_logp_noise_scale),
        "first_call_s": first_call_s,
        "avg_step_ms": (sum(times_s) / max(len(times_s), 1)) * 1000.0,
        "mem_after_alloc": mem_after_alloc,
        "mem_after_run": mem_after_run,
    }

    if wandb is not None:
        try:
            wandb.log(
                {
                    "bench/first_call_s": float(first_call_s),
                    "bench/avg_step_ms": float(out["avg_step_ms"]),
                    "bench/loss": float(out["loss"]),
                }
            )
        except Exception as e:
            print(f"wandb.log failed: {e}")

    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
