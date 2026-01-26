import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp

from plugins.training.kernels.tiled_cross_entropy_pallas import (
    CrossEntropyKernelConfig,
    cross_entropy_per_token_pallas,
    cross_entropy_per_token_reference,
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
    p.add_argument("--ignore_prob", type=float, default=0.1)
    p.add_argument("--ignore_index", type=int, default=-100)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--block_size", type=int, default=2048)
    p.add_argument("--time_block", type=int, default=8)
    args = p.parse_args()

    if args.temperature <= 0:
        raise ValueError("temperature must be > 0")

    dtype = {
        "bf16": jnp.bfloat16,
        "f16": jnp.float16,
        "f32": jnp.float32,
    }[args.dtype]

    device = jax.devices()[0]

    key = jax.random.PRNGKey(args.seed)
    key_labels, key_ign = jax.random.split(key, 2)

    # Use zeros for logits to avoid random-gen peak inflating HBM stats.
    logits = jnp.zeros((args.batch, args.time, args.vocab), dtype=dtype)

    labels = jax.random.randint(
        key_labels,
        (args.batch, args.time),
        0,
        args.vocab,
        dtype=jnp.int32,
    )
    ignore_mask = jax.random.bernoulli(key_ign, p=args.ignore_prob, shape=labels.shape)
    labels = jnp.where(ignore_mask, args.ignore_index, labels)

    valid = labels != args.ignore_index
    mask = valid.astype(jnp.float32)
    denom = mask.sum()

    cfg = CrossEntropyKernelConfig(
        block_size=int(args.block_size),
        time_block=int(args.time_block),
        ignore_index=int(args.ignore_index),
        temperature=float(args.temperature),
    )

    def scalar_loss_ref(l):
        per_loss, _per_logp = cross_entropy_per_token_reference(
            l,
            labels,
            ignore_index=args.ignore_index,
            temperature=args.temperature,
        )
        return (per_loss * mask).sum() / (denom + 1e-8)

    def scalar_loss_kernel(l):
        per_loss, _per_logp = cross_entropy_per_token_pallas(
            logits=l,
            labels=labels,
            cfg=cfg,
            interpret=False,
            debug=False,
        )
        return (per_loss * mask).sum() / (denom + 1e-8)

    loss_fn = scalar_loss_ref if args.impl == "ref" else scalar_loss_kernel
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    # Baseline HBM after data residency.
    jax.block_until_ready(logits)
    mem_after_alloc = _jsonable_memory_stats(device.memory_stats())

    t0 = time.perf_counter()
    loss, grad = loss_and_grad(logits)
    loss.block_until_ready()
    t1 = time.perf_counter()
    first_call_s = t1 - t0

    for _ in range(int(args.warmup)):
        loss, grad = loss_and_grad(logits)
        loss.block_until_ready()

    times_s = []
    for _ in range(int(args.iters)):
        t0 = time.perf_counter()
        loss, grad = loss_and_grad(logits)
        loss.block_until_ready()
        times_s.append(time.perf_counter() - t0)

    mem_after_run = _jsonable_memory_stats(device.memory_stats())

    out = {
        "impl": args.impl,
        "device": str(device),
        "shape": {"batch": args.batch, "time": args.time, "vocab": args.vocab},
        "dtype": str(dtype),
        "kernel": {"block_size": int(args.block_size), "time_block": int(args.time_block)},
        "first_call_s": first_call_s,
        "avg_step_ms": (sum(times_s) / max(len(times_s), 1)) * 1000.0,
        "mem_after_alloc": mem_after_alloc,
        "mem_after_run": mem_after_run,
    }

    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
