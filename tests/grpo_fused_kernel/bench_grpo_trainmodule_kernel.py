# basedpyright: reportAny=false
# pyright: reportUnknownParameterType=false, reportMissingParameterType=false

from __future__ import annotations

import argparse
import math
import os
import sys
import time


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _block_until_ready(x: object) -> None:
    bur = getattr(x, "block_until_ready", None)
    if callable(bur):
        _ = bur()


def _block_tree(tree: object) -> None:
    if isinstance(tree, (tuple, list)):
        for x in tree:
            _block_tree(x)
        return
    _block_until_ready(tree)


def _format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    units = ("KiB", "MiB", "GiB", "TiB", "PiB")
    f = float(n)
    for unit in units:
        f /= 1024.0
        if f < 1024.0:
            return f"{f:.2f} {unit}"
    return f"{f:.2f} EiB"


def _devices_summary(devices: list[object]) -> str:
    parts = []
    for d in devices:
        platform = getattr(d, "platform", "unknown")
        device_kind = getattr(d, "device_kind", None)
        device_id = getattr(d, "id", None)
        s = str(platform)
        if device_kind:
            s += f":{device_kind}"
        if device_id is not None:
            s += f":{device_id}"
        parts.append(s)
    return ", ".join(parts)


def _print_env_info(*, jax_mod: object, devices: list[object]) -> None:
    jax_version = getattr(jax_mod, "__version__", "unknown")
    jaxlib_version = None
    try:
        import jaxlib  # type: ignore

        jaxlib_version = getattr(jaxlib, "__version__", None)
    except Exception:
        jaxlib_version = None

    print("env:")
    print(f"  jax:    {jax_version}")
    if jaxlib_version is None:
        print("  jaxlib: <unknown>")
    else:
        print(f"  jaxlib: {jaxlib_version}")
    print(f"  devices ({len(devices)}): {_devices_summary(devices)}")


def _print_memory_analysis(*, label: str, compiled: object) -> None:
    mem_fn = getattr(compiled, "memory_analysis", None)
    if not callable(mem_fn):
        print(f"{label}: memory_analysis() unavailable for this JAX executable")
        return

    stats = mem_fn()
    print(f"{label}:")
    for field in ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes"):
        v = getattr(stats, field, None)
        if v is None:
            continue
        v_int = int(v)
        print(f"    {field}: {v_int} ({_format_bytes(v_int)})")


def _time_compiled(*, compiled: object, args: tuple[object, ...], iters: int) -> float:
    # Warm up once (compile happens before this call).
    out = compiled(*args)
    _block_tree(out)

    t0 = time.perf_counter()
    for _ in range(iters):
        out = compiled(*args)
        _block_tree(out)
    t1 = time.perf_counter()
    return (t1 - t0) / float(iters)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TPU-only microbenchmark for TrainGRPOModule-style math vs fused grpo_loss_logp_entropy"
    )
    _ = parser.add_argument(
        "--mode",
        choices=("forward", "grad", "both"),
        default="both",
        help="Benchmark forward, backward (value_and_grad), or both.",
    )
    _ = parser.add_argument("--batch", type=int, default=1)
    _ = parser.add_argument("--seq-len", type=int, default=4096)
    _ = parser.add_argument("--vocab", type=int, default=151936)
    _ = parser.add_argument("--iters", type=int, default=10)
    _ = parser.add_argument("--temperature", type=float, default=1.0)
    _ = parser.add_argument("--eps-low", type=float, default=0.2)
    _ = parser.add_argument("--eps-high", type=float, default=0.2)

    class Args(argparse.Namespace):
        mode: str = "both"
        batch: int = 1
        seq_len: int = 4096
        vocab: int = 151936
        iters: int = 10
        temperature: float = 1.0
        eps_low: float = 0.2
        eps_high: float = 0.2

    args = parser.parse_args(namespace=Args())

    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        print("JAX is required for this benchmark. Install JAX and run on TPU.")
        return 0

    devices = list(jax.devices())
    _print_env_info(jax_mod=jax, devices=devices)
    if not any(getattr(d, "platform", None) == "tpu" for d in devices):
        print("TPU required. Set PJRT_DEVICE=TPU (or equivalent) and rerun on a TPU runtime.")
        return 0

    from plugins.training.rl.grpo import FUSED_AVAILABLE, grpo_loss_logp_entropy

    if not FUSED_AVAILABLE:
        print("Pallas fused kernel not available in this environment.")
        return 0

    bsz, seq_len, vocab = args.batch, args.seq_len, args.vocab
    print(
        "config:"
        f" mode={args.mode} batch={bsz} seq_len={seq_len} vocab={vocab}"
        f" iters={args.iters} logits_dtype=bf16"
    )

    key = jax.random.PRNGKey(0)
    k_logits, k_ids, k_adv = jax.random.split(key, 3)
    logits = jax.random.normal(k_logits, (bsz, seq_len + 1, vocab), dtype=jnp.bfloat16)
    completion_ids = jax.random.randint(
        k_ids, (bsz, seq_len), minval=0, maxval=vocab, dtype=jnp.int32
    )
    advantages = jax.random.normal(k_adv, (bsz,), dtype=jnp.float32)
    completion_mask = jnp.ones((bsz, seq_len), dtype=jnp.int32)

    # Treat old_logp as a constant (like inputs["old_per_token_logps"] in the training loop).
    approx_logp = -math.log(float(vocab))
    old_logp = jax.lax.stop_gradient(jnp.full((bsz, seq_len), approx_logp, dtype=jnp.float32))

    def baseline_forward(lg, old, ids, adv, mask):
        # Mirrors legacy TrainGRPOModule math: full-vocab log_softmax + softmax entropy.
        per_token_logps = (
            jnp.take_along_axis(jax.nn.log_softmax(lg[:, :-1, :], axis=-1), ids[:, :, None], axis=-1)[
                :, :, 0
            ]
            / args.temperature
        )

        probs = jax.nn.softmax(lg[:, :-1, :] / args.temperature, axis=-1)
        token_entropy = -jnp.sum(probs * jax.lax.log(probs + 1e-9), axis=-1)

        ratio = jnp.exp(per_token_logps - old)
        clipped_ratio = jnp.clip(ratio, 1.0 - args.eps_low, 1.0 + args.eps_high)
        adv_broadcast = adv[:, None]
        per_token_loss = -jnp.minimum(ratio * adv_broadcast, clipped_ratio * adv_broadcast)

        keep = mask != 0
        per_token_loss = jnp.where(keep, per_token_loss, 0.0)
        return per_token_loss, per_token_logps, token_entropy

    def fused_forward(lg, old, ids, adv, mask):
        return grpo_loss_logp_entropy(
            lg,
            old_logp=old,
            ref_logp=None,
            completion_ids=ids,
            advantages=adv,
            completion_mask=mask,
            temperature=args.temperature,
            beta=0.0,  # legacy TrainGRPOModule behavior
            eps_low=args.eps_low,
            eps_high=args.eps_high,
            use_fused=True,
        )

    baseline_forward_jit = jax.jit(baseline_forward)
    fused_forward_jit = jax.jit(fused_forward)

    def baseline_scalar_loss(lg, old, ids, adv, mask):
        per_token_loss, _logp, _entropy = baseline_forward(lg, old, ids, adv, mask)
        return jnp.sum(per_token_loss)

    def fused_scalar_loss(lg, old, ids, adv, mask):
        per_token_loss, _logp, _entropy = fused_forward(lg, old, ids, adv, mask)
        return jnp.sum(per_token_loss)

    baseline_val_grad_jit = jax.jit(jax.value_and_grad(baseline_scalar_loss))
    fused_val_grad_jit = jax.jit(jax.value_and_grad(fused_scalar_loss))

    forward_args = (logits, old_logp, completion_ids, advantages, completion_mask)

    if args.mode in ("forward", "both"):
        baseline_forward_compiled = baseline_forward_jit.lower(*forward_args).compile()
        fused_forward_compiled = fused_forward_jit.lower(*forward_args).compile()

        _print_memory_analysis(label="baseline forward", compiled=baseline_forward_compiled)
        _print_memory_analysis(label="fused forward", compiled=fused_forward_compiled)

        baseline_dt = _time_compiled(compiled=baseline_forward_compiled, args=forward_args, iters=args.iters)
        fused_dt = _time_compiled(compiled=fused_forward_compiled, args=forward_args, iters=args.iters)

        print("timing (forward):")
        print(f"  baseline: {baseline_dt * 1e3:.3f} ms/iter")
        print(f"  fused:    {fused_dt * 1e3:.3f} ms/iter")

        base_loss, base_logp, base_entropy = baseline_forward_compiled(*forward_args)
        fused_loss, fused_logp, fused_entropy = fused_forward_compiled(*forward_args)

        loss_diff = jnp.max(jnp.abs(base_loss.astype(jnp.float32) - fused_loss.astype(jnp.float32)))
        logp_diff = jnp.max(jnp.abs(base_logp.astype(jnp.float32) - fused_logp.astype(jnp.float32)))
        ent_diff = jnp.max(
            jnp.abs(base_entropy.astype(jnp.float32) - fused_entropy.astype(jnp.float32))
        )
        _block_tree((loss_diff, logp_diff, ent_diff))
        print("max_abs_diffs (forward):")
        print(f"  loss:    {float(loss_diff):.6e}")
        print(f"  logp:    {float(logp_diff):.6e}")
        print(f"  entropy: {float(ent_diff):.6e}")

    if args.mode in ("grad", "both"):
        baseline_grad_compiled = baseline_val_grad_jit.lower(*forward_args).compile()
        fused_grad_compiled = fused_val_grad_jit.lower(*forward_args).compile()

        _print_memory_analysis(label="baseline value_and_grad", compiled=baseline_grad_compiled)
        _print_memory_analysis(label="fused value_and_grad", compiled=fused_grad_compiled)

        baseline_dt = _time_compiled(compiled=baseline_grad_compiled, args=forward_args, iters=args.iters)
        fused_dt = _time_compiled(compiled=fused_grad_compiled, args=forward_args, iters=args.iters)

        print("timing (value_and_grad):")
        print(f"  baseline: {baseline_dt * 1e3:.3f} ms/iter")
        print(f"  fused:    {fused_dt * 1e3:.3f} ms/iter")

        base_loss, base_grads = baseline_grad_compiled(*forward_args)
        fused_loss, fused_grads = fused_grad_compiled(*forward_args)
        loss_diff = jnp.abs(base_loss.astype(jnp.float32) - fused_loss.astype(jnp.float32))
        grad_diff = jnp.max(jnp.abs(base_grads.astype(jnp.float32) - fused_grads.astype(jnp.float32)))
        _block_tree((loss_diff, grad_diff))
        print("max_abs_diffs (value_and_grad):")
        print(f"  loss: {float(loss_diff):.6e}")
        print(f"  grad: {float(grad_diff):.6e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

