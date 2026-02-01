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

from tests.grpo_fused_kernel import grpo_fused_pallas, grpo_reference


def _block_until_ready(x: object) -> None:
    # JAX arrays implement block_until_ready; lists/None do not.
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


def main() -> int:
    parser = argparse.ArgumentParser(description="GRPO fused-kernel sandbox benchmark (TPU-only)")
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
    _ = parser.add_argument("--beta", type=float, default=0.0)
    _ = parser.add_argument("--temperature", type=float, default=1.0)
    _ = parser.add_argument("--eps-low", type=float, default=0.2)
    _ = parser.add_argument("--eps-high", type=float, default=0.2)
    class Args(argparse.Namespace):
        # Defaults match the parser defaults; argparse will overwrite them.
        mode: str = "both"
        batch: int = 1
        seq_len: int = 4096
        vocab: int = 151936
        iters: int = 10
        beta: float = 0.0
        temperature: float = 1.0
        eps_low: float = 0.2
        eps_high: float = 0.2

    args = parser.parse_args(namespace=Args())

    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        print("JAX is required for the benchmark. Install JAX and run on TPU.")
        return 0

    devices = list(jax.devices())
    _print_env_info(jax_mod=jax, devices=devices)

    if not any(getattr(d, "platform", None) == "tpu" for d in devices):
        print("TPU required. Set PJRT_DEVICE=TPU (or equivalent) and rerun on a TPU runtime.")
        return 0

    bsz, seq_len, vocab = args.batch, args.seq_len, args.vocab
    print(
        "config:"
        f" mode={args.mode} batch={bsz} seq_len={seq_len} vocab={vocab}"
        f" iters={args.iters} logits_dtype=bf16"
    )
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    # Logits are large at the target shape; bf16 keeps memory manageable.
    logits = jax.random.normal(k1, (bsz, seq_len + 1, vocab), dtype=jnp.bfloat16)
    completion_ids = jax.random.randint(
        k2, (bsz, seq_len), minval=0, maxval=vocab, dtype=jnp.int32
    )
    advantages = jax.random.normal(k3, (bsz,), dtype=jnp.float32)
    completion_mask = jnp.ones((bsz, seq_len), dtype=jnp.int32)

    # old/ref logp are treated as constants in RL-style objectives.
    # Keep them independent of logits to avoid affecting backward-time and to
    # ensure the gradient is non-trivial (old_logp=None would make grad ~ 0).
    approx_logp = -math.log(float(vocab))
    old_logp = jax.lax.stop_gradient(
        jnp.full((bsz, seq_len), approx_logp, dtype=jnp.float32)
    )
    ref_logp = jax.lax.stop_gradient(old_logp + 0.3)

    def _make_forward_fns():
        ref_fn = jax.jit(
            lambda lg, old, ref, ids, adv: grpo_reference.grpo_loss_reference_jax(
                lg,
                old,
                ref,
                ids,
                adv,
                completion_mask=completion_mask,
                temperature=args.temperature,
                beta=args.beta,
                eps_low=args.eps_low,
                eps_high=args.eps_high,
            )
        )
        fused_fn = jax.jit(
            lambda lg, old, ref, ids, adv: grpo_fused_pallas.grpo_loss_fused_pallas(
                lg,
                old,
                ref,
                ids,
                adv,
                completion_mask=completion_mask,
                temperature=args.temperature,
                beta=args.beta,
                eps_low=args.eps_low,
                eps_high=args.eps_high,
                backend="jax",
            )
        )
        return ref_fn, fused_fn

    def _make_grad_fns():
        def scalar_loss_ref(lg, old, ref, ids, adv):
            per_token_loss, _kl, _is_clipped = grpo_reference.grpo_loss_reference_jax(
                lg,
                old,
                ref,
                ids,
                adv,
                completion_mask=completion_mask,
                temperature=args.temperature,
                beta=args.beta,
                eps_low=args.eps_low,
                eps_high=args.eps_high,
            )
            return jnp.sum(jnp.asarray(per_token_loss))

        def scalar_loss_fused(lg, old, ref, ids, adv):
            per_token_loss, _kl, _is_clipped = grpo_fused_pallas.grpo_loss_fused_pallas(
                lg,
                old,
                ref,
                ids,
                adv,
                completion_mask=completion_mask,
                temperature=args.temperature,
                beta=args.beta,
                eps_low=args.eps_low,
                eps_high=args.eps_high,
                backend="jax",
            )
            return jnp.sum(jnp.asarray(per_token_loss))

        # Jit value_and_grad to include backward in the benchmark.
        ref_vg = jax.jit(jax.value_and_grad(scalar_loss_ref))
        fused_vg = jax.jit(jax.value_and_grad(scalar_loss_fused))
        return ref_vg, fused_vg

    def _time_loop(fn, *fn_args):
        out = None
        t0 = time.perf_counter()
        for _ in range(args.iters):
            out = fn(*fn_args)
        assert out is not None
        _block_tree(out)
        dt = time.perf_counter() - t0
        return out, (1e3 * dt / args.iters)

    if args.mode in ("forward", "both"):
        ref_fn, fused_fn = _make_forward_fns()

        ref_compiled = ref_fn.lower(logits, old_logp, ref_logp, completion_ids, advantages).compile()
        fused_compiled = fused_fn.lower(
            logits, old_logp, ref_logp, completion_ids, advantages
        ).compile()

        print("forward memory (compiled):")
        _print_memory_analysis(label="  ref", compiled=ref_compiled)
        _print_memory_analysis(label="  fused", compiled=fused_compiled)

        # Warmup
        _block_tree(ref_compiled(logits, old_logp, ref_logp, completion_ids, advantages))
        _block_tree(fused_compiled(logits, old_logp, ref_logp, completion_ids, advantages))

        out_ref, ref_ms = _time_loop(
            ref_compiled, logits, old_logp, ref_logp, completion_ids, advantages
        )
        out_fused, fused_ms = _time_loop(
            fused_compiled, logits, old_logp, ref_logp, completion_ids, advantages
        )

        print("forward:")
        print(f"ref:   {ref_ms:.3f} ms/iter")
        print(f"fused: {fused_ms:.3f} ms/iter")

        ref_loss, ref_kl, ref_is_clipped = out_ref
        fused_loss, fused_kl, fused_is_clipped = out_fused
        max_abs = jnp.max(jnp.abs(jnp.asarray(ref_loss) - jnp.asarray(fused_loss))).item()
        print(f"max|loss_ref-loss_fused| = {max_abs:.6e}")
        if ref_kl is None:
            assert fused_kl is None
        else:
            max_abs_kl = jnp.max(jnp.abs(jnp.asarray(ref_kl) - jnp.asarray(fused_kl))).item()
            print(f"max|kl_ref-kl_fused| = {max_abs_kl:.6e}")
        assert jnp.array_equal(jnp.asarray(ref_is_clipped), jnp.asarray(fused_is_clipped))

    if args.mode in ("grad", "both"):
        ref_vg, fused_vg = _make_grad_fns()

        ref_compiled = ref_vg.lower(logits, old_logp, ref_logp, completion_ids, advantages).compile()
        fused_compiled = fused_vg.lower(logits, old_logp, ref_logp, completion_ids, advantages).compile()

        print("grad memory (compiled):")
        _print_memory_analysis(label="  ref", compiled=ref_compiled)
        _print_memory_analysis(label="  fused", compiled=fused_compiled)

        # Warmup
        _block_tree(ref_compiled(logits, old_logp, ref_logp, completion_ids, advantages))
        _block_tree(fused_compiled(logits, old_logp, ref_logp, completion_ids, advantages))

        out_ref, ref_ms = _time_loop(ref_compiled, logits, old_logp, ref_logp, completion_ids, advantages)
        out_fused, fused_ms = _time_loop(fused_compiled, logits, old_logp, ref_logp, completion_ids, advantages)

        (ref_loss_s, ref_grad), (fused_loss_s, fused_grad) = out_ref, out_fused
        loss_diff = jnp.abs(jnp.asarray(ref_loss_s) - jnp.asarray(fused_loss_s)).item()
        # Compare a tiny gradient slice to avoid an expensive full-tensor diff.
        grad_slice_ref = jnp.asarray(ref_grad)[0, :1, :32].astype(jnp.float32)
        grad_slice_fused = jnp.asarray(fused_grad)[0, :1, :32].astype(jnp.float32)
        grad_slice_diff = jnp.max(jnp.abs(grad_slice_ref - grad_slice_fused)).item()

        print("grad (value_and_grad):")
        print(f"ref:   {ref_ms:.3f} ms/iter")
        print(f"fused: {fused_ms:.3f} ms/iter")
        print(f"abs(loss_ref-loss_fused) = {loss_diff:.6e}")
        print(f"max|grad_ref-grad_fused| (slice) = {grad_slice_diff:.6e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
