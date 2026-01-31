# GRPO Fused-Kernel Sandbox

This folder is a standalone sandbox for experimenting with a fused GRPO loss
kernel (eventually via JAX Pallas + custom_vjp). Nothing here is wired into
MaxText/sglang-jax yet.

## Baseline vs Fused

- Baseline: `grpo_reference.py`
  - Reference implementation of the GRPO token-level PPO-style objective.
  - Designed to be readable and to match the intended API surface.

- Fused: `grpo_fused_pallas.py`
  - Placeholder for a future JAX Pallas fused kernel.
  - For now it delegates to the reference implementation so the test/bench
    harness can land early.

## Correctness (CPU)

The correctness tests are intentionally CPU-friendly and only validate imports
and basic shapes.

```bash
python -m pytest -q tests/grpo_fused_kernel
```

## Benchmark (TPU-only)

The benchmark script is meant to be run on a TPU runtime.

Expected environment variables (examples):

- `PJRT_DEVICE=TPU` (common on TPU VMs)
- Alternatively, some setups use `JAX_PLATFORMS=tpu`

Run:

```bash
PJRT_DEVICE=TPU python tests/grpo_fused_kernel/bench_grpo_fused.py --batch 32 --seq-len 256 --vocab 32000
```

If you run the benchmark on a non-TPU platform, it exits early with a clear
message.
