# SOP: Inspect JAX SplashAttention and design a tiled CE/logprob kernel

- **Title**: SOP: Inspect JAX TPU SplashAttention (Pallas) patterns and sketch a tiled (streaming) cross-entropy / selective logprob kernel for LM head without materializing logits
  **Prereqs**: Ubuntu Linux; `git`; `rg`; outbound network access (only if reference repos are missing and need cloning)
  **Environment (verified)**:
  - Host: `Linux john-Lecoo-N155A 6.14.0-37-generic x86_64`
  - Python: `3.12.2`
  - git: `2.48.1`
  - ripgrep: `15.1.0`
  - Repo: `/home/john/workdir/mllm-jax-rl`
  - Reference JAX clone: `workdir/jax@58c0f57` (`https://github.com/jax-ml/jax.git`)

## Goal

- Understand how JAX implements TPU SplashAttention via Pallas:
  - streaming softmax (`m/l/o` accumulators)
  - scratch + grid semantics
  - `custom_vjp` wrappers
- Use the same patterns to design a **tiled CE/logprob kernel** that:
  - streams over vocab tiles (no `[BT,V]` logits materialization),
  - supports vocab sharding (pmax/psum combine),
  - can return loss/logprob and (optionally) `dh`.

## Steps (commands actually used in this repo)

### 1) Confirm reference clones exist (gitignored)

- `cd /home/john/workdir/mllm-jax-rl`
- `ls workdir`
- `git -C workdir/jax rev-parse --short HEAD`

### 2) Locate SplashAttention implementation + sharding/tests

- `rg -n "splash_attention" workdir/jax -S --no-mmap | head -n 50`
- `sed -n '1,120p' workdir/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py`
- `rg -n "pallas_call" workdir/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py | head -n 120`
- `sed -n '740,860p' workdir/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py`
- `sed -n '2400,2620p' workdir/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py`
- `sed -n '300,420p' workdir/jax/tests/pallas/tpu_splash_attention_kernel_test.py`
- `sed -n '200,320p' workdir/jax/tests/pallas/tpu_splash_attention_kernel_sharded_test.py`

### 3) Inspect CE kernel references (Unsloth / Liger)

- `rg -n "cross_entropy_loss.py" workdir/unsloth -S --no-mmap | head -n 20`
- `sed -n '1,260p' workdir/unsloth/unsloth/kernels/cross_entropy_loss.py`
- `sed -n '1,260p' workdir/liger-kernel/src/liger_kernel/ops/fused_linear_cross_entropy.py`

### 4) Record the design notes

- See `memory/20260125_jax-splash-tiled-ce-kernel/README.md`

## Expected result

- You can point to the exact SplashAttention code locations for:
  - streaming softmax update (`m/l/o`) and end-of-stream finalize
  - `pl.pallas_call` + `PrefetchScalarGridSpec` usage
  - `jax.custom_vjp` wrapper pattern
- You have a concrete kernel design for tiled CE/logprob:
  - grid axes + scratch layout + numerics
  - sharded combine formulas (`pmax/psum`) across vocab shards

## Troubleshooting

- `rg` missing: install ripgrep or use `git grep`.
- Missing reference clones under `workdir/`: follow `docs/sops/clone-reference-repos-into-workdir.md`.

## References

- `memory/20260125_jax-splash-tiled-ce-kernel/README.md`
- `docs/sops/clone-reference-repos-into-workdir.md`

