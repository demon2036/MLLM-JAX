# JAX SplashAttention inspection + tiled CE kernel design notes

## Goal

- Inspect how JAX implements TPU SplashAttention (Pallas) to reuse the same kernel patterns.
- Design a **tiled/streaming cross-entropy (or selective logprob) kernel** that fuses the LM head projection pattern and avoids materializing `[BT, V]` logits.
- Use Unsloth/Liger as references for CE numerics + ergonomics.

## Evidence: repos + files inspected

### Clones (repo-local, gitignored)

- `workdir/jax` @ `58c0f57` (`origin https://github.com/jax-ml/jax.git`)
- `workdir/unsloth` (CE Triton kernels)
- `workdir/liger-kernel` (CE + fused-linear-CE wrappers)

### Local sanity

- Repo tests: `python -m pytest -q` (exit `0`, `31 passed, 1 warning`)

### JAX SplashAttention entry points

- `workdir/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py`
- `workdir/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py`
- `workdir/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py`
- Docs: `workdir/jax/docs/pallas/tpu/sparse.md`, `workdir/jax/docs/pallas/tpu/matmul.md`, `workdir/jax/docs/pallas/tpu/details.rst`
- Tests: `workdir/jax/tests/pallas/tpu_splash_attention_kernel_test.py`, `workdir/jax/tests/pallas/tpu_splash_attention_kernel_sharded_test.py`

### Unsloth / Liger CE references

- Unsloth logits-CE: `workdir/unsloth/unsloth/kernels/cross_entropy_loss.py`
- Liger fused-linear-CE wrapper: `workdir/liger-kernel/src/liger_kernel/ops/fused_linear_cross_entropy.py`
- Liger CE kernel entry: `workdir/liger-kernel/src/liger_kernel/ops/cross_entropy.py`

## Key takeaways from SplashAttention (what we will reuse)

1. **Streaming softmax** over an inner “arbitrary” grid axis:
   - Maintain `m` (running max) and `l` (running sumexp) in scratch.
   - Update rule (same as FlashAttention): rescale previous accumulators when `m` increases.
2. **Scratch + pipeline structure**:
   - Use `pltpu.PrefetchScalarGridSpec(...)` with `scratch_shapes=[pltpu.VMEM(...), ...]`.
   - Use `pltpu.CompilerParams(dimension_semantics=(..., "arbitrary"))` to model the streaming axis.
3. **Numerics**:
   - Do matmul accum in `float32` via `preferred_element_type=jnp.float32`.
   - Use stable `logsumexp = log(l) + m`.
4. **custom_vjp wrapper**:
   - SplashAttention is `jax.custom_vjp` with a Pallas fwd and one or more Pallas bwd kernels.
   - Pattern is reusable for CE/logprob kernels (forward produces residuals; backward consumes).

## Tiled CE kernel: what we’re aiming for

### Problem

For LM head / embedding projection, computing `logits = h @ W.T` with `h:[BT,H]`, `W:[V,H]` materializes `[BT,V]` logits which is expensive in memory/bandwidth. We often only need:

- `logsumexp(logits)` per token, and
- the target token’s logit `logits[target]`,

and for backward:

- `dh = (softmax @ W) - W[target]` (if only backprop to hidden; common if vocab weights frozen / LoRA).

### Core trick: “treat vocab like KV blocks”

Reuse FlashAttention’s streaming softmax idea:

- For each token-block `X_blk:[Bq,H]` and vocab-tile `W_blk:[Bv,H]`:
  - `logits_blk = X_blk @ W_blk.T`  (like `qk`)
  - `s_blk = exp(logits_blk - m_next)` (like attention scores after stabilization)
  - Update `m,l` and an “output accumulator”:
    - `o_blk += s_blk @ W_blk` (like `s @ v`)  → yields unnormalized `Σ exp(logit) * W`
- At the end:
  - `expected_W = o / l`  → `Σ softmax(logit) * W`
  - `logsumexp = log(l) + m`
  - `loss = logsumexp - logits[target]` (ignore-index masked)
  - `dh = expected_W - W[target]`

This gives **loss + dh** without materializing logits/dlogits.

## Implementation sketch (no code yet)

### Kernel grid + tiling

- Grid axes:
  - `pid_bt`: token blocks (parallel)
  - `pid_v`: vocab tiles (arbitrary / streaming)
- Block sizes:
  - `block_bt`: 16/32/64 (tunable; limited by `H` and scratch footprint)
  - `block_v`: 128/256/512 (must be large enough for MXU efficiency; likely multiple of 128)
  - `H` assumed multiple of 128 (pad otherwise).

### Kernel scratch

Per token-block:

- `m_ref[block_bt]` (`f32`)
- `l_ref[block_bt]` (`f32`)
- `o_ref[block_bt, H]` (`f32`)  # unnormalized Σ exp(logit-m) * W
- `target_logit_ref[block_bt]` (`f32`)  # only one tile writes per token
- optional: `target_w_ref[block_bt, H]` (bf16 or f32) if we also want `dh` inside-kernel

### Sharding plan (vocab-sliced)

If `W` is sharded across vocab on mesh axis `vocab`:

- Each device computes local `(m_local, l_local, o_local, target_logit_local, target_w_local)`.
- Combine across `vocab` shards:
  - `m = pmax(m_local)`
  - `l = psum(exp(m_local - m) * l_local)`
  - `o = psum(exp(m_local - m) * o_local)`
  - `target_logit = psum(target_logit_local)`
  - `target_w = psum(target_w_local)`
- Then `logsumexp = log(l) + m`, `dh = o/l - target_w`.

This is the same associative-softmax-combine pattern as streaming attention.

## Next steps (when we start coding)

1. Define a `TiledCrossEntropyKernelConfig` (block sizes, temperature, ignore_index).
2. Implement a Pallas forward kernel that returns local `(m,l,o,target_logit[,target_w])`.
3. Wrap with a JAX function that:
   - runs the kernel,
   - does `pmax/psum` reductions if vocab-sharded,
   - returns per-token loss/logp and a residual for custom_vjp.
4. Backward:
   - v0: support only `dh` (weight frozen) by storing `dh_unscaled` as residual (Liger-style).
   - v1: add `dW` with a second pallas kernel if needed.
