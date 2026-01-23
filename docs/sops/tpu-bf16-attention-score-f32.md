# TPU: Preserve attention-score precision on TPU (bf16 operands, fp32 output)

- **Title**: SOP: Get fp32 attention-score accumulation/output on TPU without switching the whole model to fp32/fp16
  **Prereqs**: TPU VM is `READY`; conda env `mllm-jax` exists
  **Environment (verified)**:
  - TPU VM: `mllm-jax-v6e-8-260120021350` (`v6e-8`, 1 host), zone `us-east1-d`
  - Python: `3.12.12` (conda env `mllm-jax`)
  - JAX/JAXLIB: `0.8.2` / `0.8.2` (backend `tpu`)

## Problem

- `bf16 @ bf16` matmul returns **bf16** by default in JAX, so the attention score matrix is rounded to bf16.
- In RL (PPO/GRPO), small logit/logprob differences can matter (sampling + ratio/advantage sensitivity), so people often suspect “fp32 works better”.
- On TPU we usually want bf16 performance, but we still want **fp32 accumulation/output** at key points (e.g. attention scores) to reduce numeric error.

## What works on TPU (verified)

### A) `jnp.matmul`/`@` with bf16 inputs returns bf16

- Output dtype: `bfloat16`

### B) `jax.lax.dot_general(... preferred_element_type=jnp.float32)` keeps bf16 operands but returns fp32 output

- Output dtype: `float32`
- For bf16 inputs, the fp32 output matches the fp32 baseline of the same bf16-valued inputs (no bf16 output rounding).

## Steps (commands actually used)

Run on the TPU VM:

1) Activate the env

- `source /root/miniconda3/etc/profile.d/conda.sh && conda activate mllm-jax`

2) Verify dtype + error for decode-like shapes (`q_len=1`, `head_dim=128`)

- `python - <<'PY'`
  - `import time`
  - `import jax`
  - `import jax.numpy as jnp`
  - `from jax import lax`
  - `print('backend=', jax.default_backend())`
  - `print('jax=', jax.__version__)`
  - `B,H,Q,D = 1,1,1,128; S=512`
  - `key = jax.random.PRNGKey(0)`
  - `q = jax.random.normal(key, (B,H,Q,D), dtype=jnp.bfloat16)`
  - `k = jax.random.normal(key, (B,H,S,D), dtype=jnp.bfloat16)`
  - `out_matmul = q @ jnp.swapaxes(k, 2, 3)`
  - `print('matmul_out_dtype=', out_matmul.dtype)`
  - `out_dot_f32 = lax.dot_general(q, k, dimension_numbers=(((3,), (3,)), ((0,1), (0,1))), preferred_element_type=jnp.float32)`
  - `print('dot_general(preferred=float32)_out_dtype=', out_dot_f32.dtype)`
  - `q32=q.astype(jnp.float32); k32=k.astype(jnp.float32); out32=q32 @ jnp.swapaxes(k32,2,3)`
  - `print('max_abs_err matmul(bf16->f32) vs f32=', jnp.max(jnp.abs(out_matmul.astype(jnp.float32)-out32)).item())`
  - `print('max_abs_err dot_f32 vs f32=', jnp.max(jnp.abs(out_dot_f32-out32)).item())`
  - `@jax.jit`
  - `def f_matmul(q,k): return q @ jnp.swapaxes(k,2,3)`
  - `@jax.jit`
  - `def f_dot_f32(q,k): return lax.dot_general(q,k,dimension_numbers=(((3,), (3,)), ((0,1), (0,1))), preferred_element_type=jnp.float32)`
  - `f_matmul(q,k).block_until_ready(); f_dot_f32(q,k).block_until_ready()`
  - `def bench(fn,iters=50):`
  - `    t0=time.time()`
  - `    for _ in range(iters): fn(q,k).block_until_ready()`
  - `    return (time.time()-t0)/iters`
  - `print('avg_s matmul=', bench(f_matmul))`
  - `print('avg_s dot_f32=', bench(f_dot_f32))`
- `PY`

Expected output (from the verified run):
- `matmul_out_dtype= bfloat16`
- `dot_general(preferred=float32)_out_dtype= float32`
- `max_abs_err matmul(bf16->f32) vs f32= 0.0673675537109375`
- `max_abs_err dot_f32 vs f32= 0.0`
- `avg_s matmul= 0.00019278526306152344`
- `avg_s dot_f32= 0.00018614768981933594`

3) (Optional) Verify float16 matmul does not provide a clear speed advantage

- `python - <<'PY'`
  - `import time`
  - `import jax`
  - `import jax.numpy as jnp`
  - `print('backend=', jax.default_backend())`
  - `B,H,Q,D,S=1,1,1,128,512`
  - `key=jax.random.PRNGKey(0)`
  - `q16=jax.random.normal(key,(B,H,Q,D),dtype=jnp.float16)`
  - `k16=jax.random.normal(key,(B,H,S,D),dtype=jnp.float16)`
  - `out = q16 @ jnp.swapaxes(k16,2,3)`
  - `print('matmul fp16 inputs out dtype=', out.dtype)`
  - `@jax.jit`
  - `def f(q,k): return q @ jnp.swapaxes(k,2,3)`
  - `f(q16,k16).block_until_ready()`
  - `t0=time.time()`
  - `for _ in range(10): f(q16,k16).block_until_ready()`
  - `print('avg_s fp16 matmul=', (time.time()-t0)/10)`
- `PY`

Expected output (from the verified run):
- `matmul fp16 inputs out dtype= float16`
- `avg_s fp16 matmul= 0.00021932125091552734`

## How this applies to this repo

- For rollout decode attention (`q_len=1`) we can keep speed while improving score precision by using:
  - `jax.lax.dot_general(... preferred_element_type=jnp.float32)` for the QK score matrix.
- Current implementation (opt-in) is in:
  - `plugins/training/rollout_optimizations/qwen2_decode_attention.py`

## References

- JAX API: `jax.lax.dot_general`
- `plugins/training/rollout_optimizations/qwen2_decode_attention.py`

