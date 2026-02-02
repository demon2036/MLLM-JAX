# SOP: Enable fused GRPO kernel in `TrainGRPOModule`

- **Title**: SOP: Enable fused GRPO kernel in `TrainGRPOModule` (avoid full-vocab `log_softmax`/`softmax`)
- **Prereqs**:
  - Python + JAX installed.
  - For TPU runs: Pallas available in your JAX build.

## What this does

- Adds an env-gated fused path to `MLLM_JAX/train_modules/__init__.py::TrainGRPOModule`.
- When enabled, the GRPO policy loss and entropy metrics avoid materializing full-vocab
  `jax.nn.log_softmax` / `jax.nn.softmax`.
- TPU v6/v6e note: the fused forward `logsumexp+entropy` stats path emits **per-1024-column vocab
  group** stats (and reduces across groups in JAX) to avoid large `(vocab_blocks_inner, num_tokens)`
  fp32 intermediates. See
  `plugins/training/rl/grpo/fused_grpo_loss_pallas.py::_grpo_fused_logsumexp_stats_pallas`.

## Env gate

- `MLLM_JAX_GRPO_FUSED=1`: enable fused path (Pallas when available; otherwise JAX fallback).
- `MLLM_JAX_GRPO_FUSED=0`: disable fused path (force legacy implementation).
- Unset: enabled on TPU by default; legacy path on CPU/GPU.

## Steps (commands actually used)

### 1) Smoke-test the fused kernel on CPU (interpret-mode)

```bash
MLLM_JAX_GRPO_FUSED=1 python - <<'PY'
import jax
import jax.numpy as jnp
from plugins.training.rl.grpo import grpo_loss_logp_entropy

logits = jnp.zeros((1, 3, 8), dtype=jnp.bfloat16)  # [B, L+1, V] with L=2
completion_ids = jnp.zeros((1, 2), dtype=jnp.int32)
advantages = jnp.ones((1,), dtype=jnp.float32)
completion_mask = jnp.ones((1, 2), dtype=jnp.int32)

loss, logp, entropy = grpo_loss_logp_entropy(
    logits,
    old_logp=None,
    ref_logp=None,
    completion_ids=completion_ids,
    advantages=advantages,
    completion_mask=completion_mask,
    temperature=1.0,
    beta=0.0,
    eps_low=0.2,
    eps_high=0.2,
    use_fused=True,
)
print("loss", loss)
print("logp", logp)
print("entropy", entropy)
print(
    "grads",
    jax.grad(
        lambda x: jnp.sum(
            grpo_loss_logp_entropy(
                x,
                old_logp=None,
                ref_logp=None,
                completion_ids=completion_ids,
                advantages=advantages,
                completion_mask=completion_mask,
                use_fused=True,
            )[0]
        )
    )(logits).shape,
)
PY
```

### 2) Run the existing fused-kernel correctness tests

```bash
python -m pytest -q tests/grpo_fused_kernel
```

### 3) TPU microbench (TrainGRPOModule baseline vs fused kernel)

- See: `docs/sops/tpu-vm-v4-8-grpo-trainmodule-kernel-bench.md`
- Summary (this run, TPU v4-8, commit `a9f6046`, `WANDB_MODE=disabled`):
  - forward timing: baseline 5.322 ms/iter, fused 9.744 ms/iter
  - value_and_grad timing: baseline 18.839 ms/iter, fused 15.923 ms/iter
  - temp_size_in_bytes (forward): baseline 1244917760, fused 1244885504
  - temp_size_in_bytes (value_and_grad): baseline 1245192704, fused 1244788736
- Note: v6e-8 TPU creation in `us-central2-b` failed due to quota limit 0 (details in the TPU SOP).

## Expected result

- The smoke-test prints finite `loss/logp/entropy` and a gradient array with shape `[B, L+1, V]`.
- `pytest` reports all tests passing.

## Troubleshooting

- If you hit Pallas import errors: set `MLLM_JAX_GRPO_FUSED=0` to force the legacy path.
- If you see TPU Mosaic lowering errors: check the kernel notes in
  `.sisyphus/notepads/grpo-trainmodule-kernel/learnings.md`.
- If you cannot create a v6e-8 TPU due to quota `Limit: 0`, use v4-8 (or request v6e quota) and run
  the v4-8 microbench SOP first.

## References

- `plugins/training/rl/grpo/fused_grpo_loss_pallas.py`
- `tests/grpo_fused_kernel/grpo_fused_pallas.py`
