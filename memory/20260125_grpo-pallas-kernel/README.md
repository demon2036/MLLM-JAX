# GRPO Pallas kernel + TPU gradcheck

## Goal

- Implement a JAX Pallas GRPO loss kernel (forward + backward).
- Validate on TPU `mllm-jax-v6e-8-spot-260124132428` by comparing loss + `dlogits` vs the existing non-kernel reference on Qwen2.5-1.5B.

## Completion criteria

- A deterministic gradcheck script exits `0` on TPU and prints/logs:
  - `loss_ref`, `loss_kernel`, `abs_diff_loss`, `max_abs_diff_dlogits`, `max_rel_diff_dlogits`
  - tolerance gates met (recorded below)
- W&B logging enabled with `wandb_mode=online` (or record why not possible).
- No Python traceback / non-zero exit in the TPU run.
- SOP added/updated with the exact commands run.

## Work log (evidence)

### Observed TPU failures (before fix)

- Symptoms:
  - Forward mismatch: `per_token_logps` / `per_token_loss` differ from reference.
  - Backward instability: `dlogits` diff contains NaNs (or non-finite).
  - Occasional device crash: `program continuator has halted unexpectedly`, then host transfers fail.
- W&B run ids (failed):
  - Loss mismatch + NaN grads: `ot0xkbf4`, `ony5axz7`, `gel14eri`, `94duwzx3`, `w6l3g8cl`
  - Device halt / host fetch failure: `erg2jovl`, `ql7sn4qh`
  - Bounds-check halt while experimenting with output accumulation: `y9kkxagp`

### Local reference code inspection

- Cloned reference repos under `workdir/` (gitignored):
  - `workdir/unsloth`
  - `workdir/liger-kernel` (contains a Triton `grpo_loss` forward/backward kernel)
  - `workdir/jax` (Pallas TPU docs/examples)

- Revisions (depth-1 clones):
  - `workdir/unsloth`: `4cb7229` (`https://github.com/unslothai/unsloth.git`)
  - `workdir/liger-kernel`: `9eb9a1e` (`https://github.com/linkedin/Liger-Kernel.git`)
  - `workdir/jax`: `58c0f57` (`https://github.com/jax-ml/jax.git`)

- Liger GRPO kernel references:
  - `workdir/liger-kernel/src/liger_kernel/ops/grpo_loss.py` (fwd+bwd kernels)
  - `workdir/liger-kernel/src/liger_kernel/transformers/grpo_loss.py` (public wrapper)
  - `workdir/liger-kernel/test/transformers/test_grpo_loss.py` (numerical checks)

### Commands (to fill with verified runs)

- Local tests:
  - `python -m pytest -q`
- TPU gradcheck:
  - `<tpu ssh command>`
  - `python -u scripts/grpo_kernel_gradcheck.py --config <yaml>`

### Files added/changed (to update)

- `<to fill>`
