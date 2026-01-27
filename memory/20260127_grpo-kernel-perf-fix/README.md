# GRPO Pallas kernel perf+memory fix (TPU microbench proof)

## Goal

- Fix logits-level GRPO kernel so it is:
  - numerically aligned with the repo's JAX reference, and
  - measurably faster **and** lower-peak-HBM than the pure-JAX baseline on TPU
    for per-device shape `(B=1, T=4096, V=151936)`.
- Keep multi-host/multi-device compatibility via `jax.shard_map` (no invasive upstream edits).

## Completion criteria

- TPU microbench runs (W&B `online`) exist for the same shape/config knobs:
  - baseline: `kernel.enabled=false`
  - kernel: `kernel.enabled=true` with `bwd_impl=pallas` and `bwd_impl=jax` variants
- Both speed + memory are better than baseline (record below):
  - `bench/step_s_mean` lower than baseline
  - `mem/peak_bytes_in_use_max` lower than baseline
- No traceback / non-zero exit on TPU.
- SOP updated with the exact commands actually run.

## Evidence (to be filled after TPU runs)

### Code changes

- `<commit sha>`: `<summary>`
- Files:
  - `plugins/training/kernels/grpo_loss_pallas.py`
  - `plugins/training/kernels/entropy_pallas.py`
  - `plugins/training/configs/grpo_kernel_microbench_*`
  - `docs/sops/<new sop>.md`

### W&B runs (microbench)

- Baseline (kernel off): `<wandb link>`
- Kernel (pallas bwd): `<wandb link>`
- Kernel (jax bwd): `<wandb link>`

### Key metrics (copy from W&B)

- `bench/step_s_mean`:
  - baseline: `<value>`
  - pallas bwd: `<value>`
  - jax bwd: `<value>`
- `mem/peak_bytes_in_use_max`:
  - baseline: `<value>`
  - pallas bwd: `<value>`
  - jax bwd: `<value>`

