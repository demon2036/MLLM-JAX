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

- `feat-kernel@0ccacd5`: two-pass (block-stats + JAX reduce) GRPO+entropy Pallas fwd; configs split for `bwd_impl`; TPU BlockSpec fixes (fold blocks axis) + avoid unsupported gather.
- Files:
  - `plugins/training/kernels/grpo_loss_pallas.py`
  - `plugins/training/kernels/entropy_pallas.py`
  - `plugins/training/configs/grpo_kernel_microbench_*`
  - `docs/sops/tpu-vm-v6e-8-grpo-kernel-microbench-t4096-vocab151936.md`

### W&B runs (microbench)

- Baseline (kernel off): https://wandb.ai/johntitordemon2036/mllm-jax-grpo-kernel-microbench/runs/wc4gfkdy
- Kernel (pallas bwd): https://wandb.ai/johntitordemon2036/mllm-jax-grpo-kernel-microbench/runs/2jesd1au
- Kernel (jax bwd): https://wandb.ai/johntitordemon2036/mllm-jax-grpo-kernel-microbench/runs/1l866e2l

### Key metrics (copy from W&B)

- `bench/step_s_mean`:
  - baseline: `0.00901273879717337`
  - pallas bwd: `0.05521106299856911`
  - jax bwd: `0.05293255299620796`
- `mem/peak_bytes_in_use_max`:
  - baseline: `6226795008`
  - pallas bwd: `6226977280`
  - jax bwd: `6227687424`

## Current conclusion

- For logits-level `(B=1,T=4096,V=151936)` microbench, the current Pallas path is **still slower** than the pure-JAX reference, and **does not reduce** `peak_bytes_in_use`.
- The original suspicion holds: without fusing the LM-head matmul + loss (avoiding logits/dlogits materialization), a logits-level Pallas kernel is unlikely to beat XLA's highly optimized softmax/log-softmax lowering on TPU.
