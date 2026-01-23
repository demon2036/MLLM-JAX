# Memory: v6e-8 vs v6e-16 GRPO multi-host speedup

## Goal

- Explain why TPU `v6e-16` can be much slower than `v6e-8` for GRPO training in this repo.
- Deliver concrete optimizations (mesh + launch guardrails + optional code changes) that **improve v6e-16 throughput**.
- Verify by running **both** `v6e-8` and `v6e-16` with `wandb_mode=online` so results are observable.

## Working hypotheses (from existing SOP)

Primary known failure modes already documented in `docs/sops/tpu-vm-v6e-8-v6e-16-grpo-gsm8k-speed-gap-debug.md`:

1) **Not actually running multi-host on v6e-16**
   - `v6e-16` is **4 workers** (JAX `process_count=4`, `local_device_count=4`, `device_count=16`).
   - If you launch only worker 0, you effectively run on **4 chips**, which can easily look slower than a single-host `v6e-8` (8 chips).

2) **Cross-host FSDP sharding**
   - Default `mesh_shape: 1,-1,1` becomes `dp=1, fsdp=16, tp=1` on v6e-16.
   - That shards params across all 16 chips (spanning hosts) and can make decode-heavy rollout slower due to cross-host collectives.
   - Preferred for this workload: keep `fsdp` local per worker and use `dp` across workers, e.g. `mesh_shape: 4,4,1`.

## Completion criteria

- Repo changes:
  - A safe multihost launcher exists that fails fast if a subset of workers is launched.
  - A v6e-16-friendly mesh is easy to select (config and/or `auto` mesh resolution).
- Validation:
  - `python -m pytest -q` passes locally (exit 0).
  - One `v6e-8` run completes with `wandb_mode=online`.
  - One `v6e-16` run completes with `wandb_mode=online` and shows improved step time vs the misconfigured baseline.

## Evidence log (filled as we run)

- (to be appended) commands, exit codes, W&B run URLs, and any timing deltas.

