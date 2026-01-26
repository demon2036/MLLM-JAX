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

### Root cause (fixed)

- The GRPO reference used `jax.nn.log_softmax` directly on BF16 logits, and JAX returns BF16 logps on TPU.
  Our Pallas kernel computes logsumexp/logp in FP32, so the reference drifted noticeably at large vocab sizes.
- Fix: cast logits to `float32` in `_selective_log_softmax_reference` so the reference numerics match the kernel.
  - Commit: `d7aae9e` (`fix: GRPO reference uses f32 log_softmax`)

### TPU gradcheck (completed)

- TPU VM: `mllm-jax-v6e-8-spot-260124132428` (zone `us-east1-d`)
- Repo: `/root/MLLM-JAX` @ `d7aae9e`
- Conda env: `mllm-jax` (Python `3.12.12`, JAX/jaxlib `0.9.0/0.9.0`)
- W&B run (online): `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-kernel/runs/j71299gc`

Commands actually run (Windows host via `gcloud`, note the `echo y |` is for non-interactive host-key prompt):

- Sync repo to the TPU checkout:
  - `echo y | gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --worker=0 --quiet --command "set -euo pipefail; cd /root/MLLM-JAX; git fetch --all --prune; git pull --ff-only; git rev-parse --short HEAD"`
- Run the logits-level GRPO kernel gradcheck:
  - `echo y | gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --worker=0 --quiet --command "set -euo pipefail; cd /root/MLLM-JAX; /root/miniconda3/bin/conda run -n mllm-jax python -u scripts/grpo_kernel_gradcheck.py --config plugins/training/configs/grpo_kernel_gradcheck_qwen25_1p5b.yaml"`
- Install pytest (to run repo unit tests in the TPU env):
  - `echo y | gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --worker=0 --quiet --command "set -euo pipefail; /root/miniconda3/bin/conda run -n mllm-jax pip install -U pytest"`
- Run the kernel unit test:
  - `echo y | gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --worker=0 --quiet --command "set -euo pipefail; cd /root/MLLM-JAX; /root/miniconda3/bin/conda run -n mllm-jax python -m pytest -q tests/test_grpo_pallas_kernel.py"`

Result (exit `0` + tight diffs printed by the script):

- `abs_diff_loss`: `2.384185791015625e-07`
- `fwd/logp_max_abs`: `1.9073486328125e-06`
- `dlogits_max_abs`: `1.4901161193847656e-08`
- `dlogits_max_rel`: `0.006879059597849846`
- `kernel/block_size`: `2048`
- `shape/vocab`: `151643` (Qwen2.5-1.5B tokenizer vocab used by the gradcheck script)
- `pytest -q tests/test_grpo_pallas_kernel.py`: `2 passed, 1 warning` (transparent hugepages warning)

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

Note: local pytest was not run from this Windows workstation (no local JAX). TPU validation is recorded above.

### Files added/changed (to update)

- `plugins/training/kernels/grpo_loss_pallas.py` (reference uses FP32 log_softmax for TPU equivalence)
- `docs/sops/grpo-pallas-kernel-gradcheck.md` (filled with verified TPU commands + environment)

## Next: fused-linear GRPO (Unsloth-style, no logits materialization)

If we want Unsloth/Liger "fused linear" behavior (compute logp/loss from `hidden_states @ lm_head.T` without building
`[B,T,V]` logits), we need an LM-head-aware kernel.

High-level design (TPU/Pallas):

- Inputs: `hidden_states [B,T,H]`, `lm_head_kernel [H,V]`, `chosen_ids [B,T]`, `old_per_token_logps [B,T]`, `advantages [B]`.
- Forward:
  - Stream over vocab tiles; for each vocab tile compute `logits_tile = hidden_states @ W_tile` using a Pallas matmul
    pattern that also tiles `H` (see `jax.experimental.pallas.ops.tpu.matmul`).
  - Maintain running `(m, l)` (max and sumexp) per token for stable logsumexp; capture `chosen_logit` when the tile
    contains the chosen id.
  - Output `per_token_logp = (chosen_logit - (log(l) + m)) / temperature` and GRPO per-token loss.
- Backward:
  - Re-scan vocab tiles to recompute `logits_tile` and `probs = exp(logits_tile - lse)`.
  - Compute `dlogp` from PPO-style clipping, then accumulate:
    - `dhidden = (onehot - probs) @ W_tile.T * (dlogp/temperature)` (streaming over vocab tiles)
  - Optional: `dlm_head_kernel` is possible but needs a separate reduction strategy over tokens/time (or a dedicated
    kernel that owns `(H_tile, V_tile)` and reduces across `[B,T]`).
