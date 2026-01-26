# SOP: Inspect Unsloth/Liger fused CE (LM head fused?)

- **Title**: SOP: Inspect Unsloth + Liger fused cross-entropy implementations and determine whether LM head is fused
  **Prereqs**: `git`; outbound network; enough disk; (optional) `rg` for faster search
  **Environment (verified)**:
  - Host: Windows (PowerShell)
  - Repo: `D:\Users\johntitor.wu\workdir1\tiled-ce-pallas`

## Goal

- Find the canonical CE implementations in:
  - Unsloth (logits-level Triton CE)
  - Unsloth Zoo (LM-head fused CE helpers)
  - Liger Kernel (logits-level CE + fused-linear-CE wrapper)
- Answer: do they fuse `hidden_states @ lm_head_weight` into the CE loss path, or only fuse CE on logits?

## Steps (commands actually used)

From repo root:

### 1) Clone reference repos into `workdir/` (gitignored)

- `mkdir workdir` (if missing)
- `git clone --depth 1 https://github.com/unslothai/unsloth.git workdir/unsloth`
- `git clone --depth 1 https://github.com/unslothai/unsloth-zoo.git workdir/unsloth-zoo`
- `git clone --depth 1 https://github.com/linkedin/Liger-Kernel.git workdir/liger-kernel`

Record revisions:

- `git -C workdir/unsloth rev-parse --short HEAD`
- `git -C workdir/unsloth-zoo rev-parse --short HEAD`
- `git -C workdir/liger-kernel rev-parse --short HEAD`

### 2) Locate logits-level CE kernels

- Unsloth: `workdir/unsloth/unsloth/kernels/cross_entropy_loss.py`
  - Entry: `fast_cross_entropy_loss` (`workdir/unsloth/unsloth/kernels/cross_entropy_loss.py:423`)
- Liger: `workdir/liger-kernel/src/liger_kernel/ops/cross_entropy.py`
  - Kernel: `liger_cross_entropy_kernel` (`workdir/liger-kernel/src/liger_kernel/ops/cross_entropy.py:16`)

### 3) Locate LM-head fused CE paths

- Unsloth Zoo:
  - `cut_cross_entropy` wrapper: `workdir/unsloth-zoo/unsloth_zoo/loss_utils.py:178` (`fused_linear_cross_entropy`)
  - Chunked fused CE: `workdir/unsloth-zoo/unsloth_zoo/fused_losses/cross_entropy_loss.py:345` (`unsloth_fused_ce_loss`)
- Liger:
  - Wrapper: `workdir/liger-kernel/src/liger_kernel/ops/fused_linear_cross_entropy.py:17`
  - Key evidence:
    - matmul: `logits_chunk = _input_chunk @ weight.t()` (`workdir/liger-kernel/src/liger_kernel/ops/fused_linear_cross_entropy.py:97`)
    - CE kernel call: `liger_cross_entropy_kernel[...]` (`workdir/liger-kernel/src/liger_kernel/ops/fused_linear_cross_entropy.py:145`)

## Expected result

- You can answer precisely:
  - Unsloth’s `fast_cross_entropy_loss` is **logits-level** only.
  - Unsloth Zoo provides **LM-head fused** CE options (either via `cut_cross_entropy` or via chunked `linear+CE`).
  - Liger provides both logits-level CE and a **fused-linear-CE wrapper** (implemented as chunked matmul + Triton CE).

## References

- `memory/20260126_unsloth-liger-fused-ce/README.md`
- `docs/sops/clone-reference-repos-into-workdir.md`
