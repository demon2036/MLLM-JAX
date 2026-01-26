# SOP: Inspect Unsloth/Liger GRPO fused kernels (LM head fused?)

- **Title**: SOP: Inspect Unsloth + Liger GRPO fused implementations and determine whether LM head is fused
  **Prereqs**: `git`; outbound network; enough disk; (optional) `rg`
  **Environment (verified)**:
  - Host: Windows (PowerShell)
  - Repo: `D:\Users\johntitor.wu\workdir1\tiled-ce-pallas`

## Goal

- Identify:
  - Liger’s Triton GRPO loss kernel (logits-level)
  - Liger’s fused-linear GRPO path (LM head included vs not)
  - Unsloth’s GRPO “efficient” path (hidden-states + LM head selective log-softmax)
- Answer: does the fused GRPO kernel fuse LM head (`hidden_states @ W.T`) or operate on logits?

## Steps (commands actually used)

From repo root:

### 1) Clone reference repos (gitignored under `workdir/`)

- `mkdir workdir` (if missing)
- `git clone --depth 1 https://github.com/unslothai/unsloth.git workdir/unsloth`
- `git clone --depth 1 https://github.com/unslothai/unsloth-zoo.git workdir/unsloth-zoo`
- `git clone --depth 1 https://github.com/linkedin/Liger-Kernel.git workdir/liger-kernel`

Record revisions:

- `git -C workdir/unsloth rev-parse --short HEAD`
- `git -C workdir/unsloth-zoo rev-parse --short HEAD`
- `git -C workdir/liger-kernel rev-parse --short HEAD`

### 2) Locate Liger Triton GRPO loss (logits-level)

- Kernel file: `workdir/liger-kernel/src/liger_kernel/ops/grpo_loss.py`
  - Look for: `fused_selective_log_softmax`, `_grpo_loss_fwd_kernel`, `_grpo_loss_bwd_kernel`, `GrpoLossFunction`
- Wrapper/integration: `workdir/liger-kernel/src/liger_kernel/transformers/grpo_loss.py`
  - Look for `GrpoLossFunction.apply(logits, ...)` and the demo using `logits = model(...).logits`.

### 3) Locate Liger fused-linear GRPO (LM head included)

- `workdir/liger-kernel/src/liger_kernel/chunked_loss/grpo_loss.py`
- Base class showing LM head is done via matmul+log_softmax:
  - `workdir/liger-kernel/src/liger_kernel/chunked_loss/fused_linear_ppo.py`

### 4) Locate Unsloth GRPO efficient path (hidden-states + LM head)

- File: `workdir/unsloth-zoo/unsloth_zoo/rl_replacements.py`
  - It flips `UNSLOTH_RETURN_HIDDEN_STATES=1` in `grpo_accumulated_loss`
  - Computes selective log-softmax via chunked `hidden_states @ lm_head.T` (`chunked_hidden_states_selective_log_softmax`)
  - Uses `UnslothEfficientGRPO.apply(...)` for the GRPO loss on logps

## Expected result

- Liger’s Triton GRPO kernel is **logits-level** (no LM head fused).
- Liger has an LM-head-including GRPO path, but implemented as **chunked matmul + log_softmax** (not a single fused kernel).
- Unsloth’s GRPO path is **LM-head-aware** by computing per-token logps from hidden states, but relies on chunking +
  `torch.compile`/autograd (not an explicit Triton GRPO loss kernel).

## References

- `memory/20260126_unsloth-liger-grpo-fused/README.md`
- `docs/sops/clone-reference-repos-into-workdir.md`
