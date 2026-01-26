# Unsloth / Liger fused cross-entropy (LM head fused?)

## Goal

- Compare Unsloth/Liger CE implementations with our current JAX Pallas CE (logits-level) and answer:
  - Do they fuse the LM head (`hidden_states @ lm_head_weight`) into the CE loss path?
  - What “fused” means in their context (single kernel vs chunked).

## Reference repos (gitignored clones)

- `workdir/unsloth` @ `4cb7229` (`https://github.com/unslothai/unsloth.git`)
- `workdir/unsloth-zoo` @ `bb2375e` (`https://github.com/unslothai/unsloth-zoo.git`)
- `workdir/liger-kernel` @ `9eb9a1e` (`https://github.com/linkedin/Liger-Kernel.git`)

## Findings

### Unsloth

- Logits-level fused CE kernel (Triton):
  - Entry: `workdir/unsloth/unsloth/kernels/cross_entropy_loss.py:423` (`fast_cross_entropy_loss`)
  - Input is `logits [B, T, V]` + `labels [B, T]`; **LM head is not fused** here.
  - Backward writes `dlogits` **in-place into the logits buffer** (kernel uses `tl.store(logits_ptr + ...)`).

- LM-head fused CE paths exist, but live in `unsloth_zoo`:
  - `cut_cross_entropy` path: `workdir/unsloth-zoo/unsloth_zoo/loss_utils.py:178` (`fused_linear_cross_entropy`)
    - Calls `cut_cross_entropy.linear_cross_entropy(hidden_states, lm_weight, targets=labels, shift=True, ...)`,
      so **LM head is fused** at the API level (and likely in the underlying op).
  - Chunked fallback path: `workdir/unsloth-zoo/unsloth_zoo/fused_losses/cross_entropy_loss.py:345` (`unsloth_fused_ce_loss`)
    - Splits tokens into chunks based on a target GB, then runs `linear(...)` + `cross_entropy(...)` per chunk using
      `torch.func.grad_and_value` to accumulate `dhidden/dW/db` without ever materializing the full `[BT, V]` logits.
  - HF model integration uses this path to avoid returning logits (example):
    - `workdir/unsloth/unsloth/models/llama.py:1421` calls `unsloth_fused_ce_loss(...)` and returns `EMPTY_LOGITS`.

### Liger

- Logits-level CE kernel (Triton):
  - `workdir/liger-kernel/src/liger_kernel/ops/cross_entropy.py:16` (`liger_cross_entropy_kernel`)
  - Computes loss and also stores gradients **in-place into the logits buffer** (`X_ptr`).

- LM-head “fused linear cross entropy”:
  - Wrapper: `workdir/liger-kernel/src/liger_kernel/ops/fused_linear_cross_entropy.py:17`
  - It **does include LM head** at the API level:
    - Computes `logits_chunk = _input_chunk @ weight.t()` (`workdir/liger-kernel/src/liger_kernel/ops/fused_linear_cross_entropy.py:97`)
    - Runs the CE kernel on `logits_chunk` (in-place grads) (`workdir/liger-kernel/src/liger_kernel/ops/fused_linear_cross_entropy.py:145`)
    - Then computes `grad_input = grad_logits_chunk @ weight` and accumulates `grad_weight`/`grad_bias`.
  - Important nuance: it is **not a single kernel** that fuses GEMM+softmax; it’s **chunked matmul + Triton CE**.
  - HF patch docs describe this as “logits will not be materialized” meaning **full `[BT, V]`** is avoided via chunking
    (`workdir/liger-kernel/src/liger_kernel/transformers/monkey_patch.py:157`).

### Compared to our current JAX Pallas CE

- Current implementation is logits-level only:
  - `plugins/training/kernels/tiled_cross_entropy_pallas.py:71` contract is `logits [B,T,V]` + `labels [B,T]`.
  - Entry: `plugins/training/kernels/tiled_cross_entropy_pallas.py:380` (`cross_entropy_per_token_pallas`).
  - So it corresponds to **Unsloth/Liger logits-level CE**, not their LM-head fused variants.

## Implication

- If we want a large memory win on TPU, we likely need an LM-head-aware kernel (avoid full logits materialization),
  similar in intent to Liger/Unsloth “fused linear CE” — but implemented with Pallas + streaming vocab tiles.
