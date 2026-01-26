# Unsloth / Liger GRPO fused kernels (LM head fused?)

## Goal

- Inspect how Unsloth and Liger implement “fused GRPO” and whether they fuse the LM head
  (`hidden_states @ lm_head_weight`) into that path.

## Reference repos (gitignored clones)

- `workdir/unsloth` @ `4cb7229` (`https://github.com/unslothai/unsloth.git`)
- `workdir/unsloth-zoo` @ `bb2375e` (`https://github.com/unslothai/unsloth-zoo.git`)
- `workdir/liger-kernel` @ `9eb9a1e` (`https://github.com/linkedin/Liger-Kernel.git`)

## Findings

### Liger

**1) Triton GRPO loss kernel (logits-level; LM head NOT fused)**

- Kernel lives in `workdir/liger-kernel/src/liger_kernel/ops/grpo_loss.py`:
  - `fused_selective_log_softmax` (`workdir/liger-kernel/src/liger_kernel/ops/grpo_loss.py:50`) computes per-token
    logp for selected ids from `logits` (no grad).
  - `_grpo_loss_fwd_kernel` (`workdir/liger-kernel/src/liger_kernel/ops/grpo_loss.py:71`) does online logsumexp over vocab,
    computes `logp`, ratio/clipping, KL penalty, and writes per-token loss.
  - `_grpo_loss_bwd_kernel` (`workdir/liger-kernel/src/liger_kernel/ops/grpo_loss.py:154`) streams vocab again to write `dlogits`.
- TRL integration example still starts from model logits:
  - `logits = model(...).logits` (`workdir/liger-kernel/src/liger_kernel/transformers/grpo_loss.py:101`)
  - `GrpoLossFunction.apply(logits, ...)` (`workdir/liger-kernel/src/liger_kernel/transformers/grpo_loss.py:32`)
- Conclusion: **the fused kernel is on logits**, not on hidden-states/LM-head.

**2) “Fused linear GRPO” (LM head included; but not a single kernel)**

- Implemented via a chunked autograd path (`torch.func.grad_and_value`) in:
  - `workdir/liger-kernel/src/liger_kernel/chunked_loss/grpo_loss.py`
  - Base: `workdir/liger-kernel/src/liger_kernel/chunked_loss/fused_linear_ppo.py`
- It includes LM head by doing:
  - `logits = torch.matmul(input_chunk, weight.t())` (`workdir/liger-kernel/src/liger_kernel/chunked_loss/fused_linear_ppo.py:322`)
  - `log_probs = F.log_softmax(logits.float(), dim=-1)` (`workdir/liger-kernel/src/liger_kernel/chunked_loss/fused_linear_ppo.py:329`)
- Conclusion: **LM head is included**, but it’s chunked matmul + log_softmax (not Triton-fused GEMM+softmax+loss).

### Unsloth (Zoo path)

Unsloth’s GRPO “memory efficient” path avoids returning full logits by switching the model to return hidden states and
computing only the selected-token logprobs.

- In `grpo_accumulated_loss`, it forces “return hidden states”:
  - `os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"` (`workdir/unsloth-zoo/unsloth_zoo/rl_replacements.py:614`)
- Per-token logprobs are computed from hidden states + LM head:
  - `chunk_logits = chunk_hidden_states.to(lm_head.dtype) @ lm_head.t()` (`workdir/unsloth-zoo/unsloth_zoo/rl_replacements.py:86`)
  - returns per-token logps (selective log-softmax) via chunking (`workdir/unsloth-zoo/unsloth_zoo/rl_replacements.py:66`)
- Then the GRPO loss is computed on `new_logprobs` via a custom autograd wrapper:
  - `UnslothEfficientGRPO.apply(...)` (`workdir/unsloth-zoo/unsloth_zoo/rl_replacements.py:901`)
  - and it turns off the hidden-state return flag after (`workdir/unsloth-zoo/unsloth_zoo/rl_replacements.py:917`).

Conclusion:

- Unsloth’s GRPO optimization is **LM-head-aware** (logprobs are computed from `hidden_states @ lm_head.T`),
  but it is implemented as **chunked matmul + logsumexp + gather** under `torch.compile`/autograd, not as a dedicated
  Triton GRPO loss kernel.

## Relation to our JAX kernel

- Our current `plugins/training/kernels/grpo_loss_pallas.py` is a logits-level fused loss+grad kernel (like Liger’s Triton GRPO loss).
- To get the same “don’t materialize logits” behavior as Unsloth/Liger fused-linear variants on TPU,
  we’d need an LM-head-aware kernel (stream vocab tiles and compute loss + `dhidden` directly).
