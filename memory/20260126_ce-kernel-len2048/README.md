# CE Pallas kernel @ len/time=2048 (TPU v6e-8)

## Goal

- Validate CE Pallas kernel correctness at `seq_len=2048` (gradcheck script) and measure speed/memory at `time=2048`.

## Shapes (this run)

- Gradcheck script (`scripts/cross_entropy_kernel_gradcheck.py`):
  - `input_ids`: `[B, seq_len]` = `[1, 2048]`
  - `logits`: `[B, seq_len, V]` = `[1, 2048, 151643]`
  - Kernel inputs:
    - `logits_for_loss = logits[:, :-1, :]`: `[B, T, V]` = `[1, 2047, 151643]`
    - `labels = input_ids[:, 1:]`: `[B, T]` = `[1, 2047]`
  - Kernel outputs:
    - `per_token_loss`: `[B, T]`
    - `per_token_logps`: `[B, T]`

- Microbench (`scripts/cross_entropy_kernel_bench.py`):
  - Inputs: `logits [1, 2048, 151643]`, `labels [1, 2048]`
  - Output: scalar `loss` (+ `dlogits` for `value_and_grad`)

## TPU gradcheck (W&B online)

- TPU VM: `mllm-jax-v6e-8-spot-260124132428` (zone `us-east1-d`)
- Repo checkout: `4e0e898` (`tiled-ce-pallas`)
- Config used: `plugins/training/configs/cross_entropy_kernel_gradcheck_qwen25_1p5b_len2048.yaml`
- W&B run: `https://wandb.ai/johntitordemon2036/mllm-jax-ce-kernel/runs/tr12ugeg`

### Command (Windows OpenSSH)

```bash
ssh -i D:\Users\johntitor.wu\.ssh\google_compute_engine -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL root@34.23.119.201 \
  "set -euo pipefail; cd /root/MLLM-JAX; set -a; if [ -f /root/.env ]; then . /root/.env; fi; set +a; /root/venvs/mllm-jax/bin/python -u scripts/cross_entropy_kernel_gradcheck.py --config plugins/training/configs/cross_entropy_kernel_gradcheck_qwen25_1p5b_len2048.yaml"
```

### Result (exit 0)

- `abs_diff_loss`: `0.0`
- `fwd/logp_max_abs`: `1.9073486328125e-06`
- `dlogits_max_abs`: `7.450580596923828e-09`
- `dlogits_max_rel`: `0.007387231569737196`
- `time/forward_s`: `5.395`
- `time/ref_grad_s`: `0.463`
- `time/kernel_grad_s`: `0.327`

## Speed + memory microbench (synthetic logits, time=2048, vocab=151643)

Implementation: run `value_and_grad` on a scalar loss; `logits` are bf16 zeros to avoid random-gen peak inflating HBM stats.

### Reference (`cross_entropy_per_token_reference`)

- `first_call_s`: `0.483`
- `avg_step_ms` (5 iters): `19.95`
- `mem_after_alloc.bytes_in_use`: `622,082,048` (~593 MiB)  (just logits)
- `mem_after_run.peak_bytes_in_use`: `1,865,430,528` (~1,779 MiB)
- `mem_after_run.bytes_reserved`: `2,485,075,968` (~2,370 MiB)

### Kernel (`cross_entropy_per_token_pallas`, `block_size=2048`, `time_block=8`)

- `first_call_s`: `0.385`
- `avg_step_ms` (5 iters): `22.54`
- `mem_after_alloc.bytes_in_use`: `622,082,048` (~593 MiB)
- `mem_after_run.peak_bytes_in_use`: `1,867,724,288` (~1,781 MiB)
- `mem_after_run.bytes_reserved`: `1,258,356,736` (~1,200 MiB)

### Kernel `time_block` sweep (`block_size=2048`)

Same shapes, `--iters 5 --warmup 1`, single core (`TPU_0`):

| `time_block` | `avg_step_ms` |
| --- | ---: |
| 8 | 22.54 |
| 16 | 14.37 |
| 32 | 10.10 |
| 64 | 7.81 |
| 128 | 6.60 |
| 256 | 6.15 |
| 512 | 5.95 |

`time_block` increases compilation time (bigger tile) but improves steady-state throughput by reducing per-program overhead.

### Interpretation (time_block=8)

- Peak HBM in-use is dominated by `logits` + `dlogits` (both paths), so this benchmark doesn’t show a large
  `peak_bytes_in_use` win from the kernel at this scale.
- The kernel does reserve less memory in the allocator in this run (`bytes_reserved` lower).
- The kernel’s **first-call** time is lower, but **steady-state** step time is slower than the reference here.

### Interpretation (tuning)

- `time_block=8` is too small for TPU throughput; `time_block>=32` is faster than the reference in this setup.
- Best tested here: `time_block=512` at ~`5.95ms` vs reference ~`19.95ms` (same shapes, bf16 logits, single core).
