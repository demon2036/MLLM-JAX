# SOP: TPU v4-8 GRPO fused-kernel benchmark（B=1, L=4096, V=151936；含 forward + grad）

- **Title**: SOP: Reproduce TPU v4-8 GRPO fused-kernel benchmark (batch=1, seq_len=4096, vocab=151936) with forward + value_and_grad
  **Prereqs**:
  - TPU VM `v4-8` is `READY` and reachable via SSH (see `docs/sops/tpu-vm-create-v4-8-or-v6e-8.md`)
  - This repo is synced to TPU VM via Git, and you can checkout the target commit (see `docs/sops/tpu-vm-repo-sync.md`)
  - TPU VM has a working conda env at `/root/miniconda3/envs/mllm-jax/` (see `docs/sops/tpu-vm-bootstrap.md`)
  - This run was executed **without** `WANDB_API_KEY`, so `WANDB_MODE=disabled` is used throughout (delivery note)
  **Environment (verified in this run)**:
  - Zone: `us-central2-b`
  - TPU name: `grpo-fused-kernel-v4-8-260202013435`
  - Repo commit on TPU: `a9d1f69`
  - JAX: `0.9.0`, jaxlib: `0.9.0` (see benchmark output)

## Steps（commands actually used）

### 0) Create a TPU VM v4-8 (or reuse an existing READY one)

- Follow: `docs/sops/tpu-vm-create-v4-8-or-v6e-8.md`
- Record (this run):
  - `TPU_NAME=grpo-fused-kernel-v4-8-260202013435`
  - `ZONE=us-central2-b`

### 1) Sync repo to TPU VM (Git) and checkout the target commit

- Follow: `docs/sops/tpu-vm-repo-sync.md`
- Checkout commit (this run): `a9d1f69`

### 2) Bootstrap the TPU VM Python/JAX environment

- Follow: `docs/sops/tpu-vm-bootstrap.md`
- This benchmark run uses:
  - Python: `/root/miniconda3/envs/mllm-jax/bin/python`

### 3) Confirm kernel config (BLOCK_T/BLOCK_V) and fused availability

- cmd:

```bash
PJRT_DEVICE=TPU /root/miniconda3/envs/mllm-jax/bin/python - <<"PY"
from tests.grpo_fused_kernel import grpo_fused_pallas
print("BLOCK_T", grpo_fused_pallas.BLOCK_T)
print("BLOCK_V", grpo_fused_pallas.BLOCK_V)
print("FUSED_AVAILABLE", grpo_fused_pallas.FUSED_AVAILABLE)
PY
```

- expected output:

```text
BLOCK_T 128
BLOCK_V 4096
FUSED_AVAILABLE True
```

### 4) Benchmark forward (iters=10)

- cmd:

```bash
PJRT_DEVICE=TPU WANDB_MODE=disabled /root/miniconda3/envs/mllm-jax/bin/python tests/grpo_fused_kernel/bench_grpo_fused.py --mode forward --batch 1 --seq-len 4096 --vocab 151936 --iters 10
```

- expected output:

```text
env:
  jax:    0.9.0
  jaxlib: 0.9.0
  devices (4): tpu:TPU v4:0, tpu:TPU v4:1, tpu:TPU v4:2, tpu:TPU v4:3
config: mode=forward batch=1 seq_len=4096 vocab=151936 iters=10 logits_dtype=bf16
forward memory (compiled):
  ref:
    argument_size_in_bytes: 1247123968 (1.16 GiB)
    output_size_in_bytes: 33280 (32.50 KiB)
    temp_size_in_bytes: 2489641984 (2.32 GiB)
  fused:
    argument_size_in_bytes: 1247140352 (1.16 GiB)
    output_size_in_bytes: 33280 (32.50 KiB)
    temp_size_in_bytes: 1244885504 (1.16 GiB)
forward:
ref:   5.086 ms/iter
fused: 7.272 ms/iter
max|loss_ref-loss_fused| = 2.741814e-06
```

### 5) Benchmark grad (value_and_grad, iters=10)

- cmd:

```bash
PJRT_DEVICE=TPU WANDB_MODE=disabled /root/miniconda3/envs/mllm-jax/bin/python tests/grpo_fused_kernel/bench_grpo_fused.py --mode grad --batch 1 --seq-len 4096 --vocab 151936 --iters 10
```

- expected output:

```text
env:
  jax:    0.9.0
  jaxlib: 0.9.0
  devices (4): tpu:TPU v4:0, tpu:TPU v4:1, tpu:TPU v4:2, tpu:TPU v4:3
config: mode=grad batch=1 seq_len=4096 vocab=151936 iters=10 logits_dtype=bf16
grad memory (compiled):
  ref:
    argument_size_in_bytes: 1247123968 (1.16 GiB)
    output_size_in_bytes: 1247091712 (1.16 GiB)
    temp_size_in_bytes: 4978913280 (4.64 GiB)
  fused:
    argument_size_in_bytes: 1247140352 (1.16 GiB)
    output_size_in_bytes: 1247091712 (1.16 GiB)
    temp_size_in_bytes: 1244788736 (1.16 GiB)
grad (value_and_grad):
ref:   25.171 ms/iter
fused: 13.467 ms/iter
abs(loss_ref-loss_fused) = 2.441406e-04
max|grad_ref-grad_fused| (slice) = 0.000000e+00
```

## Conclusion（达标点）

- forward-only：fused **temp memory 更好**但时间仍慢于 ref（1.16 GiB vs 2.32 GiB；7.272 ms/iter vs 5.086 ms/iter）。
- training 关键路径（`value_and_grad`）：fused **速度与内存均优于 ref**（13.467 ms/iter vs 25.171 ms/iter；1.16 GiB vs 4.64 GiB）——达标。

## References

- `docs/sops/tpu-vm-create-v4-8-or-v6e-8.md`
- `docs/sops/tpu-vm-repo-sync.md`
- `docs/sops/tpu-vm-bootstrap.md`
