# GRPO Pallas kernel bench (speed + memory) + batch>1 TPU lowering fix

## Goal

- Add a reproducible benchmark comparing:
  - Reference GRPO loss (`jax.nn.log_softmax` + gather) vs
  - Pallas GRPO loss kernel (`plugins/training/kernels/grpo_loss_pallas.py`)
- Capture wall-clock latency (`value_and_grad`) and TPU HBM `device.memory_stats()`.
- Ensure TPU mosaic lowering works for `batch>1` (not just interpret mode).

## Completion criteria

- TPU runs exit `0` with W&B `wandb_mode=online`.
- Bench script prints JSON including:
  - `avg_step_ms`, `first_call_s`
  - `mem_after_alloc`, `mem_after_run` (including `peak_bytes_reserved`)
- TPU unit test includes a non-interpret (mosaic lowering) case and passes.

## Work log (evidence)

### What was added

- New benchmark script: `scripts/grpo_kernel_bench.py` (committed as `450e53c`)

### TPU mosaic lowering: batch>1 fixes

- Symptom: running the kernel (interpret=False) with `batch>1` raised TPU lowering errors about BlockSpec constraints
  for `advantages` (rank-1 / small block shapes).
- Fix strategy:
  - Reshape `advantages` to rank-2 (`[B, 1]`) and map as a full-block load `(B, 1)` in the BlockSpec.
  - Index inside the kernel using `pid_b = pl.program_id(0)`.
- Commits:
  - `8107b3c`: initial reshape + TPU-only non-interpret test
  - `e51d160`: adjust BlockSpec to satisfy TPU tiling rules (final fix)

### TPU environment

- TPU VM: `mllm-jax-v6e-8-spot-260124132428` (zone `us-east1-d`)
- Repo: `/root/MLLM-JAX` @ `e51d160` (initial bench + lowering fix)
- Conda env: `mllm-jax` (Python `3.12.12`, JAX/jaxlib `0.9.0/0.9.0`)

### Commands actually run (TPU)

Windows host uses `echo y |` to answer `plink` host-key prompts.

- Sync repo:
  - `echo y | gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --worker=0 --quiet --command "set -euo pipefail; cd /root/MLLM-JAX; git fetch --all --prune; git pull --ff-only; git rev-parse --short HEAD"`

- Bench (batch=1, time=2048, vocab=151643, bf16):
  - `echo y | gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --worker=0 --quiet --command "set -euo pipefail; cd /root/MLLM-JAX; /root/miniconda3/bin/conda run -n mllm-jax python -u scripts/grpo_kernel_bench.py --impl ref --batch 1 --time 2048 --vocab 151643 --dtype bf16 --iters 3 --warmup 1 --old_logp_noise_scale 0.3 --epsilon_low 0.2 --epsilon_high 0.2 --temperature 1.0 --block_size 2048"`
    - W&B: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-kernel/runs/py9thh4v`
    - `avg_step_ms=20.4027`, `peak_bytes_reserved=2485075968`
  - `echo y | gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --worker=0 --quiet --command "set -euo pipefail; cd /root/MLLM-JAX; /root/miniconda3/bin/conda run -n mllm-jax python -u scripts/grpo_kernel_bench.py --impl kernel --batch 1 --time 2048 --vocab 151643 --dtype bf16 --iters 3 --warmup 1 --old_logp_noise_scale 0.3 --epsilon_low 0.2 --epsilon_high 0.2 --temperature 1.0 --block_size 2048"`
    - W&B: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-kernel/runs/vf7oejuq`
    - `avg_step_ms=23.1274`, `peak_bytes_reserved=1258340352`

- Bench (batch=4, time=1024, vocab=151643, bf16):
  - `echo y | gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --worker=0 --quiet --command "set -euo pipefail; cd /root/MLLM-JAX; /root/miniconda3/bin/conda run -n mllm-jax python -u scripts/grpo_kernel_bench.py --impl ref --batch 4 --time 1024 --vocab 151643 --dtype bf16 --iters 2 --warmup 1 --old_logp_noise_scale 0.3 --epsilon_low 0.2 --epsilon_high 0.2 --temperature 1.0 --block_size 2048"`
    - W&B: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-kernel/runs/vese3nvh`
    - `avg_step_ms=8.7252`, `peak_bytes_reserved=2484649984`
  - `echo y | gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --worker=0 --quiet --command "set -euo pipefail; cd /root/MLLM-JAX; /root/miniconda3/bin/conda run -n mllm-jax python -u scripts/grpo_kernel_bench.py --impl kernel --batch 4 --time 1024 --vocab 151643 --dtype bf16 --iters 2 --warmup 1 --old_logp_noise_scale 0.3 --epsilon_low 0.2 --epsilon_high 0.2 --temperature 1.0 --block_size 2048"`
    - W&B: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-kernel/runs/trk0hl7n`
    - `avg_step_ms=46.2062`, `peak_bytes_reserved=2516647936`

- TPU unit tests (includes non-interpret TPU-lowering test):
  - `echo y | gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --worker=0 --quiet --command "set -euo pipefail; cd /root/MLLM-JAX; /root/miniconda3/bin/conda run -n mllm-jax python -m pytest -q tests/test_grpo_pallas_kernel.py"`
    - Exit `0`: `3 passed, 1 warning`

## Notes

- The initial kernel used a hardcoded `time_block=8` (like a too-small query block in SplashAttention), which adds a lot of
  per-program overhead. After making `time_block` tunable, throughput improves by ~3–4x at the same `block_size`.
- For correctness diffs vs reference on a real model (Qwen2.5-1.5B), see `memory/20260125_grpo-pallas-kernel/README.md`.

## Update: make `time_block` tunable (TPU perf tuning)

### Change

- `GRPOKernelConfig` now includes `time_block` (commit: `c5227d3`).
- `scripts/grpo_kernel_bench.py` now accepts `--time_block`.

### TPU environment (this update)

- Repo: `/root/MLLM-JAX` @ `c5227d3`

### Commands actually run (TPU, additional)

- Reference (batch=1, time=2048):
  - `echo y | gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --worker=0 --quiet --command "set -euo pipefail; cd /root/MLLM-JAX; /root/miniconda3/bin/conda run -n mllm-jax python -u scripts/grpo_kernel_bench.py --impl ref --batch 1 --time 2048 --vocab 151643 --dtype bf16 --iters 3 --warmup 1 --old_logp_noise_scale 0.3 --epsilon_low 0.2 --epsilon_high 0.2 --temperature 1.0 --block_size 2048"`
    - W&B: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-kernel/runs/dpnjhmrg`
    - `avg_step_ms=20.3351`, `peak_bytes_reserved=2485075968`

- Kernel `time_block` sweep (batch=1, time=2048):
  - `--time_block 8`
    - W&B: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-kernel/runs/bsyxgdiw`
    - `avg_step_ms=23.1246`, `peak_bytes_reserved=1258340352`
  - `--time_block 64`
    - W&B: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-kernel/runs/rq4hvunw`
    - `avg_step_ms=7.8725`, `peak_bytes_reserved=1258340352`
  - `--time_block 512`
    - W&B: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-kernel/runs/5zoxdi68`
    - `avg_step_ms=5.9955`, `peak_bytes_reserved=1258340352`

- Reference (batch=4, time=1024):
  - W&B: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-kernel/runs/30xm6rdq`
  - `avg_step_ms=8.6500`, `peak_bytes_reserved=2484649984`

- Kernel `time_block` sweep (batch=4, time=1024, block_size=2048):
  - `--time_block 8`
    - W&B: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-kernel/runs/5g444ard`
    - `avg_step_ms=46.3271`
  - `--time_block 32`
    - W&B: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-kernel/runs/pclu2elo`
    - `avg_step_ms=20.7081`
  - `--time_block 64`
    - W&B: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-kernel/runs/aom5j8up`
    - `avg_step_ms=16.2729`
  - `--time_block 128`
    - W&B: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-kernel/runs/zlhbd6yk`
    - `avg_step_ms=13.8715`
  - `--time_block 256`
    - W&B: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-kernel/runs/7wsqagxk`
    - `avg_step_ms=12.9438`
  - `--time_block 512`
    - W&B: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-kernel/runs/xzk8g27e`
    - `avg_step_ms=12.6092`
  - `--time_block 1024`
    - W&B: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-kernel/runs/u0zz87kr`
    - `avg_step_ms=12.4994`

- Kernel (batch=4, time=1024, block_size=4096, time_block=512):
  - W&B: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-kernel/runs/dslwhn9w`
  - `avg_step_ms=12.4916`
