# SOP: Benchmark GRPO Pallas kernel (speed + memory)

## Prereqs

- TPU VM: `mllm-jax-v6e-8-spot-260124132428` (zone `us-east1-d`)
- TPU repo checkout: `/root/MLLM-JAX`
- Conda env on TPU: `mllm-jax`
- W&B: `WANDB_API_KEY` available (loaded from `/root/.env` by the script); use `--wandb_mode online`

## Steps (commands actually run)

All commands below were run from a Windows host; `echo y |` answers the `plink` host-key prompt.

### 1) Sync repo on TPU

```bash
echo y | gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --worker=0 --quiet --command "set -euo pipefail; cd /root/MLLM-JAX; git fetch --all --prune; git pull --ff-only; git rev-parse --short HEAD"
```

### 2) Benchmark (batch=1, time=2048, vocab=151643, bf16)

Reference:

```bash
echo y | gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --worker=0 --quiet --command "set -euo pipefail; cd /root/MLLM-JAX; /root/miniconda3/bin/conda run -n mllm-jax python -u scripts/grpo_kernel_bench.py --impl ref --batch 1 --time 2048 --vocab 151643 --dtype bf16 --iters 3 --warmup 1 --old_logp_noise_scale 0.3 --epsilon_low 0.2 --epsilon_high 0.2 --temperature 1.0 --block_size 2048"
```

Kernel:

```bash
echo y | gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --worker=0 --quiet --command "set -euo pipefail; cd /root/MLLM-JAX; /root/miniconda3/bin/conda run -n mllm-jax python -u scripts/grpo_kernel_bench.py --impl kernel --batch 1 --time 2048 --vocab 151643 --dtype bf16 --iters 3 --warmup 1 --old_logp_noise_scale 0.3 --epsilon_low 0.2 --epsilon_high 0.2 --temperature 1.0 --block_size 2048 --time_block 512"
```

### 3) Benchmark (batch=4, time=1024, vocab=151643, bf16)

Reference:

```bash
echo y | gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --worker=0 --quiet --command "set -euo pipefail; cd /root/MLLM-JAX; /root/miniconda3/bin/conda run -n mllm-jax python -u scripts/grpo_kernel_bench.py --impl ref --batch 4 --time 1024 --vocab 151643 --dtype bf16 --iters 2 --warmup 1 --old_logp_noise_scale 0.3 --epsilon_low 0.2 --epsilon_high 0.2 --temperature 1.0 --block_size 2048"
```

Kernel:

```bash
echo y | gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --worker=0 --quiet --command "set -euo pipefail; cd /root/MLLM-JAX; /root/miniconda3/bin/conda run -n mllm-jax python -u scripts/grpo_kernel_bench.py --impl kernel --batch 4 --time 1024 --vocab 151643 --dtype bf16 --iters 3 --warmup 1 --old_logp_noise_scale 0.3 --epsilon_low 0.2 --epsilon_high 0.2 --temperature 1.0 --block_size 2048 --time_block 512"
```

### 4) Run TPU unit tests

```bash
echo y | gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --worker=0 --quiet --command "set -euo pipefail; cd /root/MLLM-JAX; /root/miniconda3/bin/conda run -n mllm-jax python -m pytest -q tests/test_grpo_pallas_kernel.py"
```

## Expected result

- Each benchmark prints a JSON blob containing `avg_step_ms` and `mem_after_run.peak_bytes_reserved`.
- Each benchmark prints a W&B run URL and completes without traceback.
- `pytest` prints `3 passed` and exits `0`.

## Troubleshooting

- If Windows `gcloud tpu-vm ssh` keeps prompting for a host key, keep `echo y |` in front of the command.
- If you see TPU lowering errors about BlockSpec tiling for `advantages`, ensure the checkout includes the fix that loads
  `advantages` as a full `(batch, 1)` block and indexes with `pl.program_id(0)`.
- If the kernel is slow, try increasing `--time_block` (must be divisible by 8). `time_block=8` is usually too small for TPU throughput.
