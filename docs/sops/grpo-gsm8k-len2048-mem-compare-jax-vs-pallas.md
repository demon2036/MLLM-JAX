# GRPO/GSM8K: len=2048 memory sweep — JAX vs Pallas (v6e-8, W&B online)

- **Title**: SOP: Sweep GRPO/GSM8K `max_length_sample=2048` memory cliff (mb=3 fit, mb=4 OOM) for `policy_loss_impl=jax` vs `pallas` on v6e-8 (W&B online)
  **Prereqs**: `gcloud` authenticated; a v6e-8 TPU VM exists; `/root/.env` on the TPU VM contains `WANDB_API_KEY`; repo branch pushed

## What this validates

- Gate before memory experiments:
  - Numeric correctness: `pytest -q tests/test_grpo_pallas_kernel.py` (tolerance `1e-4`)
  - Kernel speed sanity: `scripts/grpo_kernel_bench.py` (kernel faster than JAX baseline on `B=1,T=4096,V=151643`)
- End-to-end GRPO memory sweep at `rollout.max_length_sample=2048`:
  - `micro_batch_size_per_device=3` (fits on v6e-8 for Qwen2.5-3B)
  - `micro_batch_size_per_device=4` (OOM on v6e-8 for Qwen2.5-3B; **JAX OOM margin is much larger than Pallas**)

## Target TPU VM (this run)

- TPU name: `grpo-pallas-kernel-opt-v6e8-spot-260130080500`
- Zone: `us-east5-b`
- Type: `v6e-8`
- Repo branch: `test_rl`
- Repo commit (configs + latest OOM delta): `86da4e7`

## Steps (commands actually used)

### 0) Git sync on TPU

```bash
scripts/ssh_tpu_vm_root.sh --name grpo-pallas-kernel-opt-v6e8-spot-260130080500 --zone us-east5-b --command \
  'set -euo pipefail; cd /root/MLLM-JAX; git fetch --all --prune; git checkout test_rl; git pull --ff-only; git rev-parse --short HEAD'
```

### 1) Unit tests (TPU)

```bash
scripts/ssh_tpu_vm_root.sh --name grpo-pallas-kernel-opt-v6e8-spot-260130080500 --zone us-east5-b --command \
  'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; \
   rm -f /tmp/libtpu_lockfile || true; python -m pytest -q tests/test_grpo_pallas_kernel.py'
```

### 2) Micro-bench gate (W&B online)

```bash
scripts/ssh_tpu_vm_root.sh --name grpo-pallas-kernel-opt-v6e8-spot-260130080500 --zone us-east5-b --env-file /root/.env --command \
  'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; \
   rm -f /tmp/libtpu_lockfile || true; \
   python -u scripts/grpo_kernel_bench.py --impl jax --mode off_policy --batch 1 --time 4096 --vocab 151643 --dtype bf16 --iters 1 --warmup 0 \
     --old_logp_noise_scale 0.3 --epsilon_low 0.2 --epsilon_high 0.2 --temperature 1.0 --block_size 2048 --time_block 512 --compute_dtype bf16 \
     --wandb_mode online --wandb_name grpo_kernel_bench_jax_off_policy_b1_t4096_v151643_headA3B8B45; \
   python -u scripts/grpo_kernel_bench.py --impl kernel --mode off_policy --batch 1 --time 4096 --vocab 151643 --dtype bf16 --iters 1 --warmup 0 \
     --old_logp_noise_scale 0.3 --epsilon_low 0.2 --epsilon_high 0.2 --temperature 1.0 --block_size 2048 --time_block 512 --compute_dtype bf16 \
     --wandb_mode online --wandb_name grpo_kernel_bench_kernel_off_policy_b1_t4096_v151643_headA3B8B45'
```

### 3) End-to-end sweep (W&B online, len=2048)

Environment flags used (non-hyperparam):
- `HF_HUB_ENABLE_HF_TRANSFER=1`
- `TOKENIZERS_PARALLELISM=false`
- `ROLLOUT_FAST_GENERATE=1`
- `ROLLOUT_FAST_QWEN2_DECODE_ATTENTION=1`

#### 3a) mb=3/dev (fits): Pallas vs JAX

Configs (bs=96 global sequences per step):
- Pallas: `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs96_steps2_len2048_pallas_tb512_bf16_mem_v6e8_mb3.yaml`
- JAX: `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs96_steps2_len2048_jax_mem_v6e8_mb3.yaml`

```bash
scripts/ssh_tpu_vm_root.sh --name grpo-pallas-kernel-opt-v6e8-spot-260130080500 --zone us-east5-b --env-file /root/.env --command \
  'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; \
   rm -f /tmp/libtpu_lockfile || true; export HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false ROLLOUT_FAST_GENERATE=1 ROLLOUT_FAST_QWEN2_DECODE_ATTENTION=1; \
   python -u scripts/run_grpo_gsm8k_training.py --config plugins/training/configs/grpo_gsm8k_qwen25_3b_bs96_steps2_len2048_pallas_tb512_bf16_mem_v6e8_mb3.yaml'

scripts/ssh_tpu_vm_root.sh --name grpo-pallas-kernel-opt-v6e8-spot-260130080500 --zone us-east5-b --env-file /root/.env --command \
  'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; \
   rm -f /tmp/libtpu_lockfile || true; export HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false ROLLOUT_FAST_GENERATE=1 ROLLOUT_FAST_QWEN2_DECODE_ATTENTION=1; \
   python -u scripts/run_grpo_gsm8k_training.py --config plugins/training/configs/grpo_gsm8k_qwen25_3b_bs96_steps2_len2048_jax_mem_v6e8_mb3.yaml'
```

#### 3b) mb=4/dev (OOM): Pallas vs JAX

Configs (bs=96 global sequences per step):
- Pallas: `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs96_steps2_len2048_pallas_tb512_bf16_mem_v6e8_mb4.yaml`
- JAX: `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs96_steps2_len2048_jax_mem_v6e8_mb4.yaml`

```bash
scripts/ssh_tpu_vm_root.sh --name grpo-pallas-kernel-opt-v6e8-spot-260130080500 --zone us-east5-b --env-file /root/.env --command \
  'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; \
   rm -f /tmp/libtpu_lockfile || true; export HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false ROLLOUT_FAST_GENERATE=1 ROLLOUT_FAST_QWEN2_DECODE_ATTENTION=1; \
   python -u scripts/run_grpo_gsm8k_training.py --config plugins/training/configs/grpo_gsm8k_qwen25_3b_bs96_steps2_len2048_pallas_tb512_bf16_mem_v6e8_mb4.yaml'

scripts/ssh_tpu_vm_root.sh --name grpo-pallas-kernel-opt-v6e8-spot-260130080500 --zone us-east5-b --env-file /root/.env --command \
  'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; \
   rm -f /tmp/libtpu_lockfile || true; export HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false ROLLOUT_FAST_GENERATE=1 ROLLOUT_FAST_QWEN2_DECODE_ATTENTION=1; \
   python -u scripts/run_grpo_gsm8k_training.py --config plugins/training/configs/grpo_gsm8k_qwen25_3b_bs96_steps2_len2048_jax_mem_v6e8_mb4.yaml'
```

### 4) Compare memory keys from completed mb=3 runs

```bash
scripts/ssh_tpu_vm_root.sh --name grpo-pallas-kernel-opt-v6e8-spot-260130080500 --zone us-east5-b --command \
  'set -euo pipefail; cd /root/MLLM-JAX; \
   for d in wandb/run-*-w198z9dw wandb/run-*-9dvnzlik; do echo \"== $d\"; \
     jq -r \"\\\"tpu/mem/peak_bytes_reserved=\\\" + ((.\\\"tpu/mem/peak_bytes_reserved\\\" // null)|tostring) + \\\" tpu/mem/peak_bytes_in_use=\\\" + ((.\\\"tpu/mem/peak_bytes_in_use\\\" // null)|tostring)\" \
     $d/files/wandb-summary.json; \
   done'
```

## Observed results (2026-01-30)

### Micro-bench gate (B=1,T=4096,V=151643)

- JAX run: `jyn4oy3c` (`avg_step_ms≈11.78`, `peak_bytes_reserved≈1.24GB`)
- Kernel run: `9dtbi97v` (`avg_step_ms≈7.58`, `peak_bytes_reserved≈2.49GB`)

### End-to-end len2048 sweep (Qwen2.5-3B, v6e-8)

- mb=3/dev (bs=96) **fits** for both:
  - Pallas: `w198z9dw`
  - JAX: `9dvnzlik`
  - `tpu/mem/peak_bytes_in_use` was identical (`8429703680`) while `peak_bytes_reserved` was slightly higher for Pallas.
- mb=4/dev (bs=96) **OOMs for both** during compilation, but with very different margins:
  - Pallas: `hcxmj9fi` — exceeded HBM by **18.80M**
  - JAX: `qerilbpe` — exceeded HBM by **3.58G** (total usage `>=35.08G`)
  - Interpretation: under the same mb4/dev setting, Pallas is **~3.56G closer to fitting** than the legacy JAX path; remaining gap is ~**19MB**.

## Update (2026-01-31): further attempts to close the remaining ~19MB

Status:
- Kernel tuning sweeps (`pallas_block_size` / `pallas_time_block`) did **not** change the mb4 compile-OOM margin (still **18.80M**) except `pallas_time_block=128`, which fails Mosaic compilation (`offset not aligned to sublanes`).
- Some alternative knobs (remat/optimizer/MLP checkpoint) changed XLA accounting to include large `arguments` buffers, resulting in **larger** OOM margins (not usable as a fix).
- `XLA_FLAGS=--xla_multiheap_size_constraint_per_heap=1073741824` is **not viable** on v6e in this setup (rollout fails with a MegaChip `UNIMPLEMENTED` error).

W&B runs (v6e-8, online):
- Pallas `block_size=1024` mb4/dev: `nbkoqc76` — exceeded HBM by **18.80M**
- Pallas `time_block=256` mb4/dev: `zit6d56f` — exceeded HBM by **18.80M**
- Pallas `time_block=128` mb4/dev: `qd3a6ffm` — Mosaic error `offset not aligned to sublanes`
- Remat `nothing_saveable` mb4/dev: `z7m13akl` — exceeded HBM by **2.48G**
- Optimizer `sgd` mb4/dev: `65f7kdii` — exceeded HBM by **1.07G**
- MLP checkpoint patch mb4/dev: `9qo5sd95` — exceeded HBM by **2.45G**
- `XLA_FLAGS=--xla_multiheap_size_constraint_per_heap=1073741824`: `jrnbj6o7` / `oh64awms` — MegaChip `UNIMPLEMENTED` during rollout

## References

- Kernel micro-bench: `scripts/grpo_kernel_bench.py`
- GRPO runner: `plugins/training/runner/grpo_gsm8k.py`
- Loss kernel: `plugins/training/kernels/grpo_loss_pallas.py`
