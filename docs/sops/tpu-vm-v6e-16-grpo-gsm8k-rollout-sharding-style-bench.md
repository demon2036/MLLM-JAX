# SOP: TPU v6e-16 GRPO/GSM8K rollout sharding style bench (legacy vs maxtext)

- **Title**: TPU v6e-16 host-local mesh: compare rollout `sharding_style: legacy` vs `maxtext`
  **Prereqs**: gcloud configured; `.env` contains `WANDB_API_KEY`; GitHub access to `demon2036/MLLM-JAX`
  **Scope**: TPU run + W&B evidence (speed + memory)

## What this tests

- Mesh: `mesh_shape: host_local` on `v6e-16` (4 hosts)
  - `dp=process_count=4`, `fsdp=local_device_count=4`, `tp=1`
- Rollout sharding:
  - `legacy`: shard rollout batch across `(dp,fsdp,tp)` product
  - `maxtext`: shard rollout batch across `dp` only (replicate over `fsdp`)

## Steps (commands actually used)

### 1) Create TPU VM

```bash
cd /home/john/workdir/multi-host
scripts/create_tpu_vm.sh --type v6e-16 --zone europe-west4-a --name mllm-jax-v6e-16-spot-260124184415
```

### 2) Bootstrap Miniconda on all TPU workers

```bash
cd /home/john/workdir/multi-host
scripts/bootstrap_miniconda_on_tpu_vm.sh \
  --name mllm-jax-v6e-16-spot-260124184415 \
  --zone europe-west4-a \
  --worker all \
  --env-name mllm-jax \
  --python 3.12
```

### 3) Sync secrets to all TPU workers

```bash
cd /home/john/workdir/multi-host
scripts/sync_env_to_tpu_vm.sh \
  --name mllm-jax-v6e-16-spot-260124184415 \
  --zone europe-west4-a \
  --worker all
```

### 4) Clone repo branch on all TPU workers (Git sync)

```bash
cd /home/john/workdir/multi-host
scripts/ssh_tpu_vm_root.sh \
  --name mllm-jax-v6e-16-spot-260124184415 \
  --zone europe-west4-a \
  --worker all \
  --command 'set -euo pipefail; cd /root; if [ ! -d MLLM-JAX ]; then git clone --branch john/20260124-rollout-multihost-analysis https://github.com/demon2036/MLLM-JAX.git MLLM-JAX; fi; cd /root/MLLM-JAX; git fetch origin; git checkout john/20260124-rollout-multihost-analysis; git pull --ff-only; echo GIT_SHA=$(git rev-parse --short HEAD) BRANCH=$(git branch --show-current)'
```

Observed on TPU: `GIT_SHA=cafe189` on branch `john/20260124-rollout-multihost-analysis`.

### 5) Install TPU Python deps on all TPU workers

```bash
cd /home/john/workdir/multi-host
scripts/ssh_tpu_vm_root.sh \
  --name mllm-jax-v6e-16-spot-260124184415 \
  --zone europe-west4-a \
  --worker all \
  --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; cd /root/MLLM-JAX; pip install -U -r requirements-tpu.txt'
```

### 6) Launch legacy rollout sharding run (multihost, W&B online)

```bash
cd /home/john/workdir/multi-host
scripts/ssh_tpu_vm_root.sh \
  --name mllm-jax-v6e-16-spot-260124184415 \
  --zone europe-west4-a \
  --worker all \
  --command 'set -euo pipefail; cd /root/MLLM-JAX; export PRINT_TRAIN_TIME_BREAKDOWN=1; bash scripts/tpu_vm_start_grpo_gsm8k_from_config_multihost_nohup.sh --require-jax-process-count 4 --config plugins/training/configs/grpo_gsm8k_qwen25_3b_bs64_steps12_v6e16_hostlocal_legacy.yaml'
```

W&B run:
- URL: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k-rollout-sharding-style-bench/runs/jwo1cz6k

Extract metrics from the on-VM summary JSON:

```bash
cd /home/john/workdir/multi-host
scripts/ssh_tpu_vm_root.sh \
  --name mllm-jax-v6e-16-spot-260124184415 \
  --zone europe-west4-a \
  --worker 2 \
  --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; python - <<\"PY\"\nimport json\nfrom pathlib import Path\np=Path(\"wandb/run-20260124_105731-jwo1cz6k/files/wandb-summary.json\")\nobj=json.loads(p.read_text())\nkeys=[\n  \"time/train/step_avg_last10_s\",\n  \"throughput/train/valid_tokens_per_s\",\n  \"memory/train/bytes_in_use_max_gib\",\n  \"memory/train/peak_bytes_in_use_max_gib\",\n]\nfor k in keys:\n  print(k, obj.get(k))\nPY'
```

Observed:
- `time/train/step_avg_last10_s`: `12.777984279200007`
- `throughput/train/valid_tokens_per_s`: `1510.4229563367162`
- `memory/train/bytes_in_use_max_gib`: `7.2652506828308105`
- `memory/train/peak_bytes_in_use_max_gib`: `8.067610263824463`

### 7) Launch maxtext rollout sharding run (multihost, W&B online)

```bash
cd /home/john/workdir/multi-host
scripts/ssh_tpu_vm_root.sh \
  --name mllm-jax-v6e-16-spot-260124184415 \
  --zone europe-west4-a \
  --worker all \
  --command 'set -euo pipefail; cd /root/MLLM-JAX; export PRINT_TRAIN_TIME_BREAKDOWN=1; bash scripts/tpu_vm_start_grpo_gsm8k_from_config_multihost_nohup.sh --require-jax-process-count 4 --config plugins/training/configs/grpo_gsm8k_qwen25_3b_bs64_steps12_v6e16_hostlocal_maxtext.yaml'
```

W&B run:
- URL: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k-rollout-sharding-style-bench/runs/4eir2b4f

Extract metrics from the on-VM summary JSON:

```bash
cd /home/john/workdir/multi-host
scripts/ssh_tpu_vm_root.sh \
  --name mllm-jax-v6e-16-spot-260124184415 \
  --zone europe-west4-a \
  --worker 2 \
  --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; python - <<\"PY\"\nimport json\nfrom pathlib import Path\np=Path(\"wandb/run-20260124_110524-4eir2b4f/files/wandb-summary.json\")\nobj=json.loads(p.read_text())\nkeys=[\n  \"time/train/step_avg_last10_s\",\n  \"throughput/train/valid_tokens_per_s\",\n  \"memory/train/bytes_in_use_max_gib\",\n  \"memory/train/peak_bytes_in_use_max_gib\",\n]\nfor k in keys:\n  print(k, obj.get(k))\nPY'
```

Observed:
- `time/train/step_avg_last10_s`: `13.443090026600043`
- `throughput/train/valid_tokens_per_s`: `1361.5764398761153`
- `memory/train/bytes_in_use_max_gib`: `7.265962600708008`
- `memory/train/peak_bytes_in_use_max_gib`: `8.060301780700684`

### 8) Delete TPU VM

```bash
cd /home/john/workdir/multi-host
scripts/delete_tpu_vm.sh --name mllm-jax-v6e-16-spot-260124184415 --zone europe-west4-a
```

## Summary (this run)

- Speed (step avg last10): `maxtext` was slower by ~`+5.2%` vs `legacy` on this benchmark.
- Memory (bytes_in_use / peak): essentially unchanged in this benchmark (tradeoffs likely cancel: activation replication vs avoided weight all-gather).

