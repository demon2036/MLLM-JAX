# TPU VM v6e-8 vs v6e-16 GRPO/GSM8K speed gap debug (multihost launch + mesh + baseline)

- **Title**: SOP: Diagnose why `v6e-16` can look much slower than `v6e-8` and how to eliminate the gap (GRPO/GSM8K, Qwen2.5-3B, len=1024, K=8)
  **Prereqs**: `gcloud` authenticated; TPU API enabled; outbound internet from TPU VM (HF + datasets); if using W&B, `WANDB_API_KEY` must be valid
  **Scope**: `plugins/training/runner/grpo_gsm8k.py`, `scripts/tpu_vm_start_grpo_gsm8k_from_config_multihost_nohup.sh`, v6e-16 config/script

## What we learned (root cause)

### 1) v6e-16 is multi-host (4 workers), not 2

- A `v6e-16` TPU VM is **4 workers** (JAX `process_count=4`, `local_device_count=4`, `device_count=16`).
- If you only launch worker 0, you only get **4 chips**, which can easily make the run feel dramatically slower than a `v6e-8` (8 chips).

This explains the “should be ~2× faster but feels ~8× slower” pattern when combined with “each step compute became ~8×”:
- Only launching 1/4 workers gives **4 chips**, which is **2× fewer chips than v6e-8** (8 chips).
- If per-chip v6e performance is ~2×, that cancels out.
- Net: you effectively see roughly the **full ~8× compute inflation** as wall-clock slowdown (e.g. ~13s → ~102s).

### 2) Host-local sharding can make v6e-16 slower for rollout-heavy GRPO

- In this repo, `mesh_shape: host_local` builds a host-local physical mesh:
  - `dp=process_count` (across workers)
  - `fsdp=local_device_count` (within a worker)
  - `tp=1`
- On `v6e-16`, that becomes `dp=4, fsdp=4, tp=1`, which **reduces parameter sharding**
  (fsdp=4 instead of 16) and was measured to be much slower for this decode-dominated
  GRPO/GSM8K benchmark.
- In contrast, full-device FSDP (`mesh_shape: 1,-1,1` ⇒ `dp=1, fsdp=16, tp=1`) was faster
  despite spanning hosts, because it maximizes model sharding/parallelism for rollout decode.

Recommended v6e-16 mesh for this workload:
- Prefer `mesh_shape: auto` (in this repo), which resolves to **full-device FSDP**
  (equivalent to `mesh_shape: 1,-1,1`).
- Use `mesh_shape: host_local` only if you explicitly want DP across hosts and accept lower
  model sharding throughput (it is kept as an explicit option for experiments).

## Guardrails added (repo changes)

- Multi-host runs should fail fast if only worker 0 is launched:
  - Wrapper: `scripts/tpu_vm_start_grpo_gsm8k_from_config_multihost_nohup.sh` exports `REQUIRE_MULTIHOST=1` and can also set `REQUIRE_JAX_PROCESS_COUNT`.
  - Runner: `plugins/training/runner/grpo_gsm8k.py` errors if `REQUIRE_MULTIHOST=1` but `jax.process_count()==1`.
- Mesh safety and portability:
  - `mesh_shape: auto` is supported by `plugins/training/mesh.py` and resolves to full-device FSDP (`1,-1,1`) on both single-host and multi-host.
  - `mesh_shape: host_local` is supported by `plugins/training/mesh.py` and provides an explicit host-local layout on multi-host pods.
  - Runner prints a warning when `fsdp` spans hosts so you can see when cross-host collectives are in play (and decide whether to experiment with `host_local`).

## Verified benchmarks (W&B online; Qwen2.5-3B; bs=128 seq/step; K=8; len=1024)

- `v6e-8` is **single-host** in this project (JAX `process_count=1`, `local_device_count=8`, `device_count=8`).
- `v6e-16` is **multi-host** (4 workers; 16 chips).
- Repo git SHA used for the runs below: `92fe39b` (branch `multi-host`).

### v6e-8 (single-host) benchmark

- W&B run: `potc8br6` (config: `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps12_v6e8_bench.yaml`)
  - URL: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/potc8br6
  - `time/train/step_avg_last10_s`: `15.1143`
  - `throughput/train/valid_tokens_per_s`: `2293.045`

### v6e-16 (multi-host) benchmark (fixed vs slow reference)

- Fixed (fast): `aqhfh8oo` (auto ⇒ full-device FSDP)
  - URL: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/aqhfh8oo
  - `time/train/step_avg_last10_s`: `10.0358`
  - `throughput/train/valid_tokens_per_s`: `4144.308`

- Slow reference: `on2okepg` (previous auto behavior ≈ host-local sharding)
  - URL: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/on2okepg
  - `time/train/step_avg_last10_s`: `18.6275`
  - `throughput/train/valid_tokens_per_s`: `1879.873`

Interpretation:
- The step is dominated by `rollout_generate` (decode), not the PPO update.

## Commands actually used (Linux, this iteration)

### 0) Create v6e-8 TPU (spot) + wait for READY

On the local machine:

```bash
cd /home/john/workdir/multi-host
scripts/create_tpu_vm.sh --type v6e-8 --zone europe-west4-a --name mllm-jax-v6e-8-spot-260124030148
gcloud alpha compute tpus tpu-vm list --zone europe-west4-a --format='table(name,acceleratorType,state,health)'
```

Optional (useful when `--async` is used, or when you want the exact failure reason even if the node never becomes visible):

```bash
TOKEN="$(gcloud auth print-access-token)"
curl -sS -H "Authorization: Bearer ${TOKEN}" \
  "https://tpu.googleapis.com/v2alpha1/projects/civil-rarity-482610-s5/locations/<ZONE>/operations/<OP>" | jq '{done: .done, error: .error}'
```

### 1) Bootstrap Python env + sync secrets

```bash
scripts/bootstrap_miniconda_on_tpu_vm.sh --name mllm-jax-v6e-8-spot-260124030148 --zone europe-west4-a --worker all --env-name mllm-jax --python 3.12
scripts/sync_env_to_tpu_vm.sh --name mllm-jax-v6e-8-spot-260124030148 --zone europe-west4-a --worker all
```

### 2) Clone repo on TPU + install deps

On the TPU VM (worker 0):

```bash
cd /root
git clone --branch multi-host https://github.com/demon2036/MLLM-JAX.git MLLM-JAX

source /root/miniconda3/etc/profile.d/conda.sh
conda activate mllm-jax
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

cd /root/MLLM-JAX
pip install -U -r requirements-tpu.txt
```

### 3) Launch v6e-8 benchmark (W&B online) + check exit code

```bash
cd /root/MLLM-JAX
export PRINT_TRAIN_TIME_BREAKDOWN=1
export ROLLOUT_FAST_GENERATE=1
export ROLLOUT_FAST_QWEN2_DECODE_ATTENTION=1
bash scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh --config plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps12_v6e8_bench.yaml

tail -n 50 logs/nohup_grpo_gsm8k_qwen25_3b_bs128_steps12_v6e8_bench_latest.log
cat logs/nohup_grpo_gsm8k_qwen25_3b_bs128_steps12_v6e8_bench_latest.exit
```

### 4) Delete TPU (avoid billing)

```bash
cd /home/john/workdir/multi-host
scripts/delete_tpu_vm.sh --name mllm-jax-v6e-8-spot-260124030148 --zone europe-west4-a
```

## Commands (legacy: Windows gcloud + PuTTY/plink notes)

### 0) List v6e-8 TPUs

```powershell
gcloud alpha compute tpus tpu-vm list --zone europe-west4-a --format='table(name,acceleratorType,state,health)'
```

### 1) v6e-8 baseline run (20 steps; W&B disabled)

On the TPU VM (worker 0):

```bash
cd /root/MLLM-JAX
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export STEPS=20
export PRINT_TRAIN_TIME_BREAKDOWN=1
export ROLLOUT_FAST_GENERATE=1
export ROLLOUT_FAST_QWEN2_DECODE_ATTENTION=1
bash scripts/tpu_vm_start_grpo_gsm8k_qwen25_3b_bs128_steps100_nohup.sh
```

Then:

```bash
tail -n 50 logs/nohup_grpo_gsm8k_qwen25_3b_bs128_steps100_latest.log
cat logs/nohup_grpo_gsm8k_qwen25_3b_bs128_steps100_latest.exit
```

### 2) Create v6e-16 (notes)

The following zone(s) were observed to support `v6e-16` accelerator type in this project:
- `us-east1-d`, `us-east5-b`, `europe-west4-a`, ...

We observed:
- On-demand `v6e-16` often fails with `Insufficient capacity`.
- Spot `v6e-16` can be `PREEMPTED` quickly in `us-east1-d`.

Example create command (spot):

```bash
gcloud alpha compute tpus tpu-vm create mllm-jax-v6e-16-<TS> \
  --project civil-rarity-482610-s5 \
  --zone us-east1-d \
  --accelerator-type v6e-16 \
  --version v6e-ubuntu-2404 \
  --spot \
  --quiet
```

### 3) (Windows) Getting host key fingerprints for plink batch mode

On Windows gcloud may use `plink.exe`. In batch mode, you must pass allowed host keys.

Run once to print fingerprints (it will fail, but prints SHA256 fingerprints):

```powershell
gcloud alpha compute tpus tpu-vm ssh root@<TPU_NAME> --zone <ZONE> --worker=all --quiet --command 'hostname' --ssh-flag=-batch
```

Then re-run with all 4 fingerprints:

```powershell
gcloud alpha compute tpus tpu-vm ssh root@<TPU_NAME> --zone <ZONE> --worker=all --quiet --command 'hostname' `
  --ssh-flag=-batch `
  --ssh-flag=-hostkey --ssh-flag=SHA256:<...> `
  --ssh-flag=-hostkey --ssh-flag=SHA256:<...> `
  --ssh-flag=-hostkey --ssh-flag=SHA256:<...> `
  --ssh-flag=-hostkey --ssh-flag=SHA256:<...>
```

### 4) v6e-16 run (recommended)

- Use the multihost wrapper + a config that sets `mesh_shape: auto`:
  - Short benchmark (eval disabled): `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps20_v6e16_bench.yaml`
  - Longer run (100 steps): `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml`
  - Launcher: `bash scripts/tpu_vm_start_grpo_gsm8k_from_config_multihost_nohup.sh --config <path>.yaml --require-jax-process-count 4`
- Always launch on all workers (`--worker=all`).
- Do not use env vars to override hyperparams (this repo’s launcher intentionally ignores them); instead use a YAML with `eval_every_steps: 0` for timing runs.

Expected log header (process 0):
- `backend=tpu process=0/4`
- `device_count=16 local_device_count=4`

## Troubleshooting

- `Expected a multi-host JAX runtime (REQUIRE_MULTIHOST=1) but got jax.process_count()==1`:
  - You launched only worker 0. Re-run with `--worker=all`.
- `Reservation not found` during create:
  - In this project, some zones return this when trying to create v6e TPUs without a reservation.
  - Fix: retry in a different zone (and/or use a reservation if you have one).
- `wandb ... 401 ... user is not logged in`:
  - The API key is invalid/expired OR points to a different W&B host.
  - Fix with `wandb login --relogin` (and sync `/root/.env` to all workers if using dotenv).
- Spot TPU becomes `PREEMPTED`:
  - Use on-demand or a reservation, or retry later in another zone.

## References

- `scripts/tpu_vm_start_grpo_gsm8k_from_config_multihost_nohup.sh`
- `scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh`
- `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml` (`mesh_shape: auto`)
- `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps20_v6e16_bench_badmesh.yaml`
- `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps20_v6e16_bench.yaml`
- `plugins/training/runner/grpo_gsm8k.py`
