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

### 2) Cross-host FSDP sharding can also make v6e-16 slower

- The default config `mesh_shape: 1,-1,1` becomes `dp=1, fsdp=16, tp=1` on v6e-16, which shards parameters across all 16 chips (spanning hosts).
- For rollout (decode), this can introduce large cross-host collectives and hurt per-step time.
- Prefer keeping `fsdp` **local per worker** and using `dp` across workers.

Recommended v6e-16 mesh for this workload:
- Prefer `mesh_shape: auto` (repo supports this) which resolves to:
  - `dp=process_count` (across workers)
  - `fsdp=local_device_count` (within worker)
  - `tp=1`
- Equivalent explicit mesh on v6e-16: `mesh_shape: 4,4,1` (dp=4, fsdp=4, tp=1)

## Guardrails added (repo changes)

- Multi-host runs should fail fast if only worker 0 is launched:
  - Wrapper: `scripts/tpu_vm_start_grpo_gsm8k_from_config_multihost_nohup.sh` exports `REQUIRE_MULTIHOST=1` and can also set `REQUIRE_JAX_PROCESS_COUNT`.
  - Runner: `plugins/training/runner/grpo_gsm8k.py` errors if `REQUIRE_MULTIHOST=1` but `jax.process_count()==1`.
- Mesh safety and portability:
  - `mesh_shape: auto` is supported by `plugins/training/mesh.py` (host-local on multi-host, preserves single-host behavior).
  - Runner prints a warning when `fsdp` spans hosts (common `mesh_shape: 1,-1,1` pitfall on v6e-16).

## Verified v6e-8 baseline (commit a8197ba; Qwen2.5-3B; bs=128 seq/step; K=8; len=1024)

This is the v6e-8 baseline used for comparison.

- TPU VM: `v6e-8` (1 host), zone `europe-west4-a`
- Env: `ROLLOUT_FAST_GENERATE=1`, `ROLLOUT_FAST_QWEN2_DECODE_ATTENTION=1`, `PRINT_TRAIN_TIME_BREAKDOWN=1`
- Steps 10–19 averages:
  - `avg_dt_last10 ≈ 13.521s`
  - `avg_rollout_generate_last10 ≈ 9.511s`
  - `avg_update_last10 ≈ 3.935s`

Interpretation:
- The step is dominated by `rollout_generate` (decode), not the PPO update.

## Commands actually used (Windows gcloud + PuTTY/plink notes)

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
