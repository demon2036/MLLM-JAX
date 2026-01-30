# SOP: Benchmark GRPO Pallas kernel (v6e-8, single-host) — batch=1 time=4096 vocab=151643

- **Title**: SOP: Run GRPO Pallas kernel benchmark (speed + memory) on TPU v6e-8 with B=1,T=4096,V=151643
  **Prereqs**: Windows PowerShell + `gcloud` installed/authenticated; TPU API enabled; v6e-8 quota; outbound internet on TPU VM
  **Environment (verified)**:
  - Local: Windows PowerShell; gcloud `553.0.0`; account `nitokyo8@gmail.com`; project `civil-rarity-482610-s5`
  - TPU VM: `grpo-pallas-kernel-v6e8-spot-260127230700` (`v6e-8`, spot), zone `us-east5-b`
  - TPU OS: Ubuntu `24.04.x` (kernel `6.11.0-1015-gcp`)
  - Python venv: `/root/venvs/mllm-jax` (Python `3.12.x`)
  - JAX: `0.9.0`, jaxlib `0.9.0`, libtpu `0.0.34`
  - Repo: `https://github.com/demon2036/MLLM-JAX.git`, branch `test_rl`, commit `f2f9bdf`

## Notes

- `us-east5-b` v6e-8 **spot capacity is volatile**; the first VM used in this task was `PREEMPTED` mid-run.
- This bench logs TPU memory via `jax.devices()[0].memory_stats()` (`peak_bytes_reserved`).
- W&B: on this VM, `WANDB_API_KEY` was not available to the Python process; the commands below use `--wandb_mode disabled`.

## Steps (commands actually used)

### 1) Create a v6e-8 spot TPU VM (us-east5-b)

```powershell
$ts = (Get-Date).ToUniversalTime().ToString('yyMMddHHmmss')
$name = "grpo-pallas-kernel-v6e8-spot-$ts"
gcloud alpha compute tpus tpu-vm create $name --project civil-rarity-482610-s5 --zone us-east5-b --accelerator-type v6e-8 --version v6e-ubuntu-2404 --spot --quiet
gcloud alpha compute tpus tpu-vm describe $name --project civil-rarity-482610-s5 --zone us-east5-b --format='value(state,networkEndpoints[0].accessConfig.externalIp)'
```

Record:

- `TPU_NAME=<name>`
- `EXTERNAL_IP=<externalIp>`

### 2) Bootstrap minimal runtime on TPU (venv + jax[tpu] + pytest + wandb)

SSH from Windows using OpenSSH (key created by gcloud at `D:\Users\johntitor.wu\.ssh\google_compute_engine`):

```powershell
$KEY="$env:USERPROFILE\.ssh\google_compute_engine"
ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL root@<EXTERNAL_IP> @'
set -euo pipefail
apt-get update -y
apt-get install -y git python3.12-venv
python3 -m venv /root/venvs/mllm-jax
source /root/venvs/mllm-jax/bin/activate
python -m pip install -U pip
python -m pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python -m pip install -U pytest wandb
'@
```

### 3) Git-sync repo on TPU (no scp for code)

```powershell
$KEY="$env:USERPROFILE\.ssh\google_compute_engine"
ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL root@<EXTERNAL_IP> @'
set -euo pipefail
REPO_URL=https://github.com/demon2036/MLLM-JAX.git
REPO_DIR=/root/MLLM-JAX
if [ ! -d "$REPO_DIR/.git" ]; then git clone "$REPO_URL" "$REPO_DIR"; fi
cd "$REPO_DIR"
git fetch --all --prune
git checkout test_rl
git pull --ff-only
git rev-parse --short HEAD
'@
```

### 4) TPU unit test (pallas lowering)

```powershell
$KEY="$env:USERPROFILE\.ssh\google_compute_engine"
ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL root@<EXTERNAL_IP> @'
set -euo pipefail
cd /root/MLLM-JAX
source /root/venvs/mllm-jax/bin/activate
python -m pytest -q tests/test_grpo_pallas_kernel.py
'@
```

Expected: `3 passed`.

### 5) Benchmark (B=1, T=4096, V=151643, bf16)

Legacy JAX (matches `MLLM_JAX.train_modules.TrainGRPOModule` log_softmax semantics):

```powershell
$KEY="$env:USERPROFILE\.ssh\google_compute_engine"
ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL root@<EXTERNAL_IP> @'
set -euo pipefail
cd /root/MLLM-JAX
source /root/venvs/mllm-jax/bin/activate
python -u scripts/grpo_kernel_bench.py --impl jax --mode off_policy --batch 1 --time 4096 --vocab 151643 --dtype bf16 \
  --iters 3 --warmup 1 --old_logp_noise_scale 0.3 --epsilon_low 0.2 --epsilon_high 0.2 --temperature 1.0 \
  --block_size 2048 --time_block 512 --compute_dtype bf16 --wandb_mode disabled
'@
```

Kernel:

```powershell
$KEY="$env:USERPROFILE\.ssh\google_compute_engine"
ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL root@<EXTERNAL_IP> @'
set -euo pipefail
cd /root/MLLM-JAX
source /root/venvs/mllm-jax/bin/activate
python -u scripts/grpo_kernel_bench.py --impl kernel --batch 1 --time 4096 --vocab 151643 --dtype bf16 \
  --iters 3 --warmup 1 --old_logp_noise_scale 0.3 --epsilon_low 0.2 --epsilon_high 0.2 --temperature 1.0 \
  --block_size 2048 --time_block 512 --wandb_mode disabled
'@
```

Reference:

```powershell
$KEY="$env:USERPROFILE\.ssh\google_compute_engine"
ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL root@<EXTERNAL_IP> @'
set -euo pipefail
cd /root/MLLM-JAX
source /root/venvs/mllm-jax/bin/activate
python -u scripts/grpo_kernel_bench.py --impl ref --batch 1 --time 4096 --vocab 151643 --dtype bf16 \
  --iters 3 --warmup 1 --old_logp_noise_scale 0.3 --epsilon_low 0.2 --epsilon_high 0.2 --temperature 1.0 \
  --block_size 2048 --time_block 512 --wandb_mode disabled
'@
```

#### Observed results (from JSON output)

- **Legacy JAX (`--impl jax`)**: `avg_step_ms=11.9565`, `mem_after_run.peak_bytes_reserved=1242415104`
- **Kernel**: `avg_step_ms=11.8517`, `mem_after_run.peak_bytes_reserved=2516631552`
- **Ref**: `avg_step_ms=41.2104`, `mem_after_run.peak_bytes_reserved=4969938944`

Interpretation:
- Kernel is much faster than the float32 reference (`--impl ref`) but uses **higher peak reserved HBM** than the legacy JAX implementation (`--impl jax`) on this shape.

### 5b) Re-run (2026-01-30, v6e-8, W&B online, commit `ee03e1d`)

Same shape `B=1,T=4096,V=151643`:

- **Legacy JAX (`--impl jax`)**: `avg_step_ms=12.0699`, `mem_after_run.peak_bytes_reserved=1242415104`
- **Kernel (`compute_dtype=bf16`)**: `avg_step_ms=16.4314`, `mem_after_run.peak_bytes_reserved=2516631552`
- **Kernel (`compute_dtype=f32`)**: `avg_step_ms=15.5482`, `mem_after_run.peak_bytes_reserved=2516631552`

Interpretation (as of `ee03e1d`): the kernel is currently **slower** and uses **higher peak reserved HBM** than the legacy JAX path on this benchmark.

### 5c) Re-run (2026-01-30, v6e-8, W&B online, commit `498e293`)

TPU VM (spot):
- Name: `grpo-pallas-kernel-opt-v6e8-spot-260130080500`
- Zone: `us-east5-b`
- Repo: branch `test_rl`, commit `498e293`

W&B:
- Mode: `online` (via `/root/.env`)
- Project: `mllm-jax-grpo-kernel`

Legacy JAX:

```bash
scripts/ssh_tpu_vm_root.sh --name grpo-pallas-kernel-opt-v6e8-spot-260130080500 --zone us-east5-b --env-file /root/.env --command \
  'source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; \
   python -u scripts/grpo_kernel_bench.py --impl jax --mode off_policy --batch 1 --time 4096 --vocab 151643 --dtype bf16 \
     --iters 2 --warmup 1 --old_logp_noise_scale 0.3 --epsilon_low 0.2 --epsilon_high 0.2 --temperature 1.0 \
     --block_size 2048 --time_block 512 --compute_dtype bf16 --wandb_mode online --wandb_name grpo_kernel_bench_jax_off_policy_b1_t4096_v151643_head498e293'
```

Kernel:

```bash
scripts/ssh_tpu_vm_root.sh --name grpo-pallas-kernel-opt-v6e8-spot-260130080500 --zone us-east5-b --env-file /root/.env --command \
  'source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; \
   python -u scripts/grpo_kernel_bench.py --impl kernel --mode off_policy --batch 1 --time 4096 --vocab 151643 --dtype bf16 \
     --iters 2 --warmup 1 --old_logp_noise_scale 0.3 --epsilon_low 0.2 --epsilon_high 0.2 --temperature 1.0 \
     --block_size 2048 --time_block 512 --compute_dtype bf16 --wandb_mode online --wandb_name grpo_kernel_bench_kernel_off_policy_b1_t4096_v151643_head498e293'
```

Observed results (from JSON):

- **Legacy JAX** (run `jlvc5hlq`): `avg_step_ms=11.8582`, `mem_after_run.peak_bytes_reserved=1242415104`
- **Kernel** (run `avj7ns0s`): `avg_step_ms=7.5042`, `mem_after_run.peak_bytes_reserved=2485223424`

Interpretation (as of `498e293`): the kernel is now **faster**, but still reports **higher peak reserved HBM** than the legacy JAX path on this benchmark.

### 6) Delete TPU VM (stop billing)

```powershell
gcloud alpha compute tpus tpu-vm delete <TPU_NAME> --project civil-rarity-482610-s5 --zone us-east5-b --quiet
```

## Expected result

- Kernel is faster and uses lower `peak_bytes_reserved` than reference on the same shape.

## Troubleshooting

- If `gcloud ... create` returns “no more capacity”, retry later or switch zones.
- If the spot VM is `PREEMPTED`, recreate and rerun the steps.
- If Windows `gcloud tpu-vm ssh` gets stuck on PuTTY host-key prompts, use OpenSSH to the external IP as above.
