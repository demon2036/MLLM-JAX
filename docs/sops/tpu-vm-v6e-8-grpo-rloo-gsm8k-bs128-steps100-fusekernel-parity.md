# TPU VM v6e-8 GRPO+RLOO/GSM8K 100-step Train: Fusekernel (Pallas) parity + TPU mem (W&B online)

- **Title**: SOP: Run GRPO + RLOO 100 steps on v6e-8 and compare `policy_loss_impl=pallas` vs `jax` (full-eval each epoch, TPU mem)
  **Prereqs**: `gcloud` authenticated; outbound internet (HF + datasets + wandb); W&B API key available locally as `.env`; TPU quota for `v6e-8`

## Target TPU (this run)

- TPU name: `minionerec-sft-subsetdiff-v6e8-euwest4a-260126160555`
- Zone: `europe-west4-a`
- Type: `v6e-8` (spot)
- Repo: `https://github.com/demon2036/MLLM-JAX.git`, branch `test_rl`, commit `febccc4`

## Configs

All configs are “config-driven full eval each epoch” (no env vars for eval), `wandb_mode=online`, and `train.log_tpu_memory=true`.

- GRPO baseline (jax): `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100_evalfull_epoch_jax_mem.yaml`
- GRPO fusekernel (pallas): `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100_evalfull_epoch_pallas_tb512_bf16_mem.yaml`
- (Optional diagnostic) GRPO pallas tb512 f32: `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100_evalfull_epoch_pallas_tb512_f32_mem.yaml`
- RLOO baseline (jax): `plugins/training/configs/rl_gsm8k_qwen25_3b_bs128_steps100_rloo_evalfull_epoch_jax_mem.yaml`
- RLOO fusekernel (pallas): `plugins/training/configs/rl_gsm8k_qwen25_3b_bs128_steps100_rloo_evalfull_epoch_pallas_tb512_f32_mem.yaml`

## Steps (commands actually used)

Set these once:

```bash
TPU_NAME=minionerec-sft-subsetdiff-v6e8-euwest4a-260126160555
ZONE=europe-west4-a
```

### 0) TPU state check (spot TPUs can be PREEMPTED)

```bash
gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" --zone "$ZONE" --format='value(state)'
```

If it returns `PREEMPTED`, recreate it (same name):

```bash
scripts/delete_tpu_vm.sh --name "$TPU_NAME" --zone "$ZONE"
scripts/create_tpu_vm.sh --type v6e-8 --zone "$ZONE" --name "$TPU_NAME" --spot
```

### 1) Sync secrets (W&B key)

Local `.env` must contain `WANDB_API_KEY=...` (do not commit `.env`).

```bash
scripts/sync_env_to_tpu_vm.sh --name "$TPU_NAME" --zone "$ZONE" --worker all
```

### 2) Bootstrap Miniconda + env on TPU

```bash
scripts/bootstrap_miniconda_on_tpu_vm.sh --name "$TPU_NAME" --zone "$ZONE" --env-name mllm-jax --python 3.12
```

### 3) Git sync on TPU (no SCP for code)

```bash
scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --command \
  'set -euo pipefail; REPO_URL=https://github.com/demon2036/MLLM-JAX.git; REPO_DIR=/root/MLLM-JAX; \
   if [ ! -d "$REPO_DIR/.git" ]; then rm -rf "$REPO_DIR"; git clone "$REPO_URL" "$REPO_DIR"; fi; \
   cd "$REPO_DIR"; git fetch --all --prune; git checkout test_rl; git reset --hard origin/test_rl; git clean -fd; \
   echo HEAD=$(git rev-parse --short HEAD)'
```

### 4) Install TPU deps (JAX + requirements)

```bash
scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --command \
  'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; \
   python -m pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; \
   python -m pip install -U -r /root/MLLM-JAX/requirements-tpu.txt; \
   python -c "import jax; print(\"jax\", jax.__version__); print(\"devices\", len(jax.devices()))"'
```

### 5) Launch the 4 runs (nohup, W&B online)

```bash
scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --env-file /root/.env --command \
  'set -euo pipefail; cd /root/MLLM-JAX; export PRINT_TRAIN_TIME_BREAKDOWN=1; \
   bash scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh --env-name mllm-jax --config plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100_evalfull_epoch_jax_mem.yaml'

scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --env-file /root/.env --command \
  'set -euo pipefail; cd /root/MLLM-JAX; export PRINT_TRAIN_TIME_BREAKDOWN=1; \
   bash scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh --env-name mllm-jax --config plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100_evalfull_epoch_pallas_tb512_bf16_mem.yaml'

scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --env-file /root/.env --command \
  'set -euo pipefail; cd /root/MLLM-JAX; export PRINT_TRAIN_TIME_BREAKDOWN=1; \
   bash scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh --env-name mllm-jax --config plugins/training/configs/rl_gsm8k_qwen25_3b_bs128_steps100_rloo_evalfull_epoch_jax_mem.yaml'

scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --env-file /root/.env --command \
  'set -euo pipefail; cd /root/MLLM-JAX; export PRINT_TRAIN_TIME_BREAKDOWN=1; \
   bash scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh --env-name mllm-jax --config plugins/training/configs/rl_gsm8k_qwen25_3b_bs128_steps100_rloo_evalfull_epoch_pallas_tb512_f32_mem.yaml'
```

### 6) Extract final metrics from `wandb-summary.json`

Run directory names are printed in the TPU log as:
`wandb: Run data is saved locally in /root/MLLM-JAX/wandb/run-<timestamp>-<RUN_ID>`

Example (observed):

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mllm-jax
python - <<'PY'
import json
p='wandb/run-20260129_165541-mrgr6e7e/files/wandb-summary.json'
d=json.load(open(p))
for k in ['eval_full/accuracy','time/train/step_avg_last10_s','tpu/mem/peak_bytes_reserved','tpu/mem/peak_bytes_in_use']:
  print(k, d.get(k))
PY
```

## Observed results (2026-01-29, commit `febccc4`)

- GRPO jax (`mrgr6e7e`):
  - `eval_full/accuracy=0.8324488249`
  - `time/train/step_avg_last10_s=10.0389803585`
  - `tpu/mem/peak_bytes_reserved=26440695808`
  - `tpu/mem/peak_bytes_in_use=8840656384`
- GRPO pallas tb512 bf16 (`xa3yztle`):
  - `eval_full/accuracy=0.8597422290` (parity FAIL, +0.0273)
  - `time/train/step_avg_last10_s=14.2539891243` (slower)
  - `tpu/mem/peak_bytes_reserved=26403471360` (slightly lower)
  - `tpu/mem/peak_bytes_in_use=8840656384` (same)
- GRPO pallas tb512 f32 (`tlbt6sjk`, optional diagnostic):
  - `eval_full/accuracy=0.8476118271` (parity FAIL, +0.0152)
  - `time/train/step_avg_last10_s=12.8101634145` (still slower than jax, faster than bf16)
  - `tpu/mem/peak_bytes_reserved=26403471360` (slightly lower)
  - `tpu/mem/peak_bytes_in_use=8840656384` (same)
- RLOO jax (`m7q1v6en`):
  - `eval_full/accuracy=0.8271417741`
  - `time/train/step_avg_last10_s=13.0012286718`
  - `tpu/mem/peak_bytes_reserved=26440695808`
  - `tpu/mem/peak_bytes_in_use=8840656384`
- RLOO pallas tb512 f32 (`j53ceo00`):
  - `eval_full/accuracy=0.8157695224` (parity FAIL, -0.0114)
  - `time/train/step_avg_last10_s=10.4685011274` (faster, but parity broken)
  - `tpu/mem/peak_bytes_reserved=26403471360` (slightly lower)
  - `tpu/mem/peak_bytes_in_use=8840656384` (same)

## Expected result

- All 4 runs finish with exit code `0`.
- W&B runs exist online and contain `eval_full/accuracy`, `time/train/step_avg_last10_s`, and TPU mem metrics.
- If `policy_loss_impl=pallas` is correct, `eval_full/accuracy` should match the `jax` baseline (within an agreed tolerance).

## Troubleshooting

- If SSH fails with `PREEMPTED`: delete + recreate the TPU (spot instances can be reclaimed).
- If W&B init fails: ensure `/root/.env` exists on the TPU and contains `WANDB_API_KEY`, and the YAML has `wandb_mode: online`.

## References

- `scripts/create_tpu_vm.sh`
- `scripts/delete_tpu_vm.sh`
- `scripts/sync_env_to_tpu_vm.sh`
- `scripts/bootstrap_miniconda_on_tpu_vm.sh`
- `scripts/ssh_tpu_vm_root.sh`
- `scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh`
- `plugins/training/configs/*_evalfull_epoch_*_mem.yaml`
