# SOP: TPU VM v6e-16 (spot, multi-host) MiniOneRec SID SFT + GRPO/GSM8K (W&B online)

- **Title**: SOP: Run **multi-host** MiniOneRec **SID SFT** (paper-aligned config) on `v6e-16`, then run a short **GRPO/GSM8K** multi-host benchmark; record W&B runs and verify `process_count==4`, `device_count==16`.
  **Prereqs**: `gcloud` authenticated; TPU API enabled; outbound internet from TPU VM (HF + W&B + datasets); local `.env` contains `WANDB_API_KEY` (gitignored) and is synced to TPU.
  **Environment (verified)**:
  - Local: gcloud `470.0.0`, project `civil-rarity-482610-s5`
  - TPU VM: `mllm-jax-v6e-16-sft-rl-260128161330` (`v6e-16`, spot), zone `us-east1-d`, Ubuntu `24.04.2`, kernel `6.11.0-1015-gcp`
  - Conda env: `mllm-jax` (Python `3.12.12`)
  - JAX: `jax 0.9.0`, `jaxlib 0.9.0`, `libtpu 0.0.34`
  - Repo: `https://github.com/demon2036/MLLM-JAX.git`, branch `test`
    - SFT run commit: `0b33fd3`
    - GRPO run commit: `56d43e7` (fixes Transformers `return_tensors="jax"` incompatibility)

## Steps (commands actually run)

### 1) Create `v6e-16` TPU VM (spot)

- Capacity note: `us-east5-b` returned `no more capacity`; `us-east1-d` succeeded.
- Create:
  - `TPU_NAME="mllm-jax-v6e-16-sft-rl-260128161330"; ./scripts/create_tpu_vm.sh --type v6e-16 --zone us-east1-d --name "$TPU_NAME"`

### 2) Bootstrap Miniconda + conda env on all workers

- `./scripts/bootstrap_miniconda_on_tpu_vm.sh --name "$TPU_NAME" --zone us-east1-d --worker all --env-name mllm-jax --python 3.12`

### 3) Git-sync repo on all workers (no SCP for code)

- `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east1-d --worker all --command 'set -euo pipefail; REPO_URL=https://github.com/demon2036/MLLM-JAX.git; REPO_DIR=/root/MLLM-JAX; if [ ! -d "$REPO_DIR/.git" ]; then rm -rf "$REPO_DIR"; git clone "$REPO_URL" "$REPO_DIR"; fi; cd "$REPO_DIR"; git fetch --all --prune; git checkout test; git reset --hard origin/test; git clean -fd; git rev-parse --short HEAD'`

### 4) Install TPU deps on all workers

- `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east1-d --worker all --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; python -m pip install -U pip; python -m pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; python -m pip install -U torch --index-url https://download.pytorch.org/whl/cpu; cd /root/MLLM-JAX; python -m pip install -U -r requirements-tpu.txt'`

### 5) Clone upstream `MiniOneRec` under `workdir/` (all workers)

- `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east1-d --worker all --command 'set -euo pipefail; cd /root/MLLM-JAX; mkdir -p workdir; if [ ! -d workdir/MiniOneRec/.git ]; then git clone https://github.com/AkaliKong/MiniOneRec workdir/MiniOneRec; fi'`

### 6) Sync secrets `.env` to TPU (all workers)

- Verified on this run:
  - `./scripts/sync_env_to_tpu_vm.sh --name "$TPU_NAME" --zone us-east1-d --src /root/github/MLLM-JAX-worktrees/test_rl/.env --dest /root/.env --worker all`

### 7) (Optional) Multi-host JAX sanity check (no training running)

- `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east1-d --worker all --env-file /root/.env --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; python - <<\"PY\"\nimport jax\njax.distributed.initialize()\nprint(\"process\", jax.process_index(), \"/\", jax.process_count())\nprint(\"device_count\", jax.device_count(), \"local\", jax.local_device_count())\nPY'`

### 8) Start **multi-host SFT** via nohup (launch on all workers)

- Config: `projects/minionerec/sft/configs/sid_sft_jax_qwen25_1p5b_instruct_industrial_v6e16_full.yaml` (`mesh_shape: auto`, effective BS=1024, `num_train_epochs: 10`)
- Start (multi-host guards enabled):
  - `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east1-d --worker all --command 'set -euo pipefail; cd /root/MLLM-JAX; bash scripts/tpu_vm_start_sid_sft_from_config_multihost_nohup.sh --env-name mllm-jax --config projects/minionerec/sft/configs/sid_sft_jax_qwen25_1p5b_instruct_industrial_v6e16_full.yaml --run-mode train --require-jax-process-count 4'`

### 9) Monitor SFT logs + exit code

- Note: on this run, **process 0** logs landed on **worker 2** (check the log containing `process=0/4`).
- Example checks:
  - `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east1-d --worker 2 --command 'set -euo pipefail; cd /root/MLLM-JAX; tail -n 5 logs/nohup_sid_sft_*w-2_latest.log'`
  - `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east1-d --worker 2 --command 'set -euo pipefail; cd /root/MLLM-JAX; cat logs/nohup_sid_sft_*w-2_latest.exit'`  # expect `0`

### 10) Start **multi-host GRPO/GSM8K** benchmark (after SFT)

- Config: `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps20_v6e16_bench.yaml` (`wandb_mode: online`, `mesh_shape: auto`)
- Start:
  - `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east1-d --worker all --command 'set -euo pipefail; cd /root/MLLM-JAX; bash scripts/tpu_vm_start_grpo_gsm8k_from_config_multihost_nohup.sh --env-name mllm-jax --config plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps20_v6e16_bench.yaml --require-jax-process-count 4'`

## Expected result

- SFT:
  - Log contains `process=0/4` and `device_count=16 local_device_count=4`.
  - Exit file contains `0`.
  - W&B run is created under `minionerec-sid-sft`.
- GRPO:
  - Log contains `backend=tpu process=0/4` and `device_count=16 local_device_count=4`.
  - Exit file contains `0`.
  - W&B run is created under `mllm-jax-grpo-gsm8k`.

## Observed result (this verified run)

- SFT:
  - Exit file: `0` (all workers)
  - W&B: `https://wandb.ai/johntitordemon2036/minionerec-sid-sft/runs/4ymaq1sr`
- GRPO:
  - Exit file: `0` (all workers)
  - W&B: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/gskr67g8`

## Troubleshooting

- **Create fails (capacity)**: try another zone that supports `v6e-16` (common ones: `us-east1-d`, `us-east5-b`, `europe-west4-a`).
- **Multi-host not actually used**:
  - SFT: `Expected multi-host JAX runtime (REQUIRE_MULTIHOST=1), but got jax.process_count()==1` → you launched only one worker; re-run with `--worker all`.
  - GRPO: same; use `scripts/tpu_vm_start_grpo_gsm8k_from_config_multihost_nohup.sh` and `--worker all`.
- **Process 0 not on worker 0**: locate the log that prints `process=0/4` to find the W&B URL and the checkpoint/output directory for that run.
- **W&B auth errors**: re-sync `/root/.env` to all workers; ensure `WANDB_API_KEY` is valid.

## References

- SFT runner: `projects/minionerec/sft/runner.py`
- SFT config: `projects/minionerec/sft/configs/sid_sft_jax_qwen25_1p5b_instruct_industrial_v6e16_full.yaml`
- GRPO config: `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps20_v6e16_bench.yaml`
- Multi-host launchers: `scripts/tpu_vm_start_sid_sft_from_config_multihost_nohup.sh`, `scripts/tpu_vm_start_grpo_gsm8k_from_config_multihost_nohup.sh`
