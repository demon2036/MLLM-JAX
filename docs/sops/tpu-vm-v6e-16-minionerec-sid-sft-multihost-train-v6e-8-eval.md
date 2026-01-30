# SOP: TPU VM v6e-16 multi-host MiniOneRec SID SFT (train) + TPU VM v6e-8 single-host eval + `calc.py` cross-check (W&B online)

- **Title**: SOP: Run **multi-host** MiniOneRec **SID SFT** training on `v6e-16` (verify `process_count==4`, `device_count==16`), then run **single-host** eval on `v6e-8` to produce `eval_predictions.json` and cross-check HR/NDCG with upstream `workdir/MiniOneRec/calc.py` (W&B online).
  **Prereqs**: `gcloud` authenticated; TPU API enabled; outbound internet from TPU VMs (HF + W&B); local `.env` contains `WANDB_API_KEY` (gitignored) and is synced to TPU.
  **Environment (verified)**:
  - Local: gcloud `470.0.0`, project `civil-rarity-482610-s5`
  - TPU VMs (spot):
    - Train: `mllm-jax-v6e-16-sft-260129014251` (`v6e-16`, multi-host, 4 workers), zone `us-east5-b`
    - Eval: `mllm-jax-v6e-8-sft-eval-260129024610` (`v6e-8`, single-host), zone `us-east5-b`
  - Conda env: `mllm-jax` (Python `3.12.12`)
  - JAX TPU: `jax 0.9.0`, `jaxlib 0.9.0`, `libtpu 0.0.34`
  - Repo: `https://github.com/demon2036/MLLM-JAX.git`, branch `test`
    - SFT train commit: `0f72f26` (HF safetensors hub download retry)
    - SFT eval commit: `8f45868` (bump eval `jax.max_cache_length`)

## Steps (commands actually run)

### A) v6e-16: multi-host SFT train

#### 1) Create `v6e-16` TPU VM (spot)

- `TPU_NAME="mllm-jax-v6e-16-sft-260129014251"; ./scripts/create_tpu_vm.sh --type v6e-16 --zone us-east5-b --name "$TPU_NAME"`

#### 2) Bootstrap Miniconda + conda env on all workers

- `./scripts/bootstrap_miniconda_on_tpu_vm.sh --name "$TPU_NAME" --zone us-east5-b --worker all --env-name mllm-jax --python 3.12`

#### 3) Git-sync repo on all workers (no SCP for code)

- `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east5-b --worker all --command 'set -euo pipefail; REPO_URL=https://github.com/demon2036/MLLM-JAX.git; REPO_DIR=/root/MLLM-JAX; if [ ! -d "$REPO_DIR/.git" ]; then rm -rf "$REPO_DIR"; git clone "$REPO_URL" "$REPO_DIR"; fi; cd "$REPO_DIR"; git fetch --all --prune; git checkout test; git reset --hard origin/test; git clean -fd; git rev-parse --short HEAD'`

#### 4) Install TPU deps on all workers

- `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east5-b --worker all --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; python -m pip install -U pip; python -m pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; python -m pip install -U torch --index-url https://download.pytorch.org/whl/cpu; cd /root/MLLM-JAX; python -m pip install -U -r requirements-tpu.txt'`

#### 5) Clone upstream `MiniOneRec` under `workdir/` (all workers)

- `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east5-b --worker all --command 'set -euo pipefail; cd /root/MLLM-JAX; mkdir -p workdir; if [ ! -d workdir/MiniOneRec/.git ]; then git clone https://github.com/AkaliKong/MiniOneRec workdir/MiniOneRec; fi'`

#### 6) Sync secrets `.env` to TPU (all workers)

- `./scripts/sync_env_to_tpu_vm.sh --name "$TPU_NAME" --zone us-east5-b --src .env --dest /root/.env --worker all`

#### 7) Multi-host JAX sanity check (all workers)

- `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east5-b --worker all --env-file /root/.env --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; python - <<\"PY\"\nimport jax\njax.distributed.initialize()\nprint(\"process\", jax.process_index(), \"/\", jax.process_count())\nprint(\"device_count\", jax.device_count(), \"local\", jax.local_device_count())\nPY'`

#### 8) Start **multi-host SFT train** via nohup (launch on all workers)

- Config: `projects/minionerec/sft/configs/train/v6e-16/sid_sft_jax_qwen25_1p5b_instruct_industrial_v6e16_full.yaml`
- Start (multi-host guards enabled):
  - `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east5-b --worker all --env-file /root/.env --command 'set -euo pipefail; cd /root/MLLM-JAX; bash scripts/tpu_vm_start_sid_sft_from_config_multihost_nohup.sh --env-name mllm-jax --config projects/minionerec/sft/configs/train/v6e-16/sid_sft_jax_qwen25_1p5b_instruct_industrial_v6e16_full.yaml --run-mode train --require-jax-process-count 4'`

#### 9) Check exit codes (all workers)

- `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east5-b --worker all --command 'set -euo pipefail; cd /root/MLLM-JAX; for w in 0 1 2 3; do f=logs/nohup_sid_sft_*w-${w}_latest.exit; echo \"worker=$w exit=$(cat $f)\"; done'`

#### 10) Copy checkpoint to local (artifact SCP) + delete TPU

- Copy (worker 0):
  - `gcloud alpha compute tpus tpu-vm scp "root@${TPU_NAME}:/root/MLLM-JAX/runs/sid_sft_jax_qwen25_1p5b_instruct_industrial_v6e16_full/sft_state_last.msgpack" "workdir/checkpoints/20260129_v6e16_sft/sft_state_last.msgpack" --zone us-east5-b --worker=0`
- Delete:
  - `./scripts/delete_tpu_vm.sh --name "$TPU_NAME" --zone us-east5-b`

### B) v6e-8: single-host eval + `calc.py` cross-check

#### 1) Create `v6e-8` TPU VM (spot)

- `TPU_NAME="mllm-jax-v6e-8-sft-eval-260129024610"; ./scripts/create_tpu_vm.sh --type v6e-8 --zone us-east5-b --name "$TPU_NAME"`

#### 2) Bootstrap + Git-sync + deps + MiniOneRec + `.env` (worker 0)

- Bootstrap:
  - `./scripts/bootstrap_miniconda_on_tpu_vm.sh --name "$TPU_NAME" --zone us-east5-b --worker 0 --env-name mllm-jax --python 3.12`
- Git sync:
  - `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east5-b --worker 0 --command 'set -euo pipefail; REPO_URL=https://github.com/demon2036/MLLM-JAX.git; REPO_DIR=/root/MLLM-JAX; if [ ! -d "$REPO_DIR/.git" ]; then rm -rf "$REPO_DIR"; git clone "$REPO_URL" "$REPO_DIR"; fi; cd "$REPO_DIR"; git fetch --all --prune; git checkout test; git reset --hard origin/test; git clean -fd; git rev-parse --short HEAD'`
- Deps:
  - `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east5-b --worker 0 --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; python -m pip install -U pip; python -m pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; python -m pip install -U torch --index-url https://download.pytorch.org/whl/cpu; cd /root/MLLM-JAX; python -m pip install -U -r requirements-tpu.txt'`
- MiniOneRec:
  - `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east5-b --worker 0 --command 'set -euo pipefail; cd /root/MLLM-JAX; mkdir -p workdir; if [ ! -d workdir/MiniOneRec/.git ]; then git clone https://github.com/AkaliKong/MiniOneRec workdir/MiniOneRec; fi'`
- `.env`:
  - `./scripts/sync_env_to_tpu_vm.sh --name "$TPU_NAME" --zone us-east5-b --src .env --dest /root/.env --worker 0`

#### 3) Upload checkpoint to the expected path on TPU (artifact SCP)

- Create dir:
  - `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east5-b --worker 0 --command 'mkdir -p /root/MLLM-JAX/runs/sid_sft_jax_qwen25_1p5b_instruct_industrial_v6e16_full'`
- Upload:
  - `gcloud alpha compute tpus tpu-vm scp "workdir/checkpoints/20260129_v6e16_sft/sft_state_last.msgpack" "root@${TPU_NAME}:/root/MLLM-JAX/runs/sid_sft_jax_qwen25_1p5b_instruct_industrial_v6e16_full/sft_state_last.msgpack" --zone us-east5-b --worker=0`

#### 4) Run eval (single-host TPU) to produce predictions JSON

- Config: `projects/minionerec/sft/configs/eval/v6e-16/sid_sft_jax_qwen25_1p5b_instruct_industrial_v6e16_eval_from_last.yaml` (requires `jax.max_cache_length>=132`; set to `512` in this repo)
- Start:
  - `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east5-b --worker 0 --env-file /root/.env --command 'cd /root/MLLM-JAX; bash scripts/tpu_vm_start_sid_sft_from_config_nohup.sh --env-name mllm-jax --config projects/minionerec/sft/configs/eval/v6e-16/sid_sft_jax_qwen25_1p5b_instruct_industrial_v6e16_eval_from_last.yaml --run-mode eval'`
- Confirm exit:
  - `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east5-b --worker 0 --command 'cd /root/MLLM-JAX; cat logs/nohup_sid_sft_*eval_from_last*latest.exit'`  # expect `0`

#### 5) Cross-check metrics with upstream `calc.py`

- Install `fire` (MiniOneRec `calc.py` dependency):
  - `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east5-b --worker 0 --command 'source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; python -m pip install -U fire'`
- Run `calc.py`:
  - `./scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east5-b --worker 0 --command 'source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; python workdir/MiniOneRec/calc.py --path runs/sid_sft_jax_qwen25_1p5b_instruct_industrial_v6e16_eval_from_last/eval_predictions.json --item_path workdir/MiniOneRec/data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt'`

#### 6) Delete TPU (stop billing)

- `./scripts/delete_tpu_vm.sh --name "$TPU_NAME" --zone us-east5-b`

## Expected result

- v6e-16 SFT train:
  - Log contains `process=0/4` and `device_count=16 local_device_count=4`.
  - Exit files contain `0` on all workers.
  - W&B run created under `minionerec-sid-sft`.
  - Checkpoint exists on process 0 host:
    - `runs/sid_sft_jax_qwen25_1p5b_instruct_industrial_v6e16_full/sft_state_last.msgpack`
- v6e-8 eval:
  - Exit file contains `0`.
  - Writes:
    - `runs/.../eval_predictions.json`
    - `runs/.../eval_predictions.metrics.json`
  - `workdir/MiniOneRec/calc.py` prints the same HR/NDCG (invalid=0).

## Observed result (this verified run)

- v6e-16 SFT train (multi-host):
  - W&B: https://wandb.ai/johntitordemon2036/minionerec-sid-sft/runs/icaztvzp
  - Exit: `0` (all workers), final `step=780/780`
- v6e-8 eval + calc cross-check:
  - W&B: https://wandb.ai/johntitordemon2036/minionerec-sid-sft/runs/bqjsnfs9
  - `calc.py` (n=4533, invalid=0):
    - HR@K (K=[1,3,5,10,20,50]): `[0.05692, 0.07677, 0.08604, 0.10501, 0.12817, 0.17075]`
    - NDCG@K: `[0.05692, 0.06852, 0.07237, 0.07847, 0.08425, 0.09271]`
  - `calc.py` matches `eval_predictions.metrics.json`.

## Troubleshooting

- **Create fails (capacity)**: try another zone that supports `v6e-16` / `v6e-8` (e.g. `us-east1-d`, `us-east5-b`, `europe-west4-a`).
- **HF Hub flaky / timeouts when loading safetensors**: ensure your branch includes `0f72f26` (retries for `hf_hub_download`).
- **Eval fails with `max_cache_length too small`**: set `jax.max_cache_length` high enough (this SOP uses `512`).
- **`calc.py` fails with `ModuleNotFoundError: fire`**: install `fire` in the TPU conda env.

## References

- SFT runner: `projects/minionerec/sft/runner.py`
- SFT configs:
  - Train: `projects/minionerec/sft/configs/train/v6e-16/sid_sft_jax_qwen25_1p5b_instruct_industrial_v6e16_full.yaml`
  - Eval: `projects/minionerec/sft/configs/eval/v6e-16/sid_sft_jax_qwen25_1p5b_instruct_industrial_v6e16_eval_from_last.yaml`
- Multi-host launcher: `scripts/tpu_vm_start_sid_sft_from_config_multihost_nohup.sh`
- Upstream cross-check: `workdir/MiniOneRec/calc.py`

