# SOP: Run MiniOneRec SID SFT + eval on TPU (JAX backend)

- **Title**: SOP: Run MiniOneRec SID SFT + constrained-decoding HR@K/NDCG@K eval on TPU via `plugins/sft/` (JAX)
- **Prereqs**: TPU VM reachable via `gcloud ... tpu-vm ssh`; repo synced via Git; local `.env` containing `WANDB_API_KEY` synced to TPU (e.g. `/root/.env`); network access for HF model downloads
- **Environment (verified)**:
  - TPU VM `v4-8` (spot), Ubuntu `22.04.2`, Python `3.12.12` (conda), JAX `0.9.0` + `libtpu 0.0.34`
  - TPU VM `v6e-8` (spot), Ubuntu `24.04.2`, Python `3.12` (conda), JAX `0.9.0` + `jaxlib 0.9.0`

## Steps (commands actually run)

- Create a TPU VM with a task-specific name:
  - `TPU_NAME="minionerec-sid-sft-v4-8-260124110839"; ./scripts/create_tpu_vm.sh --type v4-8 --zone us-central2-b --name "$TPU_NAME"`

- Bootstrap Miniconda + a Python 3.12 env (`mllm-jax`) on the TPU VM:
  - `./scripts/bootstrap_miniconda_on_tpu_vm.sh --name "$TPU_NAME" --zone us-central2-b --project "$(gcloud config get-value project)" --env-name mllm-jax --python 3.12`

- Clone this repo on TPU via Git (no SCP):
  - `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-central2-b --project "$(gcloud config get-value project)" --command 'set -euo pipefail; if [ ! -d /root/MLLM-JAX/.git ]; then git clone https://github.com/demon2036/MLLM-JAX.git /root/MLLM-JAX; fi; cd /root/MLLM-JAX; git fetch --all; git checkout minionerec; git pull; git rev-parse --short HEAD'`

- Install TPU runtime deps (JAX TPU + torch CPU + repo deps):
  - `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-central2-b --project "$(gcloud config get-value project)" --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; python -m pip install -U pip; python -m pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; python -m pip install -U torch --index-url https://download.pytorch.org/whl/cpu; cd /root/MLLM-JAX; python -m pip install -U -r requirements-tpu.txt; python -m pip install -U fire pandas; python - <<\"PY\"\nimport jax, jaxlib\nprint(\"jax\", jax.__version__, \"jaxlib\", jaxlib.__version__)\nprint(\"backend\", jax.default_backend())\nprint(\"process\", jax.process_index(), \"/\", jax.process_count())\nprint(\"device_count\", jax.device_count(), \"local\", len(jax.local_devices()))\nPY'`

- Ensure upstream `MiniOneRec` exists under the repo’s ignored `workdir/`:
  - `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-central2-b --project "$(gcloud config get-value project)" --command 'set -euo pipefail; cd /root/MLLM-JAX; mkdir -p workdir; if [ ! -d workdir/MiniOneRec/.git ]; then git clone https://github.com/AkaliKong/MiniOneRec workdir/MiniOneRec; fi'`

- Sync local `.env` to TPU (do not commit secrets):
  - `./scripts/sync_env_to_tpu_vm.sh --name "$TPU_NAME" --zone us-central2-b --src .env --dest /root/.env --worker all`

- Verify JAX TPU device count (v4-8 reports 4 devices):
  - `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-central2-b --project "$(gcloud config get-value project)" --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; python - <<\"PY\"\nimport jax\nprint(\"device_count\", jax.device_count())\nprint(\"local_device_count\", jax.local_device_count())\nprint(\"devices\", jax.devices())\nPY'`

- Run TPU smoke (JAX backend + W&B online):
  - `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-central2-b --project "$(gcloud config get-value project)" --env-file /root/.env --command 'set -euo pipefail; export PYTHONUNBUFFERED=1; export HF_HUB_ENABLE_HF_TRANSFER=1; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; ./scripts/run_sid_sft.sh --config plugins/sft/configs/sid_sft_jax_smoke_qwen25_1p5b_instruct_industrial_tpu.yaml --run-mode train_eval'`
  - W&B run (online): `https://wandb.ai/johntitordemon2036/minionerec-sid-sft/runs/tkgflo1t`

- Cross-check HR/NDCG with upstream `calc.py` (same predictions JSON):
  - `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-central2-b --project "$(gcloud config get-value project)" --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; python workdir/MiniOneRec/calc.py --path runs/sid_sft_jax_smoke_qwen25_1p5b_instruct_industrial_tpu/eval_predictions.json --item_path workdir/MiniOneRec/data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt'`

## Expected Result

- TPU run exits `0` and writes under `output_dir`:
  - `run_summary.json`
  - `eval_predictions.json`
  - `eval_predictions.metrics.json`
  - `sft_state_last.msgpack` (params-only checkpoint, only when `train.save_last=true`)
- Smoke config prints `effective_bs=1024` on v4-8 (JAX device_count=4, micro=8, accum=32), and W&B logs `train/effective_batch_size=1024`.
  - If `train.logging_steps=1`, W&B also logs `train/step_time_sec` each step.

## Troubleshooting

- TPU busy / `libtpu_lockfile`:
  - Stop the existing job and remove lock: `rm -f /tmp/libtpu_lockfile`
- Very slow eval on TPU:
  - SID eval now pads prompts to the global max prompt length and JIT-compiles once (vector `prompt_true_len`, no prompt-length bucketing). If eval is still slow:
    - For smoke: reduce `data.sample_test` (already `8` in `plugins/sft/configs/sid_sft_jax_smoke_qwen25_1p5b_instruct_industrial_tpu.yaml`).
    - Reduce `data.max_len` / `jax.max_cache_length` / `eval.num_beams` / `eval.batch_size` to cut compile + decode cost.
- `ValueError ... global size ... should be divisible by ...` when placing params:
  - Ensure you are on a recent `minionerec` that prints `[sft] pad_vocab_size ...` (this repo pads vocab to be divisible by `fsdp*tp` and resizes embedding/lm_head).
- Constrained decoding not working (CC > 0 in `calc.py`):
  - Switch to the base model config to avoid Instruct dependency issues: `plugins/sft/configs/sid_sft_jax_smoke_qwen25_1p5b_base_industrial_tpu.yaml`
- v6e-8 queued-resources (flex-start) quota is 0 in `us-central2-b` (example failures):
  - `gcloud alpha compute tpus queued-resources create minionerec-sid-sft-v6e-8-flex-260124121715 --zone=us-central2-b --accelerator-type=v6e-8 --runtime-version=v6e-ubuntu-2404 --node-id=minionerec-sid-sft-v6e-8-flex-260124121715-node --provisioning-model=flex-start --max-run-duration=3600s --async`
  - `gcloud alpha compute tpus queued-resources create minionerec-sid-sft-v6e-8-guaranteed-260124121906 --zone=us-central2-b --accelerator-type=v6e-8 --runtime-version=v6e-ubuntu-2404 --node-id=minionerec-sid-sft-v6e-8-guaranteed-260124121906-node --guaranteed --async`
- v6e-8 eval `RESOURCE_EXHAUSTED` (constrained beam search):
  - Reduce `eval.batch_size` (start with `1`) and reduce `jax.max_cache_length` (Industrial test prompts fit within `256`, so `512` is safe).
- Missing `workdir/MiniOneRec` data:
  - Re-run the `git clone https://github.com/AkaliKong/MiniOneRec workdir/MiniOneRec` step

## Extra: Eval official MiniOneRec HF checkpoints (v6e-8, full test)

- Download checkpoints on TPU (only once per VM):
  - `scripts/ssh_tpu_vm_root.sh --name minionerec-sid-sft-v6e-8-official-eval-260124163806 --zone us-east5-b --env-file /root/.env --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; export HF_HUB_ENABLE_HF_TRANSFER=1; cd /root/MLLM-JAX; python - <<\"PY\"\nfrom huggingface_hub import snapshot_download\nsnapshot_download(repo_id=\"kkknight/MiniOneRec\", allow_patterns=[\"Industrial_ckpt/*\"], local_dir=\"workdir/hf_ckpts/kkknight_MiniOneRec\")\nsnapshot_download(repo_id=\"kkknight/MiniOneRec\", allow_patterns=[\"Office_ckpt/*\"], local_dir=\"workdir/hf_ckpts/kkknight_MiniOneRec\")\nPY'`

- Ensure upstream `calc.py` deps exist on TPU:
  - `scripts/ssh_tpu_vm_root.sh --name minionerec-sid-sft-v6e-8-official-eval-260124163806 --zone us-east5-b --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; python -m pip install -U fire'`

- Run eval (Industrial):
  - `scripts/ssh_tpu_vm_root.sh --name minionerec-sid-sft-v6e-8-official-eval-260124163806 --zone us-east5-b --env-file /root/.env --command 'set -euo pipefail; export PYTHONUNBUFFERED=1; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; ./scripts/run_sid_sft.sh --config plugins/sft/configs/sid_sft_jax_eval_official_minionerec_industrial_ckpt.yaml --run-mode eval'`
  - Cross-check via upstream `calc.py`:
    - `scripts/ssh_tpu_vm_root.sh --name minionerec-sid-sft-v6e-8-official-eval-260124163806 --zone us-east5-b --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; python workdir/MiniOneRec/calc.py --path runs/sid_sft_jax_eval_official_minionerec_industrial_ckpt/eval_predictions.json --item_path workdir/MiniOneRec/data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt'`

- Run eval (Office):
  - `scripts/ssh_tpu_vm_root.sh --name minionerec-sid-sft-v6e-8-official-eval-260124163806 --zone us-east5-b --env-file /root/.env --command 'set -euo pipefail; export PYTHONUNBUFFERED=1; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; ./scripts/run_sid_sft.sh --config plugins/sft/configs/sid_sft_jax_eval_official_minionerec_office_ckpt.yaml --run-mode eval'`
  - Cross-check via upstream `calc.py`:
    - `scripts/ssh_tpu_vm_root.sh --name minionerec-sid-sft-v6e-8-official-eval-260124163806 --zone us-east5-b --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; python workdir/MiniOneRec/calc.py --path runs/sid_sft_jax_eval_official_minionerec_office_ckpt/eval_predictions.json --item_path workdir/MiniOneRec/data/Amazon/info/Office_Products_5_2016-10-2018-11.txt'`

## Extra: Measure SFT train step time (v6e-8)

- Note: JAX training requires static batch shapes; training batches are padded to fixed `data.max_len` (commit `2b5208a`) to avoid JIT recompiles from dynamic padding.

- Create TPU (spot):
  - `TPU_NAME="minionerec-sid-sft-step-time-v6e-8-260124174627"; ./scripts/create_tpu_vm.sh --type v6e-8 --zone us-east5-b --name "$TPU_NAME"`

- Bootstrap + install deps + clone repo + clone upstream MiniOneRec:
  - Follow the same steps above (just replace `--zone` and `--type`); ensure the repo branch is `minionerec`.

- Run a 3-step timing job (len=512, effective_bs=1024, eval disabled):
  - `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east5-b --env-file /root/.env --command 'set -euo pipefail; export PYTHONUNBUFFERED=1; export HF_HUB_ENABLE_HF_TRANSFER=1; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; ./scripts/run_sid_sft.sh --config plugins/sft/configs/sid_sft_jax_qwen25_1p5b_instruct_industrial_v6e8_step_time.yaml --run-mode train'`
  - W&B run: `https://wandb.ai/johntitordemon2036/minionerec-sid-sft/runs/wcgcvyua`
  - Observed per-step timing (W&B `train/step_time_sec`):
    - step1: `113.19s` (includes JIT compile on this warm VM)
    - step2: `4.05s`
    - step3: `4.04s`

## References

- Upstream metrics script: `workdir/MiniOneRec/calc.py`
- Plugin entrypoints: `scripts/run_sid_sft.py`, `plugins/sft/runner/sid_sft.py`
