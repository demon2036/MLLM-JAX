# SOP: Run MiniOneRec SID SFT + eval on TPU (JAX backend)

- **Title**: SOP: Run MiniOneRec SID SFT + constrained-decoding HR@K/NDCG@K eval on TPU via `plugins/sft/` (JAX)
- **Prereqs**: TPU VM reachable via `gcloud ... tpu-vm ssh`; repo synced via Git; `WANDB_API_KEY` available on TPU (e.g. `/root/.env`); network access for HF model downloads
- **Environment (to be verified)**: TPU VM (e.g. v4-8); Python 3.12; JAX TPU runtime

## Steps (placeholders until verified)

- Pick the TPU target:
  - `TPU_NAME=<TPU_NAME>; ZONE=<ZONE>; PROJECT=<PROJECT>`

- Sync repo to TPU via Git (no SCP):
  - `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --project "$PROJECT" --command 'cd /root/MLLM-JAX && git fetch --all && git checkout main && git pull'`

- Ensure upstream `MiniOneRec` exists under the repo’s ignored `workdir/`:
  - `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --project "$PROJECT" --command 'set -euo pipefail; cd /root/MLLM-JAX; mkdir -p workdir; if [ ! -d workdir/MiniOneRec/.git ]; then git clone https://github.com/AkaliKong/MiniOneRec workdir/MiniOneRec; fi'`

- Ensure Python env has required deps (JAX TPU + repo requirements):
  - `<TO_BE_FILLED_AFTER_RUN>`

- Run TPU smoke (JAX backend + W&B online):
  - `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --project "$PROJECT" --env-file /root/.env --command 'set -euo pipefail; cd /root/MLLM-JAX; ./scripts/run_sid_sft.sh --config plugins/sft/configs/sid_sft_jax_smoke_qwen25_base_industrial_tpu.yaml --run-mode train_eval'`

- Optional: run a longer TPU config:
  - `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --project "$PROJECT" --env-file /root/.env --command 'set -euo pipefail; cd /root/MLLM-JAX; ./scripts/run_sid_sft.sh --config plugins/sft/configs/sid_sft_jax_qwen25_base_industrial_tpu.yaml --run-mode train_eval'`

## Expected Result

- TPU run exits `0` and writes under `output_dir`:
  - `run_summary.json`
  - `eval_predictions.json`
  - `eval_predictions.metrics.json`
  - `sft_state_last.msgpack` (params-only checkpoint)

## Troubleshooting

- TPU busy / `libtpu_lockfile`:
  - Stop the existing job and remove lock: `rm -f /tmp/libtpu_lockfile`
- Missing `workdir/MiniOneRec` data:
  - Re-run the `git clone https://github.com/AkaliKong/MiniOneRec workdir/MiniOneRec` step

## References

- Upstream metrics script: `workdir/MiniOneRec/calc.py`
- Plugin entrypoints: `scripts/run_sid_sft.py`, `plugins/sft/runner/sid_sft.py`

