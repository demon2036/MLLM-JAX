# Memory: MiniOneRec SFT on TPU (JAX implementation)

## Goal

- Replace the HF/Trainer-based `plugins/sft` SFT pipeline with a TPU-runnable JAX implementation.
- Keep upstream `workdir/MiniOneRec/` untouched; all overrides live in `plugins/`.
- Ensure evaluation HR@K / NDCG@K matches MiniOneRec `workdir/MiniOneRec/calc.py` semantics.
- Provide deterministic, config-driven entrypoints + SOP for TPU VM runs.

## Completion Criteria

- `python -m pytest -q` exits 0.
- Local CPU smoke run completes end-to-end (train → eval → metrics) with exit 0.
- TPU VM smoke run completes end-to-end with exit 0 (when TPU is available).
- Outputs include a machine-readable summary JSON + eval metrics JSON.
- SOP updated with commands *actually run*.

## Evidence (append as executed)

- Commands + exit codes:
  - `python -m pytest -q` (exit=0)
  - `./scripts/run_sid_sft.sh --config plugins/sft/configs/sid_sft_jax_smoke_tiny_llama.yaml --run-mode train_eval` (exit=0)
  - v6e queued-resources attempt (exit=1; quota=0):
    - `gcloud alpha compute tpus queued-resources create minionerec-sid-sft-v6e-8-flex-260124121715 --zone=us-central2-b --accelerator-type=v6e-8 --runtime-version=v6e-ubuntu-2404 --node-id=minionerec-sid-sft-v6e-8-flex-260124121715-node --provisioning-model=flex-start --max-run-duration=3600s --async`
    - `gcloud alpha compute tpus queued-resources create minionerec-sid-sft-v6e-8-guaranteed-260124121906 --zone=us-central2-b --accelerator-type=v6e-8 --runtime-version=v6e-ubuntu-2404 --node-id=minionerec-sid-sft-v6e-8-guaranteed-260124121906-node --guaranteed --async`
  - TPU v4-8 (spot) create (exit=0):
    - `TPU_NAME="minionerec-sid-sft-v4-8-260124120348"; ./scripts/create_tpu_vm.sh --type v4-8 --zone us-central2-b --name "$TPU_NAME"`
  - TPU v4-8 on-demand create attempt (exit=1; no capacity):
    - `TPU_NAME="minionerec-sid-sft-v4-8-od-260124124033"; ./scripts/create_tpu_vm.sh --type v4-8 --zone us-central2-b --name "$TPU_NAME" --on-demand`
  - TPU v4-8 smoke SFT+Eval (exit=0; W&B online):
    - `./scripts/sync_env_to_tpu_vm.sh --name minionerec-sid-sft-v4-8-260124110839 --zone us-central2-b --src .env --dest /root/.env --worker all`
    - `./scripts/bootstrap_miniconda_on_tpu_vm.sh --name minionerec-sid-sft-v4-8-260124110839 --zone us-central2-b --env-name mllm-jax --python 3.12`
    - `./scripts/ssh_tpu_vm_root.sh --name minionerec-sid-sft-v4-8-260124110839 --zone us-central2-b --command '... pip install jax[tpu] ...; pip install -r requirements-tpu.txt ...; git pull ...; git clone workdir/MiniOneRec ...'`
    - `./scripts/ssh_tpu_vm_root.sh --name minionerec-sid-sft-v4-8-260124110839 --zone us-central2-b --env-file /root/.env --command 'cd /root/MLLM-JAX; ./scripts/run_sid_sft.sh --config plugins/sft/configs/sid_sft_jax_smoke_qwen25_1p5b_instruct_industrial_tpu.yaml --run-mode train_eval'`
    - W&B run: `https://wandb.ai/johntitordemon2036/minionerec-sid-sft/runs/jlijs7du`
  - TPU metrics cross-check (exit=0; matches):
    - `./scripts/ssh_tpu_vm_root.sh --name minionerec-sid-sft-v4-8-260124110839 --zone us-central2-b --command 'cd /root/MLLM-JAX; python workdir/MiniOneRec/calc.py --path runs/sid_sft_jax_smoke_qwen25_1p5b_instruct_industrial_tpu/eval_predictions.json --item_path workdir/MiniOneRec/data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt'`
  - TPU v4-8 JAX device count check (exit=0):
    - `./scripts/ssh_tpu_vm_root.sh --name minionerec-sid-sft-v4-8-260124110839 --zone us-central2-b --command 'source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; python - <<\"PY\"\nimport jax\nprint(\"device_count\", jax.device_count())\nprint(\"local_device_count\", jax.local_device_count())\nPY'`
    - Output: `device_count 4` (v4-8 uses 4 JAX devices)
  - TPU v4-8 smoke rerun with effective_bs=1024 (exit=0; W&B online; repo `6e5df4b`):
    - `./scripts/ssh_tpu_vm_root.sh --name minionerec-sid-sft-v4-8-260124110839 --zone us-central2-b --command 'cd /root/MLLM-JAX; git fetch --all; git checkout main; git pull; git rev-parse --short HEAD'`
    - `./scripts/ssh_tpu_vm_root.sh --name minionerec-sid-sft-v4-8-260124110839 --zone us-central2-b --env-file /root/.env --command 'cd /root/MLLM-JAX; ./scripts/run_sid_sft.sh --config plugins/sft/configs/sid_sft_jax_smoke_qwen25_1p5b_instruct_industrial_tpu.yaml --run-mode train_eval'`
    - W&B run: `https://wandb.ai/johntitordemon2036/minionerec-sid-sft/runs/tkgflo1t`
    - Key log: `[sft] step=1/1 ... effective_bs=1024`
    - `calc.py` cross-check (exit=0; sample_test=8):
      - `./scripts/ssh_tpu_vm_root.sh --name minionerec-sid-sft-v4-8-260124110839 --zone us-central2-b --command 'cd /root/MLLM-JAX; python workdir/MiniOneRec/calc.py --path runs/sid_sft_jax_smoke_qwen25_1p5b_instruct_industrial_tpu/eval_predictions.json --item_path workdir/MiniOneRec/data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt'`
- Key artifacts:
  - `runs/sid_sft_jax_smoke_qwen25_1p5b_instruct_industrial_tpu/run_summary.json`
  - `runs/sid_sft_jax_smoke_qwen25_1p5b_instruct_industrial_tpu/eval_predictions.json`
  - `runs/sid_sft_jax_smoke_qwen25_1p5b_instruct_industrial_tpu/eval_predictions.metrics.json`
  - `runs/sid_sft_jax_smoke_qwen25_1p5b_instruct_industrial_tpu/sft_state_last.msgpack`
- Notes:
  - Upstream SFT defaults: `workdir/MiniOneRec/sft.sh` uses `--batch_size 1024 --micro_batch_size 16` (effective batch 1024 via grad accumulation); HF Hub tag shows base model `Qwen/Qwen2.5-1.5B-Instruct`.
  - JAX/TPU SFT now uses FSDP mesh (`jax.mesh_shape: auto`) + vocab padding so embedding sharding works when `len(tokenizer)` is not divisible by `fsdp*tp` (observed: `152225 -> 152228` on v4-8).
  - v4-8 effective batch alignment: because v4-8 reports `device_count=4`, use `micro=8` + `gradient_accumulation_steps=32` to match upstream effective batch 1024 (validated via TPU log + W&B `train/effective_batch_size=1024`).
  - TPU smoke eval speed: `plugins/sft/configs/sid_sft_jax_smoke_qwen25_1p5b_instruct_industrial_tpu.yaml` uses `sample_test=8` to reduce per-prompt-length JIT compiles.
  - 1.5B base model configs also available: `plugins/sft/configs/sid_sft_jax_smoke_qwen25_1p5b_base_industrial_tpu.yaml`, `plugins/sft/configs/sid_sft_jax_qwen25_1p5b_base_industrial_tpu.yaml`.
  - TPU spot may be preempted; rerun on a fresh `v4-8` if needed.
