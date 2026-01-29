# SOP: MiniOneRec SID SFT (JAX) — Muon optimizer on TPU v6e-8

- **Title**: SOP: Run MiniOneRec SID SFT (JAX) with Muon optimizer on TPU `v6e-8`, compare vs AdamW, and cross-check HR@K/NDCG@K with upstream `calc.py`.
- **Prereqs**: TPU VM reachable via `gcloud ... tpu-vm ssh`; repo synced via Git; `/root/.env` on TPU containing `WANDB_API_KEY` (W&B `mode=online`); `workdir/MiniOneRec` + dataset prepared on TPU.
- **Environment (verified)**:
  - TPU: `v6e-8`, zone `europe-west4-a`, VM `sft-muon-v6e-8-260129145924`
  - Conda env: `mllm-jax` (Python `3.12.12`, JAX `0.9.0`, jaxlib `0.9.0`, Optax `0.2.6`)

## Steps (commands actually run)

### 1) Sync repo to the Muon branch

- `scripts/ssh_tpu_vm_root.sh --name sft-muon-v6e-8-260129145924 --zone europe-west4-a --command 'set -euo pipefail; cd /root/MLLM-JAX; git fetch origin; git checkout john/sft-muon-20260129; git pull --ff-only origin john/sft-muon-20260129; git rev-parse --short HEAD'`

### 2) Run Muon SFT train (best config)

- Start Muon SFT training (nohup, W&B online):
  - `scripts/ssh_tpu_vm_root.sh --name sft-muon-v6e-8-260129145924 --zone europe-west4-a --env-file /root/.env --command 'set -euo pipefail; cd /root/MLLM-JAX; bash scripts/tpu_vm_start_sid_sft_from_config_nohup.sh --config projects/minionerec/sft/configs/sid_sft_jax_qwen25_1p5b_base_industrial_v6e8_e3_muon_lr3e3_aux3e3_train.yaml --run-mode train'`

### 3) Run constrained-decoding eval from the last checkpoint (dp=8, bs=8)

- Start eval (nohup, W&B online):
  - `scripts/ssh_tpu_vm_root.sh --name sft-muon-v6e-8-260129145924 --zone europe-west4-a --env-file /root/.env --command 'set -euo pipefail; cd /root/MLLM-JAX; bash scripts/tpu_vm_start_sid_sft_from_config_nohup.sh --config projects/minionerec/sft/configs/sid_sft_jax_qwen25_1p5b_base_industrial_v6e8_e3_muon_lr3e3_aux3e3_last_eval_dp8_bs8.yaml --run-mode eval'`

### 4) Cross-check metrics with upstream `calc.py`

- Muon eval output:
  - `scripts/ssh_tpu_vm_root.sh --name sft-muon-v6e-8-260129145924 --zone europe-west4-a --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; python workdir/MiniOneRec/calc.py --path runs/sid_sft_jax_qwen25_1p5b_base_industrial_v6e8_e3_muon_lr3e3_aux3e3_last_eval_dp8_bs8/eval_predictions.json --item_path workdir/MiniOneRec/data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt'`
- AdamW baseline eval output (if present):
  - `scripts/ssh_tpu_vm_root.sh --name sft-muon-v6e-8-260129145924 --zone europe-west4-a --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; python workdir/MiniOneRec/calc.py --path runs/sid_sft_jax_qwen25_1p5b_base_industrial_v6e8_e3_adamw_last_eval_dp8_bs8/eval_predictions.json --item_path workdir/MiniOneRec/data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt'`

## Expected Result

- Both train and eval exit `0` (nohup writes exit code files under `logs/nohup_sid_sft_*_latest.exit`).
- Muon eval produces `eval_predictions.json` and `eval_predictions.metrics.json` under:
  - `runs/sid_sft_jax_qwen25_1p5b_base_industrial_v6e8_e3_muon_lr3e3_aux3e3_last_eval_dp8_bs8/`
- `workdir/MiniOneRec/calc.py` matches `eval_predictions.metrics.json` (invalid=0).
- Best Muon config vs AdamW baseline (Industrial, v6e-8, dp8/bs8):
  - AdamW: HR@10 `0.1336863`, NDCG@10 `0.09041011`
  - Muon: HR@10 `0.1336863`, NDCG@10 `0.09634818`

## Notes

- `bitsandbytes` is not needed on TPU; this repo already provides a stub at `plugins/stubs/bitsandbytes/` to satisfy any unconditional imports.

