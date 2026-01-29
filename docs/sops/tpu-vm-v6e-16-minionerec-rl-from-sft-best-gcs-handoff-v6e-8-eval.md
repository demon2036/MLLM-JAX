# SOP: v6e-16 MiniOneRec RL (from SFT best) + GCS handoff + v6e-8 eval

- **Title**: SOP: v6e-16 multi-host MiniOneRec RL (GRPO-style) from SFT best + GCS handoff + v6e-8 constrained-decoding eval
- **Prereqs**: TPU VMs reachable via `gcloud ... tpu-vm ssh`; repo synced via Git; `/root/.env` contains `WANDB_API_KEY`; `workdir/MiniOneRec` exists on TPUs; `gsutil` access to the artifacts bucket
- **Environment (verified)**:
  - Train TPU: `v6e-16` multi-host (`us-east5-b`)
  - Eval TPU: `v6e-8` single-host (`europe-west4-a`)
  - Repo commit used in runs: `a06e874`

## Steps (commands actually run)

### 0) Names

```bash
TRAIN_TPU_NAME="mllm-jax-v6e-16-sft-match-official-260129053447"
TRAIN_TPU_ZONE="us-east5-b"
EVAL_TPU_NAME="mllm-jax-v6e-8-sft-eval-match-official-260129055649"
EVAL_TPU_ZONE="europe-west4-a"
GCS_PREFIX="gs://mllm-jax-artifacts-482610-s5-260129-fb89/rl_from_sft_best"
```

### 1) Start RL train (multi-host) via nohup (run on every worker)

Industrial:

```bash
for w in 0 1 2 3; do
  scripts/ssh_tpu_vm_root.sh --name "$TRAIN_TPU_NAME" --zone "$TRAIN_TPU_ZONE" --worker "$w" --env-file /root/.env --command \
    'cd /root/MLLM-JAX && bash scripts/tpu_vm_start_minionerec_rl_from_config_multihost_nohup.sh --config projects/minionerec/rl/configs/minionerec_rl_jax_qwen25_1p5b_base_industrial_v6e16_from_sft_best.yaml'
done
```

Office:

```bash
for w in 0 1 2 3; do
  scripts/ssh_tpu_vm_root.sh --name "$TRAIN_TPU_NAME" --zone "$TRAIN_TPU_ZONE" --worker "$w" --env-file /root/.env --command \
    'cd /root/MLLM-JAX && bash scripts/tpu_vm_start_minionerec_rl_from_config_multihost_nohup.sh --config projects/minionerec/rl/configs/minionerec_rl_jax_qwen25_1p5b_base_office_v6e16_from_sft_best.yaml'
done
```

### 2) Wait for exit=0 on all workers

```bash
for w in 0 1 2 3; do
  scripts/ssh_tpu_vm_root.sh --name "$TRAIN_TPU_NAME" --zone "$TRAIN_TPU_ZONE" --worker "$w" --command \
    'cd /root/MLLM-JAX && ls -l logs/nohup_minionerec_rl_*_latest.exit && cat logs/nohup_minionerec_rl_*_latest.exit'
done
```

Note: process-0 is the one that prints `backend=tpu process=0/4` in its `*_latest.log`; that host will have the saved checkpoint.

### 3) Upload `rl_last` checkpoint to GCS (run on the process-0 host)

Industrial:

```bash
scripts/ssh_tpu_vm_root.sh --name "$TRAIN_TPU_NAME" --zone "$TRAIN_TPU_ZONE" --worker 1 --command \
  'cd /root/MLLM-JAX && gsutil -m cp -n runs/minionerec_rl_jax_qwen25_1p5b_base_industrial_v6e16_from_sft_best/sft_state_rl_last.msgpack runs/minionerec_rl_jax_qwen25_1p5b_base_industrial_v6e16_from_sft_best/run_summary.json runs/minionerec_rl_jax_qwen25_1p5b_base_industrial_v6e16_from_sft_best/tokenizer.json runs/minionerec_rl_jax_qwen25_1p5b_base_industrial_v6e16_from_sft_best/tokenizer_config.json runs/minionerec_rl_jax_qwen25_1p5b_base_industrial_v6e16_from_sft_best/chat_template.jinja '"$GCS_PREFIX"'/industrial/'
```

Office:

```bash
scripts/ssh_tpu_vm_root.sh --name "$TRAIN_TPU_NAME" --zone "$TRAIN_TPU_ZONE" --worker 1 --command \
  'cd /root/MLLM-JAX && gsutil -m cp -n runs/minionerec_rl_jax_qwen25_1p5b_base_office_v6e16_from_sft_best/sft_state_rl_last.msgpack runs/minionerec_rl_jax_qwen25_1p5b_base_office_v6e16_from_sft_best/run_summary.json runs/minionerec_rl_jax_qwen25_1p5b_base_office_v6e16_from_sft_best/tokenizer.json runs/minionerec_rl_jax_qwen25_1p5b_base_office_v6e16_from_sft_best/tokenizer_config.json runs/minionerec_rl_jax_qwen25_1p5b_base_office_v6e16_from_sft_best/chat_template.jinja '"$GCS_PREFIX"'/office/'
```

### 4) Download checkpoint onto v6e-8 eval TPU

Industrial:

```bash
scripts/ssh_tpu_vm_root.sh --name "$EVAL_TPU_NAME" --zone "$EVAL_TPU_ZONE" --worker 0 --command \
  'cd /root/MLLM-JAX && mkdir -p workdir/checkpoints/minionerec_rl_from_sft_best/industrial && gsutil -m cp -n '"$GCS_PREFIX"'/industrial/sft_state_rl_last.msgpack workdir/checkpoints/minionerec_rl_from_sft_best/industrial/sft_state_rl_last.msgpack'
```

Office:

```bash
scripts/ssh_tpu_vm_root.sh --name "$EVAL_TPU_NAME" --zone "$EVAL_TPU_ZONE" --worker 0 --command \
  'cd /root/MLLM-JAX && mkdir -p workdir/checkpoints/minionerec_rl_from_sft_best/office && gsutil -m cp -n '"$GCS_PREFIX"'/office/sft_state_rl_last.msgpack workdir/checkpoints/minionerec_rl_from_sft_best/office/sft_state_rl_last.msgpack'
```

### 5) Run constrained-decoding eval (v6e-8) via nohup

Industrial:

```bash
scripts/ssh_tpu_vm_root.sh --name "$EVAL_TPU_NAME" --zone "$EVAL_TPU_ZONE" --worker 0 --env-file /root/.env --command \
  'cd /root/MLLM-JAX && bash scripts/tpu_vm_start_minionerec_rl_from_config_nohup.sh --config projects/minionerec/rl/configs/minionerec_rl_jax_qwen25_1p5b_base_industrial_v6e8_eval_from_rl_last.yaml --run-mode eval'
```

Office:

```bash
scripts/ssh_tpu_vm_root.sh --name "$EVAL_TPU_NAME" --zone "$EVAL_TPU_ZONE" --worker 0 --env-file /root/.env --command \
  'cd /root/MLLM-JAX && bash scripts/tpu_vm_start_minionerec_rl_from_config_nohup.sh --config projects/minionerec/rl/configs/minionerec_rl_jax_qwen25_1p5b_base_office_v6e8_eval_from_rl_last.yaml --run-mode eval'
```

Check exit code:

```bash
scripts/ssh_tpu_vm_root.sh --name "$EVAL_TPU_NAME" --zone "$EVAL_TPU_ZONE" --worker 0 --command \
  'cd /root/MLLM-JAX && ls -l logs/nohup_minionerec_rl_*_latest.exit && cat logs/nohup_minionerec_rl_*_latest.exit'
```

### 6) Cross-check metrics with upstream `calc.py`

Industrial:

```bash
scripts/ssh_tpu_vm_root.sh --name "$EVAL_TPU_NAME" --zone "$EVAL_TPU_ZONE" --worker 0 --command \
  'set -euo pipefail; cd /root/MLLM-JAX; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; python workdir/MiniOneRec/calc.py --path runs/minionerec_rl_jax_qwen25_1p5b_base_industrial_v6e8_eval_from_rl_last/eval_predictions.json --item_path workdir/MiniOneRec/data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt'
```

Office:

```bash
scripts/ssh_tpu_vm_root.sh --name "$EVAL_TPU_NAME" --zone "$EVAL_TPU_ZONE" --worker 0 --command \
  'set -euo pipefail; cd /root/MLLM-JAX; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; python workdir/MiniOneRec/calc.py --path runs/minionerec_rl_jax_qwen25_1p5b_base_office_v6e8_eval_from_rl_last/eval_predictions.json --item_path workdir/MiniOneRec/data/Amazon/info/Office_Products_5_2016-10-2018-11.txt'
```

## Expected Result

- All workers exit `0` for RL train, and `runs/.../sft_state_rl_last.msgpack` exists on the process-0 host.
- v6e-8 eval exits `0` and produces:
  - `eval_predictions.json`
  - `run_summary.json` with `eval/hr@K`, `eval/ndcg@K`, and `eval/invalid_prediction_count==0`
- `workdir/MiniOneRec/calc.py` metrics match the eval summary (invalid=0).

## Notes / Troubleshooting

- If you can’t find `sft_state_rl_last.msgpack`, locate the process-0 host by grepping logs for `backend=tpu process=0/4`.
- If TPU is stuck “busy”, kill the training process and remove locks:
  - `fuser -k /dev/vfio/* || true; rm -f /tmp/libtpu_lockfile || true`

## References

- Upstream metrics script: `workdir/MiniOneRec/calc.py`
- This repo RL entrypoint: `scripts/run_minionerec_rl.py`

