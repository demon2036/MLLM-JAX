# SOP: TPU VM v6e-16 multi-host MiniOneRec SID SFT (base) + GCS checkpoint handoff + TPU VM v6e-8 eval (W&B online)

- **Title**: SOP: Run MiniOneRec SID **base** SFT on `v6e-16` (multi-host, 4 workers), upload the resulting checkpoint(s) to **GCS** (no SCP), then run constrained-decoding eval on a separate `v6e-8` TPU VM (W&B `mode=online`) and cross-check HR/NDCG with upstream `workdir/MiniOneRec/calc.py`.
- **Prereqs**: `gcloud` authenticated; TPU VMs exist and are reachable; outbound internet from TPU VMs (HF + W&B); TPU has `/root/.env` with `WANDB_API_KEY`; GCS bucket writable from TPU VMs.
- **Environment (verified)**:
  - Train TPU (kept running): `mllm-jax-v6e-16-sft-match-official-260129053447` (`v6e-16`, 4 workers), zone `us-east5-b`
  - Eval TPU (kept running): `mllm-jax-v6e-8-sft-eval-match-official-260129055649` (`v6e-8`, single-host), zone `europe-west4-a`
  - Repo: `https://github.com/demon2036/MLLM-JAX.git`, branch `test`
    - Train runs: commit `03455fc`
    - Added eval-from-last Industrial config: commit `3be8272`
  - GCS handoff prefix: `gs://mllm-jax-artifacts-482610-s5-260129-fb89/sft_match_official/`

## Steps (commands actually run)

### A) v6e-16 (train): verify multi-host exit=0

- Check all workers exited `0` (example: Office train):
  - `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v6e-16-sft-match-official-260129053447 --zone us-east5-b --worker all --command 'set -euo pipefail; host=$(hostname); f=\"/root/MLLM-JAX/logs/nohup_sid_sft_sid_sft_jax_qwen25_1p5b_base_office_v6e16_full_best_rightpad_mu_fp32_train_${host}_20260129_095637.exit\"; echo \"${host}:\"; cat \"$f\"'`

### B) v6e-16 (train): upload checkpoint(s) to GCS

- Upload Office best checkpoint:
  - `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v6e-16-sft-match-official-260129053447 --zone us-east5-b --worker 1 --command 'set -euo pipefail; dst=gs://mllm-jax-artifacts-482610-s5-260129-fb89/sft_match_official/office_rightpad_mu_fp32/; gsutil -m cp /root/MLLM-JAX/runs/sid_sft_jax_qwen25_1p5b_base_office_v6e16_full_best_rightpad_mu_fp32/sft_state_best.msgpack \"$dst\"; gsutil ls \"$dst\"'`

- Upload Industrial last checkpoint:
  - `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v6e-16-sft-match-official-260129053447 --zone us-east5-b --worker 1 --command 'set -euo pipefail; dst=gs://mllm-jax-artifacts-482610-s5-260129-fb89/sft_match_official/industrial_rightpad_mu_fp32/; gsutil -m cp /root/MLLM-JAX/runs/sid_sft_jax_qwen25_1p5b_base_industrial_v6e16_full_best_rightpad_mu_fp32/sft_state_last.msgpack \"$dst\"; gsutil ls \"$dst\"'`

### C) v6e-8 (eval): download checkpoint(s) from GCS to the expected `runs/...` paths

- Office best → local `runs/.../sft_state_best.msgpack`:
  - `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v6e-8-sft-eval-match-official-260129055649 --zone europe-west4-a --worker 0 --command 'set -euo pipefail; dst_dir=/root/MLLM-JAX/runs/sid_sft_jax_qwen25_1p5b_base_office_v6e16_full_best_rightpad_mu_fp32; mkdir -p \"$dst_dir\"; gsutil cp gs://mllm-jax-artifacts-482610-s5-260129-fb89/sft_match_official/office_rightpad_mu_fp32/sft_state_best.msgpack \"$dst_dir/sft_state_best.msgpack\"; ls -lh \"$dst_dir/sft_state_best.msgpack\"'`

- Industrial last → local `runs/.../sft_state_last.msgpack`:
  - `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v6e-8-sft-eval-match-official-260129055649 --zone europe-west4-a --worker 0 --command 'set -euo pipefail; dst_dir=/root/MLLM-JAX/runs/sid_sft_jax_qwen25_1p5b_base_industrial_v6e16_full_best_rightpad_mu_fp32; mkdir -p \"$dst_dir\"; gsutil cp gs://mllm-jax-artifacts-482610-s5-260129-fb89/sft_match_official/industrial_rightpad_mu_fp32/sft_state_last.msgpack \"$dst_dir/sft_state_last.msgpack\"; ls -lh \"$dst_dir/sft_state_last.msgpack\"'`

### D) v6e-8 (eval): git pull to pick up new eval config(s)

- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v6e-8-sft-eval-match-official-260129055649 --zone europe-west4-a --worker 0 --command 'set -euo pipefail; cd /root/MLLM-JAX; git fetch --all; git checkout test; git pull; git rev-parse --short HEAD'`

### E) v6e-8 (eval): run constrained-decoding eval (W&B online)

- Office eval from best:
  - `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v6e-8-sft-eval-match-official-260129055649 --zone europe-west4-a --worker 0 --env-file /root/.env --command 'set -euo pipefail; export PYTHONUNBUFFERED=1; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; ./scripts/run_sid_sft.sh --config projects/minionerec/sft/configs/sid_sft_jax_qwen25_1p5b_base_office_v6e16_eval_from_best_rightpad_mu_fp32.yaml --run-mode eval'`

- Industrial eval from last:
  - `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v6e-8-sft-eval-match-official-260129055649 --zone europe-west4-a --worker 0 --env-file /root/.env --command 'set -euo pipefail; export PYTHONUNBUFFERED=1; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; ./scripts/run_sid_sft.sh --config projects/minionerec/sft/configs/sid_sft_jax_qwen25_1p5b_base_industrial_v6e16_eval_from_last_rightpad_mu_fp32.yaml --run-mode eval'`

### F) v6e-8 (eval): cross-check with upstream `calc.py`

- Office:
  - `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v6e-8-sft-eval-match-official-260129055649 --zone europe-west4-a --worker 0 --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; python workdir/MiniOneRec/calc.py --path runs/sid_sft_jax_qwen25_1p5b_base_office_v6e16_eval_from_best_rightpad_mu_fp32/eval_predictions.json --item_path workdir/MiniOneRec/data/Amazon/info/Office_Products_5_2016-10-2018-11.txt'`

- Industrial:
  - `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v6e-8-sft-eval-match-official-260129055649 --zone europe-west4-a --worker 0 --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; python workdir/MiniOneRec/calc.py --path runs/sid_sft_jax_qwen25_1p5b_base_industrial_v6e16_eval_from_last_rightpad_mu_fp32/eval_predictions.json --item_path workdir/MiniOneRec/data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt'`

## Expected result

- All commands exit `0`.
- Eval writes:
  - `runs/<...>/eval_predictions.json`
  - `runs/<...>/eval_predictions.metrics.json`
- `calc.py` prints the same HR/NDCG as `eval_predictions.metrics.json` (invalid=0).

## Observed result (2026-01-29)

- Office eval from best (n=4866, invalid=0): https://wandb.ai/johntitordemon2036/minionerec-sid-sft/runs/9cl15w2e
  - HR@K: `[0.08816, 0.12145, 0.14159, 0.16461, 0.19441, 0.24764]`
  - NDCG@K: `[0.08816, 0.10747, 0.11577, 0.12314, 0.13069, 0.14117]`
- Industrial eval from last (n=4533, invalid=0): https://wandb.ai/johntitordemon2036/minionerec-sid-sft/runs/2mkirwka
  - HR@K: `[0.06331, 0.08868, 0.09993, 0.11846, 0.14251, 0.18994]`
  - NDCG@K: `[0.06331, 0.07828, 0.08295, 0.08891, 0.09500, 0.10445]`

## References

- Task log: `memory/20260129_v6e16_sft_match_official_metrics/README.md`
- SFT runner: `projects/minionerec/sft/runner.py`
- Eval runner: `projects/minionerec/sft/jax/evaluator.py`
