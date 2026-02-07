# SOP: Run MiniOneRec SID SFT + eval on TPU (JAX backend)

- **Title**: SOP: Run MiniOneRec SID SFT + constrained-decoding HR@K/NDCG@K eval on TPU via `projects/sid_sft/` (JAX)
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
  - `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-central2-b --project "$(gcloud config get-value project)" --env-file /root/.env --command 'set -euo pipefail; export PYTHONUNBUFFERED=1; export HF_HUB_ENABLE_HF_TRANSFER=1; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; ./scripts/run_sid_sft.sh --config projects/sid_sft/configs/sid_sft_jax_smoke_qwen25_1p5b_instruct_industrial_tpu.yaml --run-mode train_eval'`
  - W&B run (online): `https://wandb.ai/johntitordemon2036/minionerec-sid-sft/runs/tkgflo1t`

- Cross-check HR/NDCG with upstream `calc.py` (same predictions JSON):
  - `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-central2-b --project "$(gcloud config get-value project)" --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; python workdir/MiniOneRec/calc.py --path runs/sid_sft_jax_smoke_qwen25_1p5b_instruct_industrial_tpu/eval_predictions.json --item_path workdir/MiniOneRec/data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt'`

## Extra: Eval TEST split from a saved SFT checkpoint (v4-8, beam=20)

- Config:
  - `projects/sid_sft/configs/eval/v4-8/sid_sft_jax_eval_test_beam20_from_rightpad_best_20260201.yaml`
- Command:
  - `scripts/ssh_tpu_vm_root.sh --name plugins-refactor-sid-sft-muon-260131052355 --zone us-central2-b --project civil-rarity-482610-s5 --env-file /root/.env --command 'bash -lc "set -euo pipefail; export PYTHONUNBUFFERED=1; export HF_HUB_ENABLE_HF_TRANSFER=1; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; rm -rf runs/sid_sft_jax_eval_test_beam20_from_rightpad_best_20260201; mkdir -p runs/sid_sft_jax_eval_test_beam20_from_rightpad_best_20260201; bash scripts/run_sid_sft.sh --config projects/sid_sft/configs/eval/v4-8/sid_sft_jax_eval_test_beam20_from_rightpad_best_20260201.yaml --run-mode eval 2>&1 | tee runs/sid_sft_jax_eval_test_beam20_from_rightpad_best_20260201/tpu_eval.log"'`
- Output dir:
  - `runs/sid_sft_jax_eval_test_beam20_from_rightpad_best_20260201/`
- W&B run (online):
  - `johntitordemon2036/minionerec-sid-sft/runs/pwdgs5sf`
- Eval (test split, beams=20, samples=4533, invalid=0):
  - HR@3=`0.10457`, NDCG@3=`0.09172`
  - HR@5=`0.11891`, NDCG@5=`0.09772`
  - HR@10=`0.14648`, NDCG@10=`0.10658`

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
  - JAX eval buckets by `prompt_len`; many unique prompt lengths can trigger many JIT compiles. Reduce `data.sample_test` in the smoke config (already set to `8` in `projects/sid_sft/configs/sid_sft_jax_smoke_qwen25_1p5b_instruct_industrial_tpu.yaml`).
- `ValueError ... global size ... should be divisible by ...` when placing params:
  - Ensure you are on a recent `minionerec` that prints `[sft] pad_vocab_size ...` (this repo pads vocab to be divisible by `fsdp*tp` and resizes embedding/lm_head).
- Constrained decoding not working (CC > 0 in `calc.py`):
  - Switch to the base model config to avoid Instruct dependency issues: `projects/sid_sft/configs/sid_sft_jax_smoke_qwen25_1p5b_base_industrial_tpu.yaml`
- v6e-8 queued-resources (flex-start) quota is 0 in `us-central2-b` (example failures):
  - `gcloud alpha compute tpus queued-resources create minionerec-sid-sft-v6e-8-flex-260124121715 --zone=us-central2-b --accelerator-type=v6e-8 --runtime-version=v6e-ubuntu-2404 --node-id=minionerec-sid-sft-v6e-8-flex-260124121715-node --provisioning-model=flex-start --max-run-duration=3600s --async`
  - `gcloud alpha compute tpus queued-resources create minionerec-sid-sft-v6e-8-guaranteed-260124121906 --zone=us-central2-b --accelerator-type=v6e-8 --runtime-version=v6e-ubuntu-2404 --node-id=minionerec-sid-sft-v6e-8-guaranteed-260124121906-node --guaranteed --async`
- v6e-8 eval `RESOURCE_EXHAUSTED` (constrained beam search):
  - Reduce `eval.batch_size` (start with `1`) and reduce `jax.max_cache_length` (Industrial test prompts fit within `256`, so `512` is safe).
- Missing `workdir/MiniOneRec` data:
  - Re-run the `git clone https://github.com/AkaliKong/MiniOneRec workdir/MiniOneRec` step
- W&B `API key cannot start or end with whitespace` (common when syncing `.env` from Windows with CRLF / trailing spaces):
  - Fix on TPU: `sed -i "s/\\r$//" /root/.env; sed -i "s/[[:space:]]*$//" /root/.env`
- Windows `gcloud` (PuTTY/plink) host-key prompt blocks automation:
  - Use `--ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:<HOSTKEY>` (fingerprint printed by prompt) on `gcloud ... tpu-vm ssh`, and `--scp-flag=...` on `gcloud ... tpu-vm scp`.

## Extra: Compare official `evaluate.py` (torch) vs JAX eval (Industrial `first64.csv`, beam=50)

- What runs where:
  - TPU VM **CPU**: `workdir/MiniOneRec/evaluate.py` (torch generate) writes `workdir/align/*.json`.
  - TPU **devices**: JAX eval via `./scripts/run_sid_sft.sh ... --run-mode eval` writes `runs/*/eval_predictions.json`.
- W&B requirement (for JAX runs):
  - Config must set `wandb.mode=online`.
  - If running via `scripts/ssh_tpu_vm_root.sh`, pass `--env-file /root/.env` so `WANDB_API_KEY` is available (do not commit secrets).

- 1) Official torch eval (CPU) for Industrial first64, beam=50:
  - Command:
    ```bash
    cd /root/MLLM-JAX && bash -lc "set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; export PYTHONUNBUFFERED=1; time python workdir/MiniOneRec/evaluate.py --base_model workdir/hf_ckpts/kkknight_MiniOneRec/Industrial_ckpt --train_file workdir/MiniOneRec/data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv --info_file workdir/MiniOneRec/data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt --category Industrial_and_Scientific --test_data_path workdir/MiniOneRec/data/Amazon/test_subsets/Industrial_and_Scientific_5_2016-10-2018-11.first64.csv --result_json_data workdir/align/torch_eval_industrial_first64_beam50.json --batch_size 4 --K 0 --seed 42 --length_penalty 0.0 --max_new_tokens 32 --num_beams 50 2>&1 | tee workdir/align/torch_eval_industrial_first64_beam50.log"
    ```
  - Result (verified):
    - Exit: `0`
    - `real 52m40.899s`
    - Output: `workdir/align/torch_eval_industrial_first64_beam50.json` (64 rows)

- 2) Compare torch vs baseline JAX bf16 bucket output (existing file):
  - Command:
    ```bash
    cd /root/MLLM-JAX && source /root/miniconda3/etc/profile.d/conda.sh && conda activate mllm-jax && python scripts/compare_minionerec_official_vs_jax_eval.py --torch-json workdir/align/torch_eval_industrial_first64_beam50.json --jax-json runs/sid_sft_jax_eval_official_minionerec_industrial_ckpt_subset64/eval_predictions.json --info-file workdir/MiniOneRec/data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt --out-json workdir/align/compare_torch_vs_jax_first64.report.json
    ```
  - Summary:
    ```
    top1: 60/64 (0.9375)
    overlap ... jaccard_mean=0.9452
    hr@50: torch=0.234375 jax=0.218750
    ndcg@50: torch=0.158845 jax=0.155667
    ```

- 3) JAX eval (bf16) with fixed prefill on TPU (config: `projects/sid_sft/configs/sid_sft_jax_eval_official_minionerec_industrial_ckpt_subset64_fixedprefill.yaml`):
  - Command:
    ```bash
    cd /root/MLLM-JAX && source /root/miniconda3/etc/profile.d/conda.sh && conda activate mllm-jax && ./scripts/run_sid_sft.sh --config projects/sid_sft/configs/sid_sft_jax_eval_official_minionerec_industrial_ckpt_subset64_fixedprefill.yaml --run-mode eval 2>&1 | tee workdir/align/jax_eval_industrial_first64_beam50_fixedprefill.tpu.log
    ```
  - W&B run (online): `https://wandb.ai/johntitordemon2036/minionerec-sid-sft/runs/xj0x1y9s`

- 4) Compare torch vs JAX fixed-prefill output:
  - Command:
    ```bash
    cd /root/MLLM-JAX && python scripts/compare_minionerec_official_vs_jax_eval.py --torch-json workdir/align/torch_eval_industrial_first64_beam50.json --jax-json runs/sid_sft_jax_eval_official_minionerec_industrial_ckpt_subset64_fixedprefill/eval_predictions.json --info-file workdir/MiniOneRec/data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt --out-json workdir/align/compare_torch_vs_jax_industrial_first64_fixedprefill.report.json
    ```
  - Summary:
    ```
    top1: 63/64 (0.9844)
    hr@50: torch=0.234375 jax=0.218750
    ```

- 5) JAX eval (float32 params/compute) on TPU (config: `projects/sid_sft/configs/sid_sft_jax_eval_official_minionerec_industrial_ckpt_subset64_f32.yaml`):
  - Command:
    ```bash
    cd /root/MLLM-JAX && source /root/miniconda3/etc/profile.d/conda.sh && conda activate mllm-jax && ./scripts/run_sid_sft.sh --config projects/sid_sft/configs/sid_sft_jax_eval_official_minionerec_industrial_ckpt_subset64_f32.yaml --run-mode eval 2>&1 | tee workdir/align/jax_eval_industrial_first64_beam50_f32.tpu.log
    ```
  - W&B run (online): `https://wandb.ai/johntitordemon2036/minionerec-sid-sft/runs/cy4tj485`

- 6) Compare torch vs JAX float32 output:
  - Command:
    ```bash
    cd /root/MLLM-JAX && python scripts/compare_minionerec_official_vs_jax_eval.py --torch-json workdir/align/torch_eval_industrial_first64_beam50.json --jax-json runs/sid_sft_jax_eval_official_minionerec_industrial_ckpt_subset64_f32/eval_predictions.json --info-file workdir/MiniOneRec/data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt --out-json workdir/align/compare_torch_vs_jax_industrial_first64_f32.report.json
    ```
  - Summary:
    ```
    top1: 63/64 (0.9844)
    overlap ... jaccard_mean=0.9596
    hr@50: torch=0.234375 jax=0.234375
    ndcg@50: torch=0.158845 jax=0.158841
    ```

- Interpretation:
  - bf16 JAX differs slightly at K=50.
  - float32 JAX matches torch HR@50 and nearly matches NDCG@50.
  - Full top-50 ordering still differs (top1 < 64/64, jaccard_mean < 1.0).

## Extra: Eval official MiniOneRec HF checkpoints (v6e-8, full test)

- Download checkpoints on TPU (only once per VM):
  - `scripts/ssh_tpu_vm_root.sh --name minionerec-sid-sft-v6e-8-official-eval-260124163806 --zone us-east5-b --env-file /root/.env --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; export HF_HUB_ENABLE_HF_TRANSFER=1; cd /root/MLLM-JAX; python - <<\"PY\"\nfrom huggingface_hub import snapshot_download\nsnapshot_download(repo_id=\"kkknight/MiniOneRec\", allow_patterns=[\"Industrial_ckpt/*\"], local_dir=\"workdir/hf_ckpts/kkknight_MiniOneRec\")\nsnapshot_download(repo_id=\"kkknight/MiniOneRec\", allow_patterns=[\"Office_ckpt/*\"], local_dir=\"workdir/hf_ckpts/kkknight_MiniOneRec\")\nPY'`

- Ensure upstream `calc.py` deps exist on TPU:
  - `scripts/ssh_tpu_vm_root.sh --name minionerec-sid-sft-v6e-8-official-eval-260124163806 --zone us-east5-b --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; python -m pip install -U fire'`

- Run eval (Industrial):
  - `scripts/ssh_tpu_vm_root.sh --name minionerec-sid-sft-v6e-8-official-eval-260124163806 --zone us-east5-b --env-file /root/.env --command 'set -euo pipefail; export PYTHONUNBUFFERED=1; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; ./scripts/run_sid_sft.sh --config projects/sid_sft/configs/sid_sft_jax_eval_official_minionerec_industrial_ckpt.yaml --run-mode eval'`
  - Cross-check via upstream `calc.py`:
    - `scripts/ssh_tpu_vm_root.sh --name minionerec-sid-sft-v6e-8-official-eval-260124163806 --zone us-east5-b --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; python workdir/MiniOneRec/calc.py --path runs/sid_sft_jax_eval_official_minionerec_industrial_ckpt/eval_predictions.json --item_path workdir/MiniOneRec/data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt'`

- Run eval (Office):
  - `scripts/ssh_tpu_vm_root.sh --name minionerec-sid-sft-v6e-8-official-eval-260124163806 --zone us-east5-b --env-file /root/.env --command 'set -euo pipefail; export PYTHONUNBUFFERED=1; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; ./scripts/run_sid_sft.sh --config projects/sid_sft/configs/sid_sft_jax_eval_official_minionerec_office_ckpt.yaml --run-mode eval'`
  - Cross-check via upstream `calc.py`:
    - `scripts/ssh_tpu_vm_root.sh --name minionerec-sid-sft-v6e-8-official-eval-260124163806 --zone us-east5-b --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; python workdir/MiniOneRec/calc.py --path runs/sid_sft_jax_eval_official_minionerec_office_ckpt/eval_predictions.json --item_path workdir/MiniOneRec/data/Amazon/info/Office_Products_5_2016-10-2018-11.txt'`

- Verified results (2026-01-26, v6e-8 spot, `europe-west4-a`, commit `2eb3786`):
  - Industrial: https://wandb.ai/johntitordemon2036/minionerec-sid-sft/runs/2sictgwy
    - HR@K (K=[1,3,5,10,20,50]): `[0.08537, 0.11339, 0.13280, 0.15861, 0.19171, 0.24465]`
    - NDCG@K: `[0.08537, 0.10129, 0.10933, 0.11749, 0.12584, 0.13631]`
  - Office: https://wandb.ai/johntitordemon2036/minionerec-sid-sft/runs/fqe2nv2p
    - HR@K (K=[1,3,5,10,20,50]): `[0.09412, 0.12474, 0.13872, 0.15865, 0.18475, 0.23695]`
    - NDCG@K: `[0.09412, 0.11212, 0.11801, 0.12442, 0.13096, 0.14118]`
  - `workdir/MiniOneRec/calc.py` matches `eval_predictions.metrics.json` (invalid=0 for both).

## Extra: Measure SFT train step time (v6e-8)

- Note: JAX training requires static batch shapes; training batches are padded to fixed `data.max_len` (commit `2b5208a`) to avoid JIT recompiles from dynamic padding.

- Create TPU (spot):
  - `TPU_NAME="minionerec-sid-sft-step-time-v6e-8-260124174627"; ./scripts/create_tpu_vm.sh --type v6e-8 --zone us-east5-b --name "$TPU_NAME"`

- Bootstrap + install deps + clone repo + clone upstream MiniOneRec:
  - Follow the same steps above (just replace `--zone` and `--type`); ensure the repo branch is `minionerec`.

- Run a 3-step timing job (len=512, effective_bs=1024, eval disabled):
  - `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone us-east5-b --env-file /root/.env --command 'set -euo pipefail; export PYTHONUNBUFFERED=1; export HF_HUB_ENABLE_HF_TRANSFER=1; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; ./scripts/run_sid_sft.sh --config projects/sid_sft/configs/sid_sft_jax_qwen25_1p5b_instruct_industrial_v6e8_step_time.yaml --run-mode train'`
  - W&B run: `https://wandb.ai/johntitordemon2036/minionerec-sid-sft/runs/wcgcvyua`
  - Observed per-step timing (W&B `train/step_time_sec`):
    - step1: `113.19s` (includes JIT compile on this warm VM)
    - step2: `4.05s`
    - step3: `4.04s`

## References

- Upstream metrics script: `workdir/MiniOneRec/calc.py`
- Project entrypoints: `scripts/run_sid_sft.py`, `projects/sid_sft/runner/sid_sft.py`
