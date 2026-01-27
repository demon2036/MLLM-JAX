# SOP: TPU v6e-8 SID SFT official-alignment run (W&B online)

- **Title**: SOP: TPU v6e-8 SID SFT official-alignment run (W&B online)
- **Prereqs**: gcloud auth; TPU quota; W&B API key in `.env`; repo pushed to GitHub
- **Environment (verified)**: `<to be filled>`

## Goal

- Create a fresh v6e-8 TPU VM and run official-aligned SID SFT evals.
- Log runs to W&B online and cross-check with upstream `calc.py`.

## Steps (commands actually run)

- `gcloud alpha compute tpus tpu-vm create minionerec-sid-sft-v6e-8-official-2601280345 --zone europe-west4-a --accelerator-type v6e-8 --version v6e-ubuntu-2404 --spot --metadata "WANDB_MODE=online,WANDB_PROJECT=minionerec-sid-sft,REPO_URL=https://github.com/demon2036/MLLM-JAX.git,REPO_REF=john/sid3-fixed-prefill-single-compile" --metadata-from-file "startup-script=scripts/tpu_vm_start_sid_sft_official_eval_startup.sh,WANDB_API_KEY=workdir/wandb_api_key.txt"` (exit `0`)
- `gcloud alpha compute tpus tpu-vm delete minionerec-sid-sft-v6e-8-official-2601280345 --zone europe-west4-a --quiet` (exit `0`)
- `gcloud alpha compute tpus tpu-vm create minionerec-sid-sft-v6e-8-official-2601280415 --zone europe-west4-a --accelerator-type v6e-8 --version v6e-ubuntu-2404 --spot --metadata "WANDB_MODE=online,WANDB_PROJECT=minionerec-sid-sft,REPO_URL=https://github.com/demon2036/MLLM-JAX.git,REPO_REF=john/sid3-fixed-prefill-single-compile" --metadata-from-file "startup-script=scripts/tpu_vm_start_sid_sft_official_eval_bootstrap.sh,WANDB_API_KEY=workdir/wandb_api_key.txt"` (exit `1`, capacity)
- `gcloud alpha compute tpus tpu-vm create minionerec-sid-sft-v6e-8-official-2601280430 --zone us-east5-b --accelerator-type v6e-8 --version v6e-ubuntu-2404 --spot --metadata "WANDB_MODE=online,WANDB_PROJECT=minionerec-sid-sft,REPO_URL=https://github.com/demon2036/MLLM-JAX.git,REPO_REF=john/sid3-fixed-prefill-single-compile" --metadata-from-file "startup-script=scripts/tpu_vm_start_sid_sft_official_eval_bootstrap.sh,WANDB_API_KEY=workdir/wandb_api_key.txt"` (exit `1`, capacity)
- `gcloud alpha compute tpus tpu-vm create minionerec-sid-sft-v6e-8-official-2601280500 --zone us-central2-b --accelerator-type v6e-8 --version v6e-ubuntu-2404 --spot --metadata "WANDB_MODE=online,WANDB_PROJECT=minionerec-sid-sft,REPO_URL=https://github.com/demon2036/MLLM-JAX.git,REPO_REF=john/sid3-fixed-prefill-single-compile" --metadata-from-file "startup-script=scripts/tpu_vm_start_sid_sft_official_eval_bootstrap.sh,WANDB_API_KEY=workdir/wandb_api_key.txt"` (exit `1`, v6e spot quota 0)
- `gcloud alpha compute tpus tpu-vm create minionerec-sid-sft-v6e-8-official-2601280510 --zone us-east1-d --accelerator-type v6e-8 --version v6e-ubuntu-2404 --spot --metadata "WANDB_MODE=online,WANDB_PROJECT=minionerec-sid-sft,REPO_URL=https://github.com/demon2036/MLLM-JAX.git,REPO_REF=john/sid3-fixed-prefill-single-compile" --metadata-from-file "startup-script=scripts/tpu_vm_start_sid_sft_official_eval_bootstrap.sh,WANDB_API_KEY=workdir/wandb_api_key.txt"` (exit `0`)
- `gcloud alpha compute tpus tpu-vm stop minionerec-sid-sft-v6e-8-official-2601280510 --zone us-east1-d` (exit `0`)
- `gcloud alpha compute tpus tpu-vm start minionerec-sid-sft-v6e-8-official-2601280510 --zone us-east1-d` (exit `1`, capacity)

## Expected Result

- TPU run exits `0`, writes `eval_predictions.json` + metrics, and logs W&B URLs.
- `calc.py` metrics match `eval_predictions.metrics.json` for each dataset.

## Troubleshooting

- If W&B auth fails, strip CRLF in `/root/.env` (`sed -i "s/\\r$//"`).
- Startup-script path did not produce W&B heartbeats or guest-attribute status; consider SSH launch if allowed.

## References

- `docs/sops/minionerec-sid-sft-jax-tpu.md`
- `memory/20260127_tpu-v6e8-sft-official-align/README.md`
