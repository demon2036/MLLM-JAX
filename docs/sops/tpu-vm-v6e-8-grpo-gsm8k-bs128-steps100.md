# TPU VM v6e-8 (spot, 1-host) GRPO/GSM8K 100-step Train (bs=128 sequences, global) with W&B

- **Title**: SOP: Run `GRPO + GSM8K` for 100 steps on a `v6e-8` TPU VM (spot) with `batch_size=128` (global sequences/step) and W&B logging
  **Prereqs**: Windows PowerShell + `gcloud` installed/authenticated; TPU API enabled; outbound internet from TPU VM (HF + datasets + wandb); local `.env` contains `WANDB_API_KEY` (gitignored)
  **Environment (verified)**:
  - Local: Windows PowerShell; gcloud `551.0.0`; account `nitokyo8@gmail.com`; project `civil-rarity-482610-s5`
  - TPU VM: `mllm-jax-v6e-8-260120021350` (`v6e-8`, spot), zone `us-east1-d`
  - TPU OS: Ubuntu `24.04.2` (kernel `6.11.0-1015-gcp`)
  - Conda env: `mllm-jax` (Python `3.12.12`)
  - JAX: `0.8.2`, jaxlib `0.8.2`, libtpu `0.0.32`, device_count `8`
  - Repo: `https://github.com/demon2036/MLLM-JAX.git`, branch `mllm-jax-sglang`, commit `0fb993a`
  - W&B run: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/digdipr9`

## Update note (current `main` branch)

- Default config: `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml` (`rollout.batch_size=16`, `rollout.n=8` → `128 sequences/step`).
- Launcher: `bash scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh --config <path>.yaml` (W&B is config-driven via `wandb_mode` in YAML).

## Steps (commands actually used)

### 0) Delete the old v4-8 (in `us-central2-b`)

- `gcloud alpha compute tpus tpu-vm delete mllm-jax-v4-8-260117090531 --project civil-rarity-482610-s5 --zone us-central2-b --quiet`

### 1) Create a v6e-8 spot TPU in a zone with non-zero v6e-preemptible quota

In this project, `us-central2-b` reports `TPUV6E*` quota limit 0; `us-east1-d` has non-zero `tpu-v6e-preemptible` quota.

- Zone supports `v6e-8`:
  - `gcloud compute tpus accelerator-types list --zone us-east1-d --format='value(name)' | Select-String -Pattern 'acceleratorTypes/v6e-8$'`
- Create `v6e-8` spot (PowerShell snippet used):
  - `$ts = (Get-Date).ToUniversalTime().ToString('yyMMddHHmmss'); $name = "mllm-jax-v6e-8-$ts"; Write-Host "TPU_NAME=$name"; gcloud alpha compute tpus tpu-vm create $name --project civil-rarity-482610-s5 --zone us-east1-d --accelerator-type v6e-8 --version v6e-ubuntu-2404 --spot --quiet; gcloud alpha compute tpus tpu-vm describe $name --project civil-rarity-482610-s5 --zone us-east1-d --format='value(state,acceleratorType)'`

### 2) SSH from Windows (avoid PuTTY host-key prompt)

Record the host key fingerprint shown on first connect and pass it explicitly:

- `TPU_NAME=mllm-jax-v6e-8-260120021350; ZONE=us-east1-d; PROJECT=civil-rarity-482610-s5`
- `HOSTKEY=SHA256:x1xe2wDNx76iR4+mk1Cv7zd5/YyDXb+cq6szl8GY2Cs`
- `gcloud alpha compute tpus tpu-vm ssh root@$TPU_NAME --project $PROJECT --zone $ZONE --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=$HOSTKEY --command 'set -euo pipefail; whoami; cat /etc/os-release | sed -n "1,3p"; uname -r'`

### 3) Bootstrap Miniconda + conda env

- `gcloud alpha compute tpus tpu-vm ssh root@$TPU_NAME --project $PROJECT --zone $ZONE --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=$HOSTKEY --command 'set -euo pipefail; if [ ! -d /root/miniconda3 ]; then curl -fsSL -o /root/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; bash /root/miniconda.sh -b -p /root/miniconda3; rm -f /root/miniconda.sh; fi; /root/miniconda3/bin/conda --version'`
- `gcloud alpha compute tpus tpu-vm ssh root@$TPU_NAME --project $PROJECT --zone $ZONE --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=$HOSTKEY --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true; conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true; if ! conda env list | grep -Eq "^mllm-jax[[:space:]]"; then conda create -y -n mllm-jax python=3.12; fi; conda activate mllm-jax; python --version; pip install -U pip'`

### 4) Git-sync the repo on TPU (no scp for code)

- `gcloud alpha compute tpus tpu-vm ssh root@$TPU_NAME --project $PROJECT --zone $ZONE --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=$HOSTKEY --command 'set -euo pipefail; REPO_URL=https://github.com/demon2036/MLLM-JAX.git; REPO_DIR=/root/MLLM-JAX; if [ ! -d "$REPO_DIR/.git" ]; then rm -rf "$REPO_DIR"; git clone "$REPO_URL" "$REPO_DIR"; fi; cd "$REPO_DIR"; git fetch --all --prune; git checkout mllm-jax-sglang; git reset --hard origin/mllm-jax-sglang; git clean -fd; echo HEAD:; git rev-parse --short HEAD; git status -sb'`

### 5) Install runtime deps (JAX + requirements)

- `gcloud alpha compute tpus tpu-vm ssh root@$TPU_NAME --project $PROJECT --zone $ZONE --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=$HOSTKEY --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; python -m pip install -U pip; python -m pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; python -m pip install -U torch --index-url https://download.pytorch.org/whl/cpu; cd /root/MLLM-JAX; python -m pip install -U -r requirements-tpu.txt'`

### 6) Sync local `.env` to the TPU VM (for W&B)

`tpu-vm scp` on Windows also prompts for host key; pass it explicitly.

- `gcloud alpha compute tpus tpu-vm scp .env root@$TPU_NAME:/root/.env --project $PROJECT --zone $ZONE --worker 0 --quiet --scp-flag=-batch --scp-flag=-hostkey --scp-flag=$HOSTKEY`
- `gcloud alpha compute tpus tpu-vm ssh root@$TPU_NAME --project $PROJECT --zone $ZONE --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=$HOSTKEY --command 'set -euo pipefail; chmod 600 /root/.env; test -f /root/.env; ls -la /root/.env'`

### 7) Start a 100-step run (bs=128 sequences) via nohup helper

Config notes:
- Config: `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml`
- Semantics: `rollout.batch_size` is global prompts/step; `rollout.n` is samples/prompt; global sequences/step = `16 * 8 = 128`.
- `train.micro_batch_size_per_device=4` → runner infers a compatible global micro-batch and `train.grad_accum_steps` at runtime.
- W&B is config-driven via `wandb_mode` in YAML (set `online` for verification).

- `gcloud alpha compute tpus tpu-vm ssh root@$TPU_NAME --project $PROJECT --zone $ZONE --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=$HOSTKEY --command 'set -euo pipefail; cd /root/MLLM-JAX; bash scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh --env-name mllm-jax --config plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml'`

### 8) Monitor and verify exit code

- `gcloud alpha compute tpus tpu-vm ssh root@$TPU_NAME --project $PROJECT --zone $ZONE --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=$HOSTKEY --command 'ps -p <PID> -o pid=,etime=,cmd= || echo process_not_running'`
- `gcloud alpha compute tpus tpu-vm ssh root@$TPU_NAME --project $PROJECT --zone $ZONE --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=$HOSTKEY --command 'grep -n \"step=\" /root/MLLM-JAX/logs/nohup_grpo_gsm8k_qwen25_3b_bs128_steps100_latest.log | tail -n 5 || true'`
- `gcloud alpha compute tpus tpu-vm ssh root@$TPU_NAME --project $PROJECT --zone $ZONE --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=$HOSTKEY --command 'cat /root/MLLM-JAX/logs/nohup_grpo_gsm8k_qwen25_3b_bs128_steps100_latest.exit'`  # expect `0`

### 9) Extract final metrics from `wandb-summary.json` (no W&B API needed)

- `gcloud alpha compute tpus tpu-vm ssh root@$TPU_NAME --project $PROJECT --zone $ZONE --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=$HOSTKEY --command "set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; python -c 'import json,sys; d=json.load(open(sys.argv[1])); [print(k, d.get(k)) for k in sys.argv[2:]]' /root/MLLM-JAX/wandb/latest-run/files/wandb-summary.json train-reward/func/reward_correct/mean eval/reward/func/reward_correct/mean train-reward/total/mean eval/reward/total/mean time/train/step_avg_last10_s train-other/batch_local train-other/batch_global"`

## Expected Result

- `logs/nohup_grpo_gsm8k_qwen25_3b_bs128_steps100_latest.exit` contains `0`.
- `wandb-summary.json` includes final `train/*` + `eval/*` metrics.

## Observed Result (this verified run)

- W&B run: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/digdipr9`
- Ran to `step=62`, then manually stopped (no `.exit` file written for this run)

## Troubleshooting

- v6e create fails in a zone with `TPUV6E* limit 0`: use `gcloud alpha services quota list --consumer=projects/civil-rarity-482610-s5 --service=tpu.googleapis.com --filter='metric:tpu-v6e-preemptible'` and pick a zone with non-zero `effectiveLimit` (this run used `us-east1-d`).
- PuTTY host key prompt (Windows): use `--ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:...` (SSH) and `--scp-flag=-batch --scp-flag=-hostkey --scp-flag=SHA256:...` (SCP).

## References

- `docs/sops/tpu-vm-create-v4-8-or-v6e-8.md`
- `docs/sops/tpu-vm-repo-sync.md`
- `docs/sops/tpu-vm-v4-16-grpo-gsm8k-wandb-100steps.md`
- `scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh`
- `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml`
