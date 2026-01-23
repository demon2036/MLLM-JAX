# Task
- Validate the current 4-phase RL refactor on a fresh `tpu-v6e-8` run and confirm it behaves the same as before (no errors; training loop runs end-to-end).
- Modularize the optimizer so it can be configured/passed in (instead of being hardcoded in `training2.get_state`).

# Plan
1) Create memory folder and log.
2) Run local pytest sanity check.
3) Commit and push four-phase changes.
4) Create v6e-8 TPU VM.
5) Bootstrap conda and git-sync.
6) Install TPU Python requirements.
7) Launch TPU smoke training job.
8) Monitor TPU log and exitcode.
9) Design optimizer config schema.
10) Implement optimizer builder module.
11) Wire optimizer into state init.
12) Run tests and update SOP.
13) Commit and push optimizer changes.
14) Run TPU smoke new commit.
15) Delete TPU VM after validation.

## Step 1 - Create memory folder and log
Completion criteria: `memory/20260123_tpuv6e-validate-optimizer/README.md` exists and `memory/README.md` is updated.
Evidence:
- Added `memory/20260123_tpuv6e-validate-optimizer/README.md`.
- Updated `memory/README.md` with new task entry.

## Step 2 - Run local pytest sanity check
Completion criteria: local test suite passes with exit code 0.
Evidence:
- Command (exit 0): `python -m pytest -q`
- Output: `14 passed in 0.85s`

## Step 3 - Commit and push four-phase changes
Completion criteria: local changes are committed and pushed to `origin/main` so TPU can Git-sync the exact revision.
Evidence:
- Commit: `c8d87b2` (`refactor: phase-based rl training layout`)
- Command (exit 0): `git push`

## Step 4 - Create v6e-8 TPU VM
Completion criteria: a `v6e-8` TPU VM exists and reaches `READY`.
Evidence:
- Failed attempts (spot):
  - `us-east1-d`: capacity exhausted
  - `us-central2-b`: spot quota limit 0
  - `us-central1-b`: capacity exhausted
  - `us-east5-b`: capacity exhausted
  - `us-east5-c`: spot quota limit 0
  - `us-east4-a`: spot quota limit 0
  - `us-east5-a` / `us-central1-c` / `us-west1-c`: `Reservation not found`
- Successful create (exit 0):
  - `gcloud alpha compute tpus tpu-vm create mllm-jax-v6e-8-260123075313 --project civil-rarity-482610-s5 --zone europe-west4-a --accelerator-type v6e-8 --version v6e-ubuntu-2404 --spot --quiet`
  - `gcloud alpha compute tpus tpu-vm describe mllm-jax-v6e-8-260123075313 --project civil-rarity-482610-s5 --zone europe-west4-a --format='value(state,acceleratorType,networkEndpoints[0].ipAddress)'`
  - Output: `READY v6e-8 10.164.15.194`
- SSH host key fingerprint used: `SHA256:JvL8NPPiQQRJdmVaS/9OOKaHZE0XTgvat42bzU38Owo`

## Step 5 - Bootstrap conda and git-sync
Completion criteria: SSH works; Miniconda installed; conda env created; repo cloned and checked out on TPU VM.
Evidence (commands exit 0):
- SSH check:
  - `gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-260123075313 --project civil-rarity-482610-s5 --zone europe-west4-a --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:JvL8NPPiQQRJdmVaS/9OOKaHZE0XTgvat42bzU38Owo --command "whoami; cat /etc/os-release | sed -n '1,3p'; uname -r; python3 --version || true"`
- Miniconda bootstrap:
  - `gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-260123075313 --project civil-rarity-482610-s5 --zone europe-west4-a --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:JvL8NPPiQQRJdmVaS/9OOKaHZE0XTgvat42bzU38Owo --command "set -euo pipefail; if [ ! -d /root/miniconda3 ]; then curl -fsSL -o /root/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; bash /root/miniconda.sh -b -p /root/miniconda3; rm -f /root/miniconda.sh; fi; /root/miniconda3/bin/conda --version"`
- Conda env create:
  - `gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-260123075313 --project civil-rarity-482610-s5 --zone europe-west4-a --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:JvL8NPPiQQRJdmVaS/9OOKaHZE0XTgvat42bzU38Owo --command "set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true; conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true; if ! conda env list | grep -Eq '^mllm-jax[[:space:]]'; then conda create -y -n mllm-jax python=3.12; fi; conda activate mllm-jax; python --version; python -m pip install -U pip"`
- Git sync (checkout exact commit):
  - `gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-260123075313 --project civil-rarity-482610-s5 --zone europe-west4-a --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:JvL8NPPiQQRJdmVaS/9OOKaHZE0XTgvat42bzU38Owo --command 'set -euo pipefail; REPO_URL=https://github.com/demon2036/MLLM-JAX.git; REPO_DIR=/root/MLLM-JAX; if [ ! -d \"$REPO_DIR/.git\" ]; then rm -rf \"$REPO_DIR\"; git clone \"$REPO_URL\" \"$REPO_DIR\"; fi; cd \"$REPO_DIR\"; git fetch --all --prune; git checkout c8d87b2; git status -sb'`

## Step 6 - Install TPU Python requirements
Completion criteria: TPU runtime Python deps are installed in the conda env (`jax[tpu]` + `requirements-tpu.txt`).
Evidence:
- Command (exit 0):
  - `gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-260123075313 --project civil-rarity-482610-s5 --zone europe-west4-a --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:JvL8NPPiQQRJdmVaS/9OOKaHZE0XTgvat42bzU38Owo --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; python -m pip install -U pip; python -m pip install -U \"jax[tpu]\" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; python -m pip install -U torch --index-url https://download.pytorch.org/whl/cpu; cd /root/MLLM-JAX; python -m pip install -U -r requirements-tpu.txt'`
- Observed versions from install logs:
  - `jax==0.9.0`, `jaxlib==0.9.0`, `libtpu==0.0.34`

## Step 7 - Launch TPU smoke training job
Completion criteria: start a short GRPO/GSM8K run on TPU without immediate errors (runs in background via `nohup`).
Evidence:
- Command (exit 0):
  - `gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-260123075313 --project civil-rarity-482610-s5 --zone europe-west4-a --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:JvL8NPPiQQRJdmVaS/9OOKaHZE0XTgvat42bzU38Owo --command 'set -euo pipefail; cd /root/MLLM-JAX; mkdir -p logs; rm -f /tmp/libtpu_lockfile || true; ts=$(date -u +%Y%m%d_%H%M%S); LOG=logs/nohup_grpo_gsm8k_smoke_${ts}.log; nohup env WANDB_MODE=disabled HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false /root/miniconda3/envs/mllm-jax/bin/python -u scripts/run_grpo_gsm8k_training.py --config plugins/training/configs/grpo_gsm8k_default.yaml --set steps=3 --set rollout.batch_size=1 --set rollout.n=8 --set rollout.max_length_sample=64 --set eval_every_steps=0 > \"$LOG\" 2>&1 & pid=$!; echo $pid > \"${LOG}.pid\"; echo PID=$pid; echo LOG=$LOG'`
- Output:
  - `PID=13003`
  - `LOG=logs/nohup_grpo_gsm8k_smoke_20260123_081939.log`

## Step 8 - Design optimizer config schema
Completion criteria: decide a stable, minimal config schema for selecting the optimizer (no longer hardcoded), while keeping backward-compatibility.
Design:
- Add `train.optimizer` to YAML configs (optional; default preserves current behavior).
- Schema (defaults match current `training2.get_state`):
  - `train.optimizer.name`: `lion|adamw|sgd` (default `lion`)
  - `train.optimizer.clip_norm`: float (default `1.0`)
  - `train.optimizer.weight_decay`: float (default `1e-8`)
  - `train.optimizer.lr_schedule.type`: `warmup_cosine|constant` (default `warmup_cosine`)
  - `train.optimizer.lr_schedule.peak_value`: float (default `1e-6`)
  - `train.optimizer.lr_schedule.warmup_ratio`: float (default `0.05`)
  - `train.optimizer.lr_schedule.warmup_steps`: int|null (default null; derived from `steps * warmup_ratio`)
  - `train.optimizer.lr_schedule.init_value`: float (default `0.0`)
  - `train.optimizer.lr_schedule.end_value`: float (default `0.0`)
Plan for wiring:
- Add `tx` passthrough to `training2.get_state(..., tx=None)` (if provided, use it; else keep current default).
- Build `tx` from `cfg.train.optimizer` in `plugins/training/update/optimizer.py` and pass into `get_state` from the runner.

## Step 9 - Implement optimizer builder module
Completion criteria: an Optax optimizer builder exists under the update phase, and can construct the default (current) optimizer/schedule from config.
Evidence:
- Added `plugins/training/update/optimizer.py` (dataclasses: `OptimizerConfig`, `LRScheduleConfig`; builders: `build_lr_schedule`, `build_tx`).

## Step 10 - Wire optimizer into state init
Completion criteria: the training state creation accepts a pluggable optimizer (`tx`), and the runner/config path can specify it.
Evidence:
- Updated `training2.get_state(..., tx=None)` to use `tx` when provided (otherwise preserves previous default).
- Updated `plugins/training/runner/grpo_gsm8k.py` to build `tx` via `plugins/training/update/optimizer.py` and pass it into `get_state`.
- Updated `scripts/run_grpo_gsm8k_training.py` to parse `train.optimizer` into an `OptimizerConfig`.
- Updated `plugins/training/config.py` default config to include `train.optimizer` block (defaults match prior hardcoded behavior).

## Step 11 - Run tests and update SOP
Completion criteria: local tests pass and a deterministic SOP exists for the optimizer modularization.
Evidence:
- Tests (exit 0): `python -m pytest -q` -> `14 passed in 0.92s`
- Added SOP: `docs/sops/rl-pluggable-optimizer.md`
- Updated SOP index: `docs/sops.md`

## Step 12 - Commit and push optimizer changes
Completion criteria: optimizer modularization changes are committed and pushed to `origin/main` so TPU can Git-sync them.
Evidence:
- Commit: `c05519c` (`feat: pluggable optimizer config`)
- Command (exit 0): `git push`

## Step 13 - Monitor TPU log and exitcode
Completion criteria: the TPU smoke job finishes without traceback and reaches the expected number of steps.
Evidence:
- Job PID: `13003` (no longer running)
- Log: `logs/nohup_grpo_gsm8k_smoke_20260123_081939.log`
- Observed tail highlights:
  - `backend=tpu process=0/1`
  - Completed: `step=0 ...`, `step=1 ...`, `step=2 ...`
  - `grep -n "Traceback" ...` returned no matches
Note:
- This ad-hoc `nohup` launcher did not write an explicit exit code file; we used “no traceback + final step printed + process exited” as success criteria.

## Step 14 - Run TPU smoke new commit
Completion criteria: run a short GRPO/GSM8K job on the latest commit, using a non-default optimizer config, to validate the optimizer modularization end-to-end.
Evidence:
- Git sync (exit 0):
  - `gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-260123075313 --project civil-rarity-482610-s5 --zone europe-west4-a --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:JvL8NPPiQQRJdmVaS/9OOKaHZE0XTgvat42bzU38Owo --command 'set -euo pipefail; cd /root/MLLM-JAX; git fetch --all --prune; git checkout c05519c; git status -sb'`
- Launched smoke job (exit 0):
  - `PID=17886`
  - `LOG=logs/nohup_grpo_gsm8k_opt_20260123_083236.log`
  - Command included override: `--set train.optimizer.name=adamw`
- Log highlights:
  - Printed optimizer block with `name: adamw`
  - Completed: `step=0 loss=...` (with `steps: 1`)
  - `grep -n "Traceback" ...` returned no matches

## Step 15 - Delete TPU VM after validation
Completion criteria: the TPU VM used for validation is deleted (to stop billing).
Evidence:
- Command (exit 0): `gcloud alpha compute tpus tpu-vm delete mllm-jax-v6e-8-260123075313 --project civil-rarity-482610-s5 --zone europe-west4-a --quiet`
- Output: `Deleted tpu [mllm-jax-v6e-8-260123075313].`

---

# Follow-up: run v6e-8 with W&B online (urgent)

## Step 16 - Extract WANDB env from existing TPU
Completion criteria: obtain a `WANDB_API_KEY` without printing it, by copying an existing `/root/.env` from an already-provisioned TPU.
Evidence:
- Existing TPU (v4-8): `mllm-jax-v4-8-260122100610` (`us-central2-b`)
- Remote check (exit 0): `/root/.env` exists
  - `gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v4-8-260122100610 --project civil-rarity-482610-s5 --zone us-central2-b --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:Xy1NoS+m4LpYQNWiLlkc3Co5iIhoPzAFC39CNLmSN3s --command "set -euo pipefail; if [ -f /root/.env ]; then echo ENV_PRESENT; else echo ENV_MISSING; fi"`
- Remote check (exit 0): `.env` contains `WANDB_API_KEY` (value not printed)
  - `gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v4-8-260122100610 --project civil-rarity-482610-s5 --zone us-central2-b --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:Xy1NoS+m4LpYQNWiLlkc3Co5iIhoPzAFC39CNLmSN3s --command 'set -euo pipefail; set -a; source /root/.env; set +a; if [ -n "${WANDB_API_KEY:-}" ]; then echo WANDB_API_KEY_PRESENT; else echo WANDB_API_KEY_MISSING; fi'`
- Copied `/root/.env` to a local temp file (exit 0; content not printed):
  - `gcloud alpha compute tpus tpu-vm scp root@mllm-jax-v4-8-260122100610:/root/.env workdir/tpu_root_env.env --project civil-rarity-482610-s5 --zone us-central2-b --worker 0 --quiet --scp-flag=-batch --scp-flag=-hostkey --scp-flag=SHA256:Xy1NoS+m4LpYQNWiLlkc3Co5iIhoPzAFC39CNLmSN3s`
- Cleanup (exit 0): deleted local temp file after copying to v6e
  - `Remove-Item -Force workdir/tpu_root_env.env`

## Step 17 - Create v6e-8 TPU VM
Completion criteria: a new `v6e-8` TPU VM reaches `READY`.
Evidence:
- Created: `mllm-jax-v6e-8-260123090716` in `europe-west4-a` (spot), internal IP `10.164.15.195`
- SSH host key fingerprint used: `SHA256:CPEH49k4vjIrlHWA0i/upPfisxSH0ZnEC7gPDgZp/1M`

## Step 18 - Bootstrap conda environment on v6e-8
Completion criteria: Miniconda is installed and the `mllm-jax` env exists.
Evidence (exit 0):
- Miniconda bootstrap:
  - `gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-260123090716 --project civil-rarity-482610-s5 --zone europe-west4-a --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:CPEH49k4vjIrlHWA0i/upPfisxSH0ZnEC7gPDgZp/1M --command 'set -euo pipefail; if [ ! -d /root/miniconda3 ]; then curl -fsSL -o /root/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; bash /root/miniconda.sh -b -p /root/miniconda3; rm -f /root/miniconda.sh; fi; /root/miniconda3/bin/conda --version'`
- Env create:
  - `gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-260123090716 --project civil-rarity-482610-s5 --zone europe-west4-a --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:CPEH49k4vjIrlHWA0i/upPfisxSH0ZnEC7gPDgZp/1M --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true; conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true; if ! conda env list | grep -Eq \"^mllm-jax[[:space:]]\"; then conda create -y -n mllm-jax python=3.12; fi; conda activate mllm-jax; python --version; python -m pip install -U pip'`

## Step 19 - Git-sync latest commit on v6e-8
Completion criteria: repo is cloned and checked out to the desired commit.
Evidence (exit 0):
- `git checkout 25c413f` in `/root/MLLM-JAX`

## Step 20 - Upload /root/.env to v6e-8
Completion criteria: `/root/.env` exists on the v6e VM and contains `WANDB_API_KEY` (without printing it).
Evidence:
- SCP (exit 0):
  - `gcloud alpha compute tpus tpu-vm scp workdir/tpu_root_env.env root@mllm-jax-v6e-8-260123090716:/root/.env --project civil-rarity-482610-s5 --zone europe-west4-a --worker 0 --quiet --scp-flag=-batch --scp-flag=-hostkey --scp-flag=SHA256:CPEH49k4vjIrlHWA0i/upPfisxSH0ZnEC7gPDgZp/1M`
- Verify (exit 0; value not printed):
  - `gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-260123090716 --project civil-rarity-482610-s5 --zone europe-west4-a --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:CPEH49k4vjIrlHWA0i/upPfisxSH0ZnEC7gPDgZp/1M --command 'set -euo pipefail; chmod 600 /root/.env; set -a; source /root/.env; set +a; if [ -n "${WANDB_API_KEY:-}" ]; then echo WANDB_API_KEY_PRESENT; else echo WANDB_API_KEY_MISSING; fi'`

## Step 21 - Install TPU Python requirements on v6e-8
Completion criteria: TPU deps are installed in the conda env.
Evidence:
- Command (exit 0): install `jax[tpu]`, cpu `torch`, and `requirements-tpu.txt`
- Installed versions (from logs): `jax==0.9.0`, `jaxlib==0.9.0`, `libtpu==0.0.34`

## Step 22 - Launch bs128/steps100 training with W&B online
Completion criteria: the official nohup launcher starts the training process in background.
Evidence:
- Command (exit 0):
  - `gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-260123090716 --project civil-rarity-482610-s5 --zone europe-west4-a --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:CPEH49k4vjIrlHWA0i/upPfisxSH0ZnEC7gPDgZp/1M --command 'set -euo pipefail; cd /root/MLLM-JAX; set -a; source /root/.env; set +a; export WANDB_MODE=online; export WANDB_PROJECT=mllm-jax-grpo-gsm8k; export ENV_NAME=mllm-jax; bash scripts/tpu_vm_start_grpo_gsm8k_bs128_steps100_nohup.sh'`
- Output:
  - `PID=9983`
  - `LATEST_LOG=logs/nohup_grpo_gsm8k_bs128_steps100_latest.log`
  - `LATEST_EXIT=logs/nohup_grpo_gsm8k_bs128_steps100_latest.exit`

## Step 23 - Confirm W&B online + job progressing
Completion criteria: W&B login succeeds and the process is running without traceback.
Evidence:
- Process check (exit 0): `ps -p 9983 ...` shows the job is running.
- W&B log lines (exit 0):
  - `Loaded credentials ... from WANDB_API_KEY`
  - Run URL: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/pge8fy3q`
- `Traceback` grep returned no matches at this point.

## Step 24 - Confirm OOM root cause on v6e-8 run
Completion criteria: capture the exact failure (stack trace) + the effective resolved config printed by the runner.
Evidence:
- Remote log/exit check (exit 0):
  - `gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-260123090716 --project civil-rarity-482610-s5 --zone europe-west4-a --worker 0 --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:CPEH49k4vjIrlHWA0i/upPfisxSH0ZnEC7gPDgZp/1M --command 'set -euo pipefail; cd /root/MLLM-JAX; echo COMMIT=$(git rev-parse --short HEAD); if [ -f logs/nohup_grpo_gsm8k_bs128_steps100_latest.log.pid ]; then pid=$(cat logs/nohup_grpo_gsm8k_bs128_steps100_latest.log.pid); echo PID=$pid; ps -p $pid -o pid,cmd || true; else echo PID_FILE_MISSING; fi; if [ -f logs/nohup_grpo_gsm8k_bs128_steps100_latest.exit ]; then echo EXIT_CODE=$(cat logs/nohup_grpo_gsm8k_bs128_steps100_latest.exit); else echo EXIT_FILE_MISSING; fi; echo --- OOM grep ---; grep -nE RESOURCE_EXHAUSTED\\|OOM logs/nohup_grpo_gsm8k_bs128_steps100_latest.log | tail -n 40 || true; echo --- tail log ---; tail -n 120 logs/nohup_grpo_gsm8k_bs128_steps100_latest.log'`
- Key findings from the log:
  - Model was `Qwen/Qwen2.5-7B-Instruct` (the `*_bs128_steps100` launcher defaults to the 7B config).
  - Runner resolved `rollout.batch_size=128` prompts and `rollout.n=8` -> `sequences_global_per_step=1024`, `local_batch=1024`, `global_batch=1024`.
  - Crash happened during rollout prefill (before any optimizer/update step):
    - `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Error allocating device buffer: Attempting to allocate 64.00M ... There are 41.14M free.`

## Step 25 - Decide correct config + bs semantics going forward
Completion criteria: pick the intended bs definition and the canonical config/script to run on v6e-8.
Evidence:
- Decision:
  - Treat `bsXXX` in filenames/docs as **global sequences per training step** (historical meaning in this repo).
  - Keep runner semantics: `rollout.batch_size` remains **global prompts per step**, so to get `bs=128` with `rollout.n=8`, set `rollout.batch_size=16` (since 16 prompts * 8 = 128 sequences).
  - For the user-facing v6e-8 W&B run, use the explicit 3B config: `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml` (and its launcher script), instead of the 7B default `*_bs128_steps100` launcher.

## Step 26 - Patch bs128 configs + launchers (avoid OOM)
Completion criteria: `*_bs128_*` YAMLs match `bs=128 sequences/step` intent and TPU launcher comments stop implying 1024 sequences.
Evidence:
- Updated YAML configs to use `rollout.batch_size=16` with `rollout.n=8` (128 sequences/step):
  - `plugins/training/configs/grpo_gsm8k_bs128_steps100.yaml`
  - `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml`
  - `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100_v6e16.yaml`
- Updated legacy timing configs to the current (non-deprecated) schema:
  - `plugins/training/configs/grpo_gsm8k_bs32.yaml`
  - `plugins/training/configs/grpo_gsm8k_bs32_naive_tp.yaml`
- Updated TPU launcher script headers to reflect the intended 128 sequences/step semantics:
  - `scripts/tpu_vm_start_grpo_gsm8k_bs128_steps100_nohup.sh`
  - `scripts/tpu_vm_start_grpo_gsm8k_qwen25_3b_bs128_steps100_nohup.sh`

## Step 27 - Update SOP notes for bs128 + recommended config
Completion criteria: SOPs stop implying `rollout.batch_size=128` means 128 sequences (and point to the correct 3B config for v6e-8).
Evidence:
- Updated batch semantics SOP:
  - `docs/sops/grpo-gsm8k-runner-batch-size.md`
- Added a current-`main` clarification note + recommended 3B config/script:
  - `docs/sops/tpu-vm-v6e-8-grpo-gsm8k-bs128-steps100.md`

## Step 28 - Run local config sanity checks + pytest
Completion criteria: updated YAML configs are accepted by `--print-config`, and `pytest` passes (exit 0).
Evidence:
- Config print checks (exit 0):
  - `python scripts/run_grpo_gsm8k_training.py --print-config --config plugins/training/configs/grpo_gsm8k_bs128_steps100.yaml`
    - Contains `batch_size: 16` and `sequences_global_per_step: 128`
  - `python scripts/run_grpo_gsm8k_training.py --print-config --config plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml`
    - Contains `batch_size: 16` and `sequences_global_per_step: 128`
  - `python scripts/run_grpo_gsm8k_training.py --print-config --config plugins/training/configs/grpo_gsm8k_bs32.yaml`
  - `python scripts/run_grpo_gsm8k_training.py --print-config --config plugins/training/configs/grpo_gsm8k_bs32_naive_tp.yaml`
  - `python scripts/run_grpo_gsm8k_training.py --print-config --config plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100_v6e16.yaml`
- Test suite (exit 0):
  - `python -m pytest -q`
  - Output: `14 passed in 0.88s`

## Step 29 - Commit and push bs128 config fix
Completion criteria: changes are committed and pushed to `origin/main` so the TPU VM can Git-sync them.
Evidence:
- Commit: `3c4bfe1` (`fix(config): align bs128 to 128 seq`)
- Push (exit 0): `git push`
