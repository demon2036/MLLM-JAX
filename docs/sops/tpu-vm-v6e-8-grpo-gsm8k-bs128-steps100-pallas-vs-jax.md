# TPU VM v6e-8 GRPO/GSM8K 100-step Train: Pallas kernel vs JAX baseline (W&B online)

- **Title**: SOP: Run GRPO/GSM8K 100 steps on v6e-8 and compare `policy_loss_impl=pallas` vs `jax`
  **Prereqs**: `gcloud` authenticated; TPU VM already exists; repo branch pushed; W&B API key available on TPU; outbound internet (HF + datasets + wandb)

## Scope

This SOP verifies that swapping GRPO policy-loss implementation from pure JAX to a JAX Pallas kernel:

- runs end-to-end on TPU
- logs to W&B (`wandb_mode=online`)
- produces aligned GSM8K correctness metrics after 100 steps

## Target TPU VM (this run)

- Project: `civil-rarity-482610-s5`
- TPU name: `grpo-gsm8k-pallas-evalfull-v6e8-260128053709`
- Zone: `us-east1-d`
- Type: `v6e-8` (spot)
- Host key (Windows plink): `SHA256:48Az6fEL08v+PTc43Py5TrTryjsZuHf8mWYUCTKapnU`
- Repo: `https://github.com/demon2036/MLLM-JAX.git`, branch `test_rl`, commit `a306f8f`

## Configs

- Baseline (JAX): `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml`
- Pallas: `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100_pallas.yaml` (`train.policy_loss_impl: pallas`)

## Steps (commands actually used)

This run also enabled stable, full test-split accuracy via:
- `EVAL_FULL_SWEEP=1` (runs full GSM8K test split after training; logs `eval_full/accuracy`).

### 0) SSH sanity check

```bash
TPU_NAME=grpo-gsm8k-pallas-evalfull-v6e8-260128053709
ZONE=us-east1-d
PROJECT=civil-rarity-482610-s5
HOSTKEY=SHA256:48Az6fEL08v+PTc43Py5TrTryjsZuHf8mWYUCTKapnU

gcloud alpha compute tpus tpu-vm ssh root@$TPU_NAME --project $PROJECT --zone $ZONE --worker 0 --quiet \
  --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=$HOSTKEY \
  --command 'set -euo pipefail; whoami; cat /etc/os-release | head -n 3; uname -r; python3 --version || true'
```

### 1) Git sync (no SCP for code)

```bash
gcloud alpha compute tpus tpu-vm ssh root@$TPU_NAME --project $PROJECT --zone $ZONE --worker 0 --quiet \
  --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=$HOSTKEY \
  --command 'set -euo pipefail; REPO_URL=https://github.com/demon2036/MLLM-JAX.git; REPO_DIR=/root/MLLM-JAX; if [ ! -d "$REPO_DIR/.git" ]; then rm -rf "$REPO_DIR"; git clone "$REPO_URL" "$REPO_DIR"; fi; cd "$REPO_DIR"; git fetch --all --prune; git checkout test_rl; git reset --hard origin/test_rl; git clean -fd; echo HEAD:; git rev-parse --short HEAD; git status -sb'
```

### 2) Ensure WANDB_API_KEY is available on TPU (online mode)

This run used `/root/.env` (synced via `tpu-vm scp`) and verified the key length without printing it:

```bash
gcloud alpha compute tpus tpu-vm scp .env root@$TPU_NAME:/root/.env --project $PROJECT --zone $ZONE --worker 0 --quiet \
  --scp-flag=-batch --scp-flag=-hostkey --scp-flag=$HOSTKEY

gcloud alpha compute tpus tpu-vm ssh root@$TPU_NAME --project $PROJECT --zone $ZONE --worker 0 --quiet \
  --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=$HOSTKEY \
  --command 'set -euo pipefail; chmod 600 /root/.env; test -f /root/.env; key_len=$(grep -E "^WANDB_API_KEY=" /root/.env | head -n 1 | cut -d= -f2- | tr -d "\r\n" | wc -c); echo WANDB_API_KEY_len=$key_len'
```

### 3) Run baseline (JAX) 100 steps

Start (nohup):

```bash
gcloud alpha compute tpus tpu-vm ssh root@$TPU_NAME --project $PROJECT --zone $ZONE --worker 0 --quiet \
  --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=$HOSTKEY \
  --command 'set -euo pipefail; cd /root/MLLM-JAX; export EVAL_FULL_SWEEP=1; export PRINT_TRAIN_TIME_BREAKDOWN=1; bash scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh --env-name mllm-jax --config plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml'
```

Run artifacts:
- PID: `11522`
- Log: `logs/nohup_grpo_gsm8k_qwen25_3b_bs128_steps100_20260128_055409.log`
- Exit: `logs/nohup_grpo_gsm8k_qwen25_3b_bs128_steps100_20260128_055409.exit` (observed: `0`)
- W&B run: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/mla6lay5`

Monitor:

```bash
gcloud alpha compute tpus tpu-vm ssh root@$TPU_NAME --project $PROJECT --zone $ZONE --worker 0 --quiet \
  --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=$HOSTKEY \
  --command 'ps -p 11522 -o pid=,etime=,cmd= || echo process_not_running'
```

### 4) Run Pallas 100 steps

Start (nohup):

```bash
gcloud alpha compute tpus tpu-vm ssh root@$TPU_NAME --project $PROJECT --zone $ZONE --worker 0 --quiet \
  --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=$HOSTKEY \
  --command 'set -euo pipefail; cd /root/MLLM-JAX; export EVAL_FULL_SWEEP=1; export PRINT_TRAIN_TIME_BREAKDOWN=1; bash scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh --env-name mllm-jax --config plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100_pallas.yaml'
```

Run artifacts:
- PID: `39836`
- Log: `logs/nohup_grpo_gsm8k_qwen25_3b_bs128_steps100_pallas_20260128_062323.log`
- Exit: `logs/nohup_grpo_gsm8k_qwen25_3b_bs128_steps100_pallas_20260128_062323.exit` (observed: `0`)
- W&B run: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/v1zgrlmm`

### 5) Extract final metrics from `wandb-summary.json`

Command used (run-specific file path; avoids `latest-run` ambiguity):

```bash
gcloud alpha compute tpus tpu-vm ssh root@$TPU_NAME --project $PROJECT --zone $ZONE --worker 0 --quiet \
  --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=$HOSTKEY \
  --command "set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; python -c 'import json,sys; d=json.load(open(sys.argv[1]));\nfor k in sys.argv[2:]:\n  print(k, d.get(k))' /root/MLLM-JAX/wandb/run-20260128_055444-mla6lay5/files/wandb-summary.json eval_full/accuracy eval/reward/func/reward_correct/mean eval/reward/total/mean time/train/step_avg_last10_s time/eval_full/step_s"

gcloud alpha compute tpus tpu-vm ssh root@$TPU_NAME --project $PROJECT --zone $ZONE --worker 0 --quiet \
  --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=$HOSTKEY \
  --command "set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate mllm-jax; cd /root/MLLM-JAX; python -c 'import json,sys; d=json.load(open(sys.argv[1]));\nfor k in sys.argv[2:]:\n  print(k, d.get(k))' /root/MLLM-JAX/wandb/run-20260128_062351-v1zgrlmm/files/wandb-summary.json eval_full/accuracy eval/reward/func/reward_correct/mean eval/reward/total/mean time/train/step_avg_last10_s time/eval_full/step_s"
```

Observed summary:

- Baseline (jax, run `mla6lay5`):
  - `eval_full/accuracy = 0.7831690675`
  - `eval/reward/func/reward_correct/mean = 0.796875`
  - `eval/reward/total/mean = 1.796875`
  - `time/train/step_avg_last10_s = 11.7047013686`
  - `time/eval_full/step_s = 89.5696885040`
- Pallas (run `v1zgrlmm`):
  - `eval_full/accuracy = 0.7968157695`
  - `eval/reward/func/reward_correct/mean = 0.75`
  - `eval/reward/total/mean = 1.75`
  - `time/train/step_avg_last10_s = 12.9079808251`
  - `time/eval_full/step_s = 109.5209415700`

## Expected result

- Both runs complete with exit code `0`.
- Both runs appear in W&B (online) and have comparable training/eval curves.
- Final `eval/reward/func/reward_correct/mean` is aligned (report deltas).
- With `EVAL_FULL_SWEEP=1`, also compare stable `eval_full/accuracy`.

## Troubleshooting

- If TPU SSH prompts for host key (Windows plink): use `--ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:<fingerprint>`.
- If W&B init fails: verify `WANDB_API_KEY` is set on TPU and `wandb_mode: online` in YAML.

## References

- `scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh`
- `scripts/run_grpo_gsm8k_training.py`
- `plugins/training/kernels/grpo_loss_pallas.py`
