# GRPO/GSM8K: clean re-run under a new W&B project (wipe GCS checkpoints; resume=false)

- **Title**: SOP: Clean GRPO re-run in W&B `test-monitor` without inheriting checkpoints
  **Prereqs**: `gcloud`, `gsutil`, SSH key `~/.ssh/google_compute_engine`, TPU VM(s) reachable
  **Scope**: GCS checkpoint buckets + TPU VM job control + committed YAML configs

## Goal

Guarantee a fresh start (no checkpoint restore) by:
1) stopping TPU jobs, 2) wiping/deleting the old checkpoint bucket, 3) creating a new bucket, 4) re-running with `checkpoint.resume: false` and `wandb_project: test-monitor`.

## Steps

### 1) Stop running jobs on TPU VMs

If `gcloud alpha compute tpus tpu-vm ssh ...` is unreliable (ssh exit `255`), use direct SSH via external IP:

```bash
# Find external IP
gcloud alpha compute tpus tpu-vm describe <TPU_NAME> --zone <ZONE> --format='get(networkEndpoints[0].accessConfig.externalIp)'

# Kill the job (PID from `ps aux | grep run_train.py`)
ssh -i ~/.ssh/google_compute_engine root@<EXTERNAL_IP> "ps aux | grep '[p]rojects/gsm8k_grpo/scripts/run_train.py' || true"
ssh -i ~/.ssh/google_compute_engine root@<EXTERNAL_IP> "kill -KILL <PID>"
```

### 2) Wipe + delete the old checkpoint bucket

```bash
gsutil -m rm -r gs://<OLD_BUCKET>/**
gsutil rb gs://<OLD_BUCKET>
```

### 3) Create a new checkpoint bucket

```bash
gsutil mb -p <GCP_PROJECT_ID> -l <LOCATION> gs://<NEW_BUCKET>
```

Example (used in this task):
- `<LOCATION>=europe-west4`
- `<NEW_BUCKET>=mllm-jax-test-monitor-ckpt-516801686594-260213`

### 4) Re-run with committed YAMLs (no resume)

Use the `*_testmonitor.yaml` configs under `projects/gsm8k_grpo/configs/`:

- `..._ckptgcs_v6e8_testmonitor.yaml` (baseline, token loss)
- `..._adv0entropy0p01_ckptgcs_v6e8_testmonitor.yaml` (token loss + adv0 entropy)
- `..._seqloss_ckptgcs_v6e8_testmonitor.yaml` (baseline, sequence loss)
- `..._seqloss_adv0entropy0p01_ckptgcs_v6e8_testmonitor.yaml` (sequence loss + adv0 entropy)

Launch on TPU VM:

```bash
bash scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh --env-name mllm-jax --config <CONFIG_YAML>
```

## Expected result

- TPU logs show `step=0 ...` and **do not** show `checkpoint_restore ...`.
- W&B runs appear in project `test-monitor` with `wandb_mode=online`.
- `eval_full` prints every 50 steps.

## Monitoring metrics (single bundle)

Dashboards can filter by `monitor/*`:
- `monitor/train/entropy`
- `monitor/train/adv_zero_fraction`
- `monitor/train/adv_token_weighted_minus_mean`
- `monitor/train/adv_completion_len_corr`
- `monitor/train/entropy_adv0_reg`
- `monitor/eval_full/accuracy`

