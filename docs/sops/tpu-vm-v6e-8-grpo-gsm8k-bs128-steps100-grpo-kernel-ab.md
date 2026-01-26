# TPU VM v6e-8 (spot, 1-host) GRPO/GSM8K 100-step A/B: baseline vs GRPO Pallas kernel (W&B)

- **Title**: SOP: Run GRPO/GSM8K for 100 steps on a `v6e-8` TPU VM and compare baseline vs `train.grpo_kernel.enabled=true`
  **Prereqs**: TPU VM reachable via `gcloud ... tpu-vm ssh`; outbound internet (HF + datasets + W&B); repo synced via Git (no SCP); `/root/.env` contains `WANDB_API_KEY`; configs are committed YAMLs (no env-var hyperparam overrides)
  **Environment (verified)**:
  - TPU VM: `mllm-jax-v6e-8-grpo-kernel-260127045556` (`v6e-8`, spot), zone `europe-west4-a`
  - Conda env: `mllm-jax` (Python `3.12.12`)
  - JAX/jaxlib/libtpu: `0.9.0` / `0.9.0` / `0.0.34`, `device_count=8`
  - Repo: `https://github.com/demon2036/MLLM-JAX.git`, branch `feat-kernel`
  - W&B project: `mllm-jax-grpo-gsm8k`

## Goal

- Run two 100-step trainings with the same GRPO/GSM8K setup (Qwen2.5-3B, global sequences=128/step) and compare:
  - baseline: `train.grpo_kernel.enabled=false`
  - kernel: `train.grpo_kernel.enabled=true` (Pallas loss + streaming entropy)

## Configs

- Baseline: `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml`
- Kernel: `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100_pallas_kernel.yaml`

## Steps (commands actually used)

### 1) Start a run (nohup helper)

- `cd /root/MLLM-JAX`
- `bash scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh --env-name mllm-jax --config <CONFIG_PATH>.yaml`

### 2) Monitor and verify exit code

- `cd /root/MLLM-JAX`
- `tail -n 20 logs/nohup_<config_tag>_latest.log`
- `cat logs/nohup_<config_tag>_latest.exit`  # expect `0`

### 3) Extract final metrics (no W&B API needed)

- `cd /root/MLLM-JAX`
- `python - <<\"PY\"\nimport json,sys\np=sys.argv[1]\nd=json.load(open(p))\nkeys=sys.argv[2:]\nfor k in keys:\n  print(k,'=',d.get(k))\nPY wandb/latest-run/files/wandb-summary.json time/train/step_avg_last10_s time/train/update_s throughput/train/valid_tokens_per_s_update train-other/total_valid_token_count eval/reward/total/mean eval/reward/func/reward_correct/mean`

## Observed Result (A/B runs)

### Baseline (enabled=false)

- Config: `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml`
- Commit: `0fe25c6`
- Exit: `0`
- W&B: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/aovd31pm`
- `wandb-summary.json` highlights:
  - `time/train/step_avg_last10_s=11.578276114599998`
  - `time/train/update_s=3.932779269999628`
  - `throughput/train/valid_tokens_per_s_update=6414.547643809765`
  - `train-other/total_valid_token_count=25227`
  - `eval/reward/total/mean=1.796875`

### Kernel (enabled=true)

- Config: `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100_pallas_kernel.yaml`
- Commit: `5e8ff7e` (includes `entropy_pallas` JVP-safe fix)
- Exit: `0`
- W&B: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/ucesd0oc`
- `wandb-summary.json` highlights:
  - `time/train/step_avg_last10_s=13.63050850999998`
  - `time/train/update_s=4.2935676090000925`
  - `throughput/train/valid_tokens_per_s_update=8186.432170375367`
  - `train-other/total_valid_token_count=35149` (longer completions)
  - `eval/reward/total/mean=1.828125`

## Notes

- The absolute `time/train/step_*` is strongly affected by completion length (`train-other/total_valid_token_count`), so `throughput/*` is a better apples-to-apples signal when rollouts differ.

## References

- `memory/20260126_grpo-pallas-kernel-multidevice/README.md`
- `docs/sops/tpu-vm-repo-sync.md`
- `docs/sops/grpo-pallas-kernel-gradcheck.md`
