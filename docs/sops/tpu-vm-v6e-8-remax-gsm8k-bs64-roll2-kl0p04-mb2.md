# SOP: TPU v6e-8 ReMax/GSM8K (bs64, roll2 incl. greedy baseline, KL=0.04, mb2)

- **Title**: SOP: Run official-aligned ReMax on GSM8K with Qwen2.5-3B-Instruct on TPU v6e-8 (W&B online)
  **Prereqs**: gcloud configured; TPU VM exists; conda env `mllm-jax`; W&B logged in on TPU
  **Scope**: `projects/gsm8k_grpo` runner + configs

## Goal

Run an end-to-end ReMax training loop (rollout → reward → advantage → update) with **official-aligned grouping**:

- `rollout.batch_size=64` prompts/step (global)
- `rollout.n=2` sequences per prompt-group (**1 greedy baseline + 1 sampled**)
- sequences/step = `64*2 = 128` (matches GRPO `16*8` compute)
- token-local KL shaping via `train.beta=0.04`
- `steps=400`, `eval_full_every_steps=50`

## Why mb2

With KL shaping enabled (`train.beta > 0`, ref model forward), v6e-8 can hit JAX/XLA compile HBM OOM at larger train micro-batches.
The config used here sets `train.micro_batch_size_per_device=2` (and `grad_accum_steps=8`) to fit.

## Steps (commands used)

Local (this repo):

```bash
git checkout test-rl
ls projects/gsm8k_grpo/configs/remax_gsm8k_qwen25_3b_batch64_roll2_steps400_evalfull50_kl0p04_mb2_ckptgcs_v6e8_testmonitor.yaml

git push origin HEAD
```

TPU VM (example TPU name + zone):

```bash
# Sync repo via git (policy: no scp for code)
scripts/ssh_tpu_vm_root.sh \
  --name test-rl-grpo-baseline-v6e-8-260212192520 \
  --zone europe-west4-a \
  --command 'cd /root/MLLM-JAX && git pull'

# (Optional) If a previous job is still running, stop it
scripts/ssh_tpu_vm_root.sh \
  --name test-rl-grpo-baseline-v6e-8-260212192520 \
  --zone europe-west4-a \
  --command 'cd /root/MLLM-JAX && pid=$(cat logs/nohup_remax_*_latest.pid); kill $pid'

# Start training via nohup (YAML-only, no hyperparam overrides)
scripts/ssh_tpu_vm_root.sh \
  --name test-rl-grpo-baseline-v6e-8-260212192520 \
  --zone europe-west4-a \
  --command 'cd /root/MLLM-JAX && bash scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh --config projects/gsm8k_grpo/configs/remax_gsm8k_qwen25_3b_batch64_roll2_steps400_evalfull50_kl0p04_mb2_ckptgcs_v6e8_testmonitor.yaml'

# Monitor
scripts/ssh_tpu_vm_root.sh \
  --name test-rl-grpo-baseline-v6e-8-260212192520 \
  --zone europe-west4-a \
  --command 'cd /root/MLLM-JAX && tail -n 50 logs/nohup_remax_gsm8k_qwen25_3b_batch64_roll2_steps400_evalfull50_kl0p04_mb2_ckptgcs_v6e8_testmonitor_latest.log'
```

## Expected Result

- Log prints `sequences_global_per_step: 128` and per-step lines like `step=N ... entropy=...`.
- W&B run link is printed and metrics stream online.
- Job exits with code `0` and writes checkpoints to the configured GCS path.

## Troubleshooting

- `open(/dev/vfio/1): Device or resource busy` during TPU init:
  - Ensure no stale python/JAX process is still holding VFIO (check with `fuser -v /dev/vfio/1`).
  - Wait ~1–2 minutes after killing a job and retry.
  - Confirm `/tmp/libtpu_lockfile` is removed.
- `RESOURCE_EXHAUSTED ... hbm` during compile:
  - Reduce `train.micro_batch_size_per_device` further (create a new YAML).

## References

- Config: `projects/gsm8k_grpo/configs/remax_gsm8k_qwen25_3b_batch64_roll2_steps400_evalfull50_kl0p04_mb2_ckptgcs_v6e8_testmonitor.yaml`
- ReMax rollout backend: `plugins/training/rl/rollout/backends/remax_mixed_naive.py`
- ReMax loss module: `plugins/training/rl/remax/module.py`
- Runner: `projects/gsm8k_grpo/jax/train.py`
