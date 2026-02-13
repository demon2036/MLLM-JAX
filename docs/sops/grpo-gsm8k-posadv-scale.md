# GRPO/GSM8K: Positive-advantage scaling (`pos_adv_scale`)

- **Title**: SOP: Scale only positive GRPO advantages (e.g. `adv>0` × 4) without touching `adv<=0`
  **Prereqs**: `gcloud` authenticated; TPU API enabled; W&B online requires a valid `WANDB_API_KEY`
  **Scope**: `plugins/training/rl/advantage/modules.py`, `plugins/training/rl/algorithms/config.py`, `plugins/training/rl/algorithms/factory.py`, GRPO/GSM8K YAML configs

## Motivation

We observed that adding entropy regularization to `adv≈0` samples can be brittle.

This alternative keeps the GRPO advantage normalization, but applies an asymmetric post-processing step:

- `adv_scaled = adv * pos_adv_scale` if `adv > 0`
- `adv_scaled = adv` if `adv <= 0`

This boosts gradients for “good” samples while leaving non-positive samples unchanged.

## How to enable (YAML)

Set the GRPO estimator kwarg:

```yaml
algo:
  estimator:
    name: grpo
    kwargs:
      pos_adv_scale: 4.0
```

- Default: `pos_adv_scale: 1.0` (baseline behavior)
- Constraint: must be `> 0`

## Implementation details

- Scaling is applied inside `GroupIdGRPOAdvantageModule` after GRPO per-group normalization and before optional clipping.
- This is an estimator-level knob so it is logged in W&B config (no env var overrides).

## TPU: Stop bs128 token job, replace with bs16 posadv×4 (commands used)

### 0) Identify the running job

On the TPU VM:

```bash
pgrep -af "projects/gsm8k_grpo/scripts/run_train.py" || true
```

### 1) Stop the token-level `bs=128` run

Avoid matching the `pkill` command itself by using a bracket pattern:

```bash
pkill -f "[p]rojects/gsm8k_grpo/scripts/run_train.py --config projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_batch128_roll8_steps400_evalfull50_adv0entropy0p01_ckptgcs_v6e8_testmonitor.yaml" || true
```

If it refuses to exit (careful: force kill):

```bash
pkill -9 -f "b[a]tch128_roll8_steps400_evalfull50_adv0entropy0p01_ckptgcs_v6e8_testmonitor" || true
```

### 2) Sync repo + start the bs16 run (W&B online)

From the workstation:

```bash
# Sync secrets (WANDB_API_KEY) to /root/.env on the TPU VM
bash scripts/sync_env_to_tpu_vm.sh \
  --name test-rl-grpo-baseline-v6e-8-260212192520 \
  --zone europe-west4-a \
  --project civil-rarity-482610-s5 \
  --worker all \
  --src .env

# Pull the latest branch and start the new config
bash scripts/ssh_tpu_vm_root.sh \
  --name test-rl-grpo-baseline-v6e-8-260212192520 \
  --zone europe-west4-a \
  --project civil-rarity-482610-s5 \
  --env-file /root/.env \
  --command 'set -euo pipefail; cd /root/MLLM-JAX; git pull --ff-only origin test-rl; bash scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh --env-name mllm-jax --config projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_batch16_roll8_steps400_evalfull50_posadvscale4_ckptgcs_v6e8_testmonitor.yaml'
```

### 3) Monitor

```bash
tail -n 50 /root/MLLM-JAX/logs/nohup_grpo_gsm8k_qwen25_3b_batch16_roll8_steps400_evalfull50_posadvscale4_ckptgcs_v6e8_testmonitor_latest.log
cat /root/MLLM-JAX/logs/nohup_grpo_gsm8k_qwen25_3b_batch16_roll8_steps400_evalfull50_posadvscale4_ckptgcs_v6e8_testmonitor_latest.exit
```

## References

- Config (v6e-8): `projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_batch16_roll8_steps400_evalfull50_posadvscale4_ckptgcs_v6e8_testmonitor.yaml`
- Advantage module: `plugins/training/rl/advantage/modules.py`

