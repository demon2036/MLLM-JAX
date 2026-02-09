# TPU VM v6e-8 GSM8K Qwen2.5-3B literal batch128 roll8 (1024 seq/step) validation

- **Title**: SOP: Validate literal `rollout.batch_size=128` + `rollout.n=8` configs for GSM8K GRPO/MaxRL on TPU v6e-8
  **Prereqs**: TPU VM `tpuv6e-8` reachable; conda env `mllm-jax`; repo synced on branch `max-rl`; W&B key set on TPU (`/root/.env`)
  **Environment (verified)**:
  - Date: 2026-02-09
  - TPU: `tpuv6e-8`, zone `us-east1-d`, project `civil-rarity-482610-s5`
  - Branch/commit: `max-rl` @ `5ba54fe`

## Goal

Run two literal-batch configs:
- `projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_batch128_roll8_literal_v6e8.yaml`
- `projects/gsm8k_grpo/configs/rl_gsm8k_qwen25_3b_batch128_roll8_literal_maxrl_v6e8.yaml`

and verify whether full eval sweep can complete on v6e-8.

## Steps (commands actually used)

### 1) Sync repo on TPU

```bash
cd /root/MLLM-JAX
git fetch --all --prune
git checkout max-rl
git reset --hard origin/max-rl
git rev-parse --short HEAD
```

### 2) Local parse check (from repo root)

```bash
python projects/gsm8k_grpo/scripts/run_train.py --print-config --config projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_batch128_roll8_literal_v6e8.yaml
python projects/gsm8k_grpo/scripts/run_train.py --print-config --config projects/gsm8k_grpo/configs/rl_gsm8k_qwen25_3b_batch128_roll8_literal_maxrl_v6e8.yaml
```

Expected in both outputs:
- `rollout.batch_size: 128`
- `rollout.n: 8`
- `sequences_global_per_step: 1024`

### 3) Launch GRPO literal run

```bash
cd /root/MLLM-JAX
export EVAL_FULL_SWEEP=1
export EVAL_FULL_NUM_PRE_Q=1
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false
export ROLLOUT_FAST_GENERATE=0
export ROLLOUT_FAST_QWEN2_DECODE_ATTENTION=0
bash scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh \
  --env-name mllm-jax \
  --config projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_batch128_roll8_literal_v6e8.yaml
```

Observed W&B run:
- `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/dpmorcsj`

### 4) Launch MaxRL literal run

```bash
cd /root/MLLM-JAX
export EVAL_FULL_SWEEP=1
export EVAL_FULL_NUM_PRE_Q=1
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false
export ROLLOUT_FAST_GENERATE=0
export ROLLOUT_FAST_QWEN2_DECODE_ATTENTION=0
bash scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh \
  --env-name mllm-jax \
  --config projects/gsm8k_grpo/configs/rl_gsm8k_qwen25_3b_batch128_roll8_literal_maxrl_v6e8.yaml
```

Observed W&B run:
- `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/d32xmnfa`

### 5) Verify result files

```bash
cd /root/MLLM-JAX
for tag in grpo_gsm8k_qwen25_3b_batch128_roll8_literal_v6e8 rl_gsm8k_qwen25_3b_batch128_roll8_literal_maxrl_v6e8; do
  LOG="logs/nohup_${tag}_latest.log"
  EXIT="logs/nohup_${tag}_latest.exit"
  echo "=== $tag ==="
  echo EXIT=$(cat "$EXIT")
  grep -c "Traceback" "$LOG"
  grep -n "RESOURCE_EXHAUSTED" "$LOG" | tail -n 1
  grep -n "^step=" "$LOG" | tail -n 1 || true
  grep -n "^eval_full " "$LOG" | tail -n 1 || true
done
```

## Expected Result

- On v6e-8 for this literal profile (`1024` seq/step), both runs fail before first step due memory limit.
- `exit=1`, `Traceback=1`, no `step=` and no `eval_full` line.

## Observed Result

- GRPO literal: `exit=1`, OOM (`RESOURCE_EXHAUSTED`), no `step=`.
- MaxRL literal: `exit=1`, OOM (`RESOURCE_EXHAUSTED`), no `step=`.
- Shared OOM message includes:
  - `Attempting to reserve 19.14G ... There are 5.64G free`

## Troubleshooting

- This is a per-host memory issue at rollout prefill for local sequence batch `1024`.
- Disabling fast rollout patches (`ROLLOUT_FAST_GENERATE=0`, `ROLLOUT_FAST_QWEN2_DECODE_ATTENTION=0`) did not resolve the OOM.
- Attempt to create `v6e-16` for lower per-host batch was blocked by quota (`TPUV6EPreemptible... exhausted`).
- Practical workaround in this project: use the non-literal profile `rollout.batch_size=16, n=8` (128 seq/step), which is already validated on v6e-8.

## References

- `projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_batch128_roll8_literal_v6e8.yaml`
- `projects/gsm8k_grpo/configs/rl_gsm8k_qwen25_3b_batch128_roll8_literal_maxrl_v6e8.yaml`
- `scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh`
- `docs/sops/grpo-gsm8k-runner-batch-size.md`
- `memory/20260209_tpu_grpo_maxrl_qwen25_3b/README.md`
