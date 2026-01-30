# SOP: TPU v6e-8 GRPO/GSM8K baseline vs EMA (W&B online + full test eval)

- **Title**: SOP: Run GRPO/GSM8K on v6e-8 (baseline vs EMA) with full test-set eval + W&B online
- **Prereqs**: TPU VM `READY`; conda env `mllm-jax`; valid `WANDB_API_KEY` in `/root/.env` (synced from local `.env`, gitignored); outbound internet (HF + W&B)
- **Environment (verified)**:
  - TPU VM: `mllm-jax-grpo-ema-v6e-8-260130171808` (`v6e-8`, 1 host), zone `europe-west4-a`, project `civil-rarity-482610-s5`
  - Repo: `https://github.com/demon2036/MLLM-JAX.git`, branch `john/sft-rl-ema-20260130`

## Goal

- Run GRPO training (Qwen2.5-3B-Instruct) for 100 steps with:
  - W&B online logging
  - **full GSM8K test split eval sweep** (1319 questions, 1 sample each)
- Compare baseline vs EMA-for-eval (enabled via YAML only).

## Steps (commands actually run)

### 0) Baseline (EMA disabled)

Run on TPU:

```bash
cd /root/MLLM-JAX
export EVAL_FULL_SWEEP=1
bash scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh \
  --env-name mllm-jax \
  --config projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml
```

Verify:

```bash
cd /root/MLLM-JAX
cat logs/nohup_grpo_gsm8k_qwen25_3b_bs128_steps100_latest.exit  # expect 0
grep -n '^eval_full ' logs/nohup_grpo_gsm8k_qwen25_3b_bs128_steps100_latest.log
```

Observed:
- Exit: `0`
- W&B run: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/bgq9m0dm
- `eval_full/accuracy`: `0.7831690674753601`

### 1) EMA-for-eval (EMA enabled)

Run on TPU:

```bash
cd /root/MLLM-JAX
export EVAL_FULL_SWEEP=1
bash scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh \
  --env-name mllm-jax \
  --config projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_bs128_steps100_ema.yaml
```

Observed:
- Exit: `0`
- W&B run: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/shqmhjbc
- `eval_full/accuracy`: `0.7005307050796058`

## Notes

- With only 100 steps, using `ema.decay=0.9998` can keep EMA very close to the initial weights.
  If EMA hurts metrics, consider smaller decay for short runs or bias-corrected EMA.

## References

- `projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml`
- `projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_bs128_steps100_ema.yaml`
- `scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh`
- `docs/sops/tpu-vm-v6e-8-grpo-gsm8k-full-test-eval-sweep.md`

