# TPU VM v6e-8 GRPO/GSM8K full test-set eval sweep (Qwen2.5-3B, W&B)

- **Title**: SOP: Run full GSM8K `test` split eval (1319 Qs, each once) on `v6e-8` without changing YAML
  **Prereqs**: TPU VM is `READY`; conda env `mllm-jax` exists; repo synced via Git; outbound internet for HF + W&B
  **Environment (verified)**:
  - TPU VM: `mllm-jax-v6e-8-260120021350` (`v6e-8`, 1 host), zone `us-east1-d`, project `civil-rarity-482610-s5`
  - Python: `3.12.12`
  - JAX/JAXLIB: `0.8.2` / `0.8.2` (backend `tpu`, `device_count=8`)
  - Branch: `improve-rollout`
  - Commit: `b8e232c` (full-sweep defaults to 1 completion per question; logs `eval_full/accuracy`)
  - W&B run: https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/75542270

## Goal

Compute stable, **full test-set** GSM8K accuracy (1319 questions) with **one completion per question** and log:
- `eval_full/accuracy`
- `eval_full/samples_per_question` (expected: `1`)

This avoids the lightweight eval estimate from `eval_batches=1` (16 questions).

## Steps (commands actually used)

### 0) Git-sync the repo on TPU (no SCP for code)

```bash
cd /root/MLLM-JAX
git fetch --all --prune
git checkout improve-rollout
git reset --hard origin/improve-rollout
git rev-parse --short HEAD
```

Expected: prints `b8e232c`.

### 1) Start a run with full-sweep eval enabled (full test split, each once)

```bash
cd /root/MLLM-JAX
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=online
export WANDB_PROJECT=mllm-jax-grpo-gsm8k
export WANDB_NAME=grpo_gsm8k_v6e8_fulltest_acc_<commit>_<ts>
export PRINT_TRAIN_TIME_BREAKDOWN=1
export ROLLOUT_FAST_GENERATE=1
export ROLLOUT_FAST_QWEN2_DECODE_ATTENTION=1
export EVAL_FULL_SWEEP=1
export STEPS=4  # verified (small run to validate eval); unset for full 100-step runs
bash scripts/tpu_vm_start_grpo_gsm8k_qwen25_3b_bs128_steps100_nohup.sh
```

Notes:
- YAML config stays unchanged: `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml`
- `EVAL_FULL_SWEEP=1` runs the full sweep once **after** training finishes.
- Full sweep time is dominated by decode; with `samples_per_question=1` it is much faster than the old `k=8` behavior.

### 2) Verify completion + full-sweep output

```bash
cd /root/MLLM-JAX
cat logs/nohup_grpo_gsm8k_qwen25_3b_bs128_steps100_latest.exit
grep -n "^eval_full " logs/nohup_grpo_gsm8k_qwen25_3b_bs128_steps100_latest.log
```

Expected:
- Exit file contains `0`.
- Log contains a line like:
  - `eval_full split=test questions=1319 samples_per_question=1 accuracy=... t=...s`

### 3) Pull key metrics from `wandb-summary.json`

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mllm-jax
cd /root/MLLM-JAX
python -m json.tool wandb/latest-run/files/wandb-summary.json | grep -F time/train/step_avg_last10_s
python -m json.tool wandb/latest-run/files/wandb-summary.json | grep -F eval_full/accuracy
python -m json.tool wandb/latest-run/files/wandb-summary.json | grep -F eval_full/samples_per_question
python -m json.tool wandb/latest-run/files/wandb-summary.json | grep -F time/eval_full/step_s
python -m json.tool wandb/latest-run/files/wandb-summary.json | grep -F time/eval_full/rollout_generate_s
```

## Expected Result (Run `75542270`)

- `eval_full/accuracy = 0.6990`
- `eval_full/samples_per_question = 1`
- `time/eval_full/step_s = 140.64` (~2.34 minutes)

## Troubleshooting

- Full sweep still takes minutes:
  - Check `time/eval_full/rollout_generate_s` dominates (expected).
- No `eval_full` line printed:
  - Ensure `EVAL_FULL_SWEEP=1` is exported in the shell that launches the job.
- W&B not logging:
  - Ensure `/root/.env` contains a valid `WANDB_API_KEY` (no whitespace).

## References

- `plugins/training/runner/grpo_gsm8k.py` (full sweep implementation + metrics)
- `answers/v6e-8-grpo-rollout-speedup-qwen25-3b.md` (Run G report)
