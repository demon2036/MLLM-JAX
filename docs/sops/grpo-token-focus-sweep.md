# SOP: GRPO token-focus hyperparam sweep (p, max_tokens_per_sequence)

- **Title**: SOP: Run token-focus GRPO sweep on GSM8K (TPU v6e-8, W&B online)
  **Goal**: Scan `(prob_threshold=p, max_tokens_per_sequence=K)` while keeping the workload fixed, and compare runs by `eval/accuracy/pass_at_1`.
  **Scope**: `projects/gsm8k_grpo` runner (`projects/gsm8k_grpo/jax/train.py`) with token-focus update mask enabled.

## Sweep definition

- Configs live under `projects/gsm8k_grpo/configs/token_focus_sweep/`.
- Sweep script (runs configs sequentially): `scripts/run_grpo_gsm8k_token_focus_sweep_v6e8.sh`.
- TPU nohup launcher: `scripts/tpu_vm_start_grpo_gsm8k_token_focus_sweep_nohup.sh`.
- W&B project: `mllm-jax-grpo-gsm8k-tokenfocus-sweep` (set in each YAML).
- Default ranking metric: `eval/accuracy/pass_at_1` (periodic eval metric logged every `eval_every_steps`).

## Prereqs

- A running TPU VM (v6e-8 recommended).
- Repo synced via Git (`git pull` on TPU VM).
- `/root/.env` on TPU VM contains `WANDB_API_KEY=...` and YAML uses `wandb_mode: online`.

## Steps (TPU VM)

### 1) Start sweep via nohup

From repo root on TPU VM:

```bash
bash scripts/tpu_vm_start_grpo_gsm8k_token_focus_sweep_nohup.sh
tail -n 200 logs/nohup_grpo_gsm8k_token_focus_sweep_v6e8_latest.log
```

Expected:
- The log prints each config path before running it.
- Each run prints a W&B run URL under project `mllm-jax-grpo-gsm8k-tokenfocus-sweep`.

### 2) Check exit code

```bash
cat logs/nohup_grpo_gsm8k_token_focus_sweep_v6e8_latest.exit
```

Expected: `0`.

### 3) Rank runs (by eval accuracy)

```bash
python -u scripts/wandb_rank_grpo_token_focus_sweep.py \
  --entity johntitordemon2036 \
  --project mllm-jax-grpo-gsm8k-tokenfocus-sweep \
  --metric eval/accuracy/pass_at_1
```

Expected:
- A tab-separated ranking table with `(p, k)` and the selected metric.

## Notes

- Use `token_focus/selected_fraction` + `token_focus/eligible_fraction` to understand how aggressive each `(p, k)` point is.
- For a “final answer” config, re-run the best `(p, k)` point with a longer `steps` budget (new YAML) and compare again.

