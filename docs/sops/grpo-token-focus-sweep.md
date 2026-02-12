# SOP: GRPO token-focus hyperparam sweep (p, max_tokens_per_sequence)

- **Title**: SOP: Run token-focus GRPO sweep on GSM8K (TPU v6e-8, W&B online)
  **Goal**: Scan `(prob_threshold=p, max_tokens_per_sequence=K)` while keeping the workload fixed, and compare runs by `eval/accuracy/pass_at_1`.
  **Scope**: `projects/gsm8k_grpo` runner (`projects/gsm8k_grpo/jax/train.py`) with token-focus update mask enabled.

## Sweep definition

This repo keeps two sweep variants:

### A) Quick sweep (short runs)

- Configs: `projects/gsm8k_grpo/configs/token_focus_sweep/`
- Runner (sequential): `scripts/run_grpo_gsm8k_token_focus_sweep_v6e8.sh`
- TPU nohup launcher: `scripts/tpu_vm_start_grpo_gsm8k_token_focus_sweep_nohup.sh`
- W&B project: `mllm-jax-grpo-gsm8k-tokenfocus-sweep` (set in each YAML)
- Ranking metric: `eval/accuracy/pass_at_1` (periodic eval metric logged every `eval_every_steps`)

### B) Long sweep (delivery-grade compare)

- Configs: `projects/gsm8k_grpo/configs/token_focus_sweep_steps100/`
- Runner (sequential): `scripts/run_grpo_gsm8k_token_focus_sweep_steps100_v6e8.sh`
- TPU nohup launcher: `scripts/tpu_vm_start_grpo_gsm8k_token_focus_sweep_steps100_nohup.sh`
- W&B project: `mllm-jax-grpo-gsm8k-tokenfocus-sweep-steps100` (set in each YAML)
- Ranking metric: `eval/accuracy/pass_at_1` (mirrored from `eval_full/*` when `eval_every_steps: 0`)
- Eval behavior:
  - `steps: 100`
  - `eval_full_every_steps: 50` → full test split eval at steps `49, 99`

## Prereqs

- A running TPU VM (v6e-8 recommended).
- Repo synced via Git (`git pull` on TPU VM).
- `/root/.env` on TPU VM contains `WANDB_API_KEY=...` and YAML uses `wandb_mode: online`.

## Steps (TPU VM)

### 1) Start sweep via nohup

From repo root on TPU VM:

```bash
## Quick sweep:
bash scripts/tpu_vm_start_grpo_gsm8k_token_focus_sweep_nohup.sh
tail -n 200 logs/nohup_grpo_gsm8k_token_focus_sweep_v6e8_latest.log

## Long sweep (steps=100, eval_full_every_steps=50):
bash scripts/tpu_vm_start_grpo_gsm8k_token_focus_sweep_steps100_nohup.sh
tail -n 200 logs/nohup_grpo_gsm8k_token_focus_sweep_steps100_v6e8_latest.log
```

Expected:
- The log prints each config path before running it.
- Each run prints a W&B run URL under the YAML-selected project.

### 2) Check exit code

```bash
## Quick sweep:
cat logs/nohup_grpo_gsm8k_token_focus_sweep_v6e8_latest.exit

## Long sweep:
cat logs/nohup_grpo_gsm8k_token_focus_sweep_steps100_v6e8_latest.exit
```

Expected: `0`.

### 3) Rank runs (by eval accuracy)

```bash
python -u scripts/wandb_rank_grpo_token_focus_sweep.py \
  --entity johntitordemon2036 \
  --project mllm-jax-grpo-gsm8k-tokenfocus-sweep-steps100 \
  --metric eval/accuracy/pass_at_1

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

### Resume behavior (spot TPU / preemption)

- `scripts/run_grpo_gsm8k_token_focus_sweep_steps100_v6e8.sh` queries W&B for finished runs and skips their `config_path` values, so re-launching the sweep continues from the next unfinished config.
- Requirement: `WANDB_API_KEY` must be available in the shell env (recommended: `set -a; source /root/.env; set +a`).
