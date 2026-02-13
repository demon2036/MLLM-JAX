# Audit GRPO sequence-level policy loss (`loss_level=sequence`) on `test-rl`

- **Title**: SOP: Audit the `test-rl` branch GRPO “sequence-level loss” implementation (what it does / does not do)
- **Prereqs**: Repo checkout; local Linux shell with Python available (JAX not required for this audit)
- **Environment (verified)**:
  - OS: Linux 6.17.0-8-generic (Ubuntu)
  - Python: 3.13.7
  - Branch: `test-rl`

## Steps (commands actually used)

- Run local unit tests to ensure repo is in a sane state:
  - `pytest -q`
- Print the merged config for the target YAML (verify `loss_level` wiring + `wandb_mode`):
  - `python3 projects/gsm8k_grpo/scripts/run_train.py --print-config --config projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_batch16_roll8_steps400_evalfull50_seqloss_ckptgcs_v6e8.yaml`

## Expected Result

- `pytest -q` completes without failures (this repo skips JAX-dependent tests if JAX is not installed).
- The printed config contains:
  - `algo.update.kwargs.loss_level: sequence`
  - `wandb_mode: online`

## Audit findings (what “sequence-level loss” means in this branch)

### Audited execution path (entrypoint -> runner -> rollout -> labels mask -> TrainGRPOModule loss)

1) Entrypoint loads YAML and calls the runner:
   - `projects/gsm8k_grpo/scripts/run_train.py` → `run_grpo_gsm8k(cfg)`
2) Runner reads `algo.update.kwargs.loss_level` and passes it into the training module construction:
   - `projects/gsm8k_grpo/jax/train.py` reads `pg_loss_level = update_kwargs["loss_level"]`
   - `training2.get_state(..., pg_loss_level=pg_loss_level)` → constructs `TrainGRPOModule(loss_level=pg_loss_level)`
3) Rollout produces a completion-only labels mask:
   - Default backend `naive`: `plugins/training/rl/rollout/backends/naive_sampler.py`
   - Uses `plugins/sample/workflows/grpo_sync.py::generate_answers_and_training_batch(...)`
   - Returns `batch["labels"] = train_completions_mask` where the mask is **1 on completion tokens** and 0 elsewhere.
4) Update loop runs PPO-style update steps (old logps caching):
   - `plugins/training/rl/update/ppo.py::ppo_update(...)`
5) Loss aggregation behavior is implemented inside:
   - `MLLM_JAX/train_modules/__init__.py::TrainGRPOModule.__call__()`

### Exact semantics: `loss_level: sequence`

This option changes only the final reduction of the (masked) per-token PPO loss:

- `loss_level: token` (default):
  - Sum masked per-token loss over the whole batch, then divide by `total_valid_token_count`.
- `loss_level: sequence`:
  - For each sequence, average over its completion tokens, then mean over sequences:
    - `per_seq_loss[b] = sum_t(per_token_loss[b,t] * mask[b,t]) / max(sum_t(mask[b,t]), 1)`
    - `loss_pg = mean_b(per_seq_loss[b])`

What it is **NOT**:
- It does **not** implement “sequence-level PPO” (sequence-level ratio/clipping). The PPO ratio and clipping remain **token-wise**; only the aggregation changes.

## Risks / edge cases (watch-outs)

- `rollout.dynamic_sampling.fallback_policy: drop_group` can zero out `labels` (and `attention_mask`) for dropped groups:
  - This can produce sequences with `sum(mask)==0`. In `loss_level: sequence`, these sequences contribute `per_seq_loss=0` but still count in the final mean over sequences.
- Entropy metrics can become NaN/Inf for short/empty completions:
  - `entropy` uses `avg_entropy_per_sample = sum(masked_entropy)/sum(mask)` with no denominator clamp.
  - `entropy_loss` uses a “tokens 4–100 (within completion)” window; if that window is empty, its denominator is 0.

## Troubleshooting

- `FileNotFoundError: ... seqloss ... yaml`: you are likely on the wrong branch/worktree; ensure you are on `test-rl` and the config exists under `projects/gsm8k_grpo/configs/`.
- If `pytest` output differs: ensure you are in the same checkout; JAX-dependent tests are expected to be skipped when `jax` is not installed.

## References

- `projects/gsm8k_grpo/scripts/run_train.py`
- `projects/gsm8k_grpo/jax/train.py`
- `training2.py`
- `plugins/training/rl/algorithms/config.py` (normalizes `algo.update.kwargs.loss_level` aliases)
- `plugins/training/rl/update/ppo.py`
- `plugins/training/rl/rollout/backends/naive_sampler.py`
- `plugins/sample/workflows/grpo_sync.py`
- `MLLM_JAX/train_modules/__init__.py`
