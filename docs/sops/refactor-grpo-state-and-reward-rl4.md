# SOP: Refactor GRPO state + reward plumbing (keep RL 4-phase interfaces)

- **Title**: SOP: Refactor GRPO state + reward plumbing (keep RL 4-phase interfaces)
  **Prereqs**: Repo checkout; Python `3.12.x`; `pytest`
  **Environment (verified)**: Ubuntu Linux; repo `/home/john/workdir/mllm-jax-minionerec` (2026-01-25)

## Goal

- Keep RL 4-phase interfaces stable (`plugins/training/api/interfaces.py`).
- Remove `plugins/training/runner/grpo_gsm8k.py` dependency on legacy `training2.get_state`.
- Move GSM8K reward functions into `plugins/` (runner no longer imports `training2`).
- Consolidate duplicated “checkpoint params → sharded params” placement logic.

## Steps (commands actually used)

### 1) Create a new working branch

- `git checkout -b refactor-rl4-platform`

### 2) Run local tests

- `python -m pytest -q`

## Expected Result

- `pytest` exits with code `0`.
- `plugins/training/runner/grpo_gsm8k.py` no longer imports `training2.get_state` or GSM8K reward fns from `training2.py`.
- Shared param placement helpers are used (no duplicated `_place_params` bodies across runners).

