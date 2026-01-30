# SOP: Make optimizer configurable (train.optimizer -> tx passthrough)

- NOTE (2026-01-30): Paths referenced in this SOP were updated after removing legacy shims and
  extracting GSM8K/GRPO into `projects/gsm8k_grpo/`.

- **Title**: SOP: Make the GRPO/GSM8K runner accept a pluggable optimizer (`train.optimizer`) instead of a hardcoded Optax `tx`
  **Prereqs**: Windows PowerShell; Python; repo checkout
  **Environment (verified)**:
  - OS: Windows `11` (`Windows-11-10.0.22631-SP0`)
  - Python: `3.13.9` (Anaconda)

## Steps (commands actually used in this repo)

### 1) Add an Optax optimizer builder under the update phase

- Add `plugins/training/core/optim/optimizer.py`
  - `OptimizerConfig` + `LRScheduleConfig`
  - `build_tx(training_steps, cfg)` returns an Optax `GradientTransformation`

### 2) Allow state init to accept a passed-in optimizer

- Update `training2.get_state(..., tx=None)`:
  - If `tx` is provided, use it for `TrainState.create(..., tx=tx)`
  - Otherwise keep the existing default schedule + `optax.lion(...)`

### 3) Wire the runner + CLI config to pass the optimizer through

- Update `plugins/training/rl/config.py` to include default `train.optimizer` (so configs have a documented schema).
- Update `projects/gsm8k_grpo/scripts/run_train.py` to parse `train.optimizer` into an `OptimizerConfig`.
- Update `projects/gsm8k_grpo/jax/train.py` to:
  - build `tx = build_tx(training_steps=cfg.steps, cfg=cfg.train.optimizer)`
  - call `get_state(..., tx=tx)`

### 4) Run tests

- `python -m pytest -q`

## Expected Result

- Optimizer is configurable via YAML (`train.optimizer`) without changing runner code.
- Default behavior matches the previous hardcoded optimizer when `train.optimizer` is omitted.
- `python -m pytest -q` exits `0` (in this run: `14 passed`).

## References

- `plugins/training/core/optim/optimizer.py`
- `training2.py` (`get_state`)
- `projects/gsm8k_grpo/jax/train.py`
- `projects/gsm8k_grpo/scripts/run_train.py`
- `plugins/training/rl/config.py`
