# SOP: Phase-folder layout for RL training code

- **Title**: SOP: Restructure training code into 4 phase folders (`rollout/`, `reward/`, `advantage/`, `update/`)
  **Prereqs**: Windows PowerShell; `git`; Python
  **Environment (verified)**:
  - OS: Windows `11` (`Windows-11-10.0.22631-SP0`)
  - Python: `3.13.9` (Anaconda)

## Goal

- Make the code layout match the 4-phase interface split:
  - rollout
  - reward
  - advantage estimate
  - update

## Steps (commands actually used)

From repo root:

### 1) Inventory imports of the old `grpo/` folder

- `git grep -n "plugins\\.training\\.grpo"`

### 2) Move phase utilities into phase folders and remove `grpo/`

- `git mv plugins/training/grpo/batching.py plugins/training/rollout/batching.py`
- `git mv plugins/training/grpo/sampling.py plugins/training/rollout/sampling.py`
- `git mv plugins/training/grpo/rewarding.py plugins/training/reward/weighted.py`
- `git mv plugins/training/grpo/advantages.py plugins/training/advantage/grpo.py`
- `git mv plugins/training/grpo/update.py plugins/training/update/ppo.py`
- `git mv plugins/training/grpo/train_step.py plugins/training/update/train_step.py`
- `git rm -f plugins/training/grpo/__init__.py`

### 3) Move rollout backends/optimizations under rollout

- `git mv plugins/training/rollout_backends plugins/training/rollout/backends`
- `git mv plugins/training/rollout_optimizations plugins/training/rollout/optimizations`

### 4) Compile changed modules (fast import check)

- `python -m py_compile plugins/training/config.py scripts/run_grpo_gsm8k_training.py plugins/training/runner/grpo_gsm8k.py plugins/training/rollout/backends/__init__.py plugins/training/rollout/backends/base.py plugins/training/rollout/backends/naive_sampler.py plugins/training/rollout/backends/factory.py tests/test_rollout_backend_factory.py`

### 5) Run tests

- `python -m pytest -q`

## Expected Result

- `plugins/training/grpo/` no longer exists.
- Phase code is under:
  - `plugins/training/rollout/`
  - `plugins/training/reward/`
  - `plugins/training/advantage/`
  - `plugins/training/update/`
- `python -m pytest -q` exits `0`.

## References

- `plugins/training/api/interfaces.py`
- `plugins/training/runner/grpo_gsm8k.py`
- `memory/20260123_rl-four-phase-interfaces/README.md`

