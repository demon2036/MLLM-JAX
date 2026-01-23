# SOP: Make GRPO runner use 4-phase modules (rollout/reward/adv/update)

- **Title**: SOP: Refactor `plugins/training/runner/grpo_gsm8k.py` to use the 4-phase training module interfaces
  **Prereqs**: Windows PowerShell; Python; `git`; repo checkout
  **Environment (verified)**:
  - OS: Windows `11` (`Windows-11-10.0.22631-SP0`)
  - Python: `3.13.9` (Anaconda)

## Steps (commands actually used in this repo)

From repo root:

### 1) Verify the runner still has direct phase calls

- `Select-String -Path plugins/training/runner/grpo_gsm8k.py -Pattern "compute_weighted_rewards|compute_grpo_advantages_by_group_id|ppo_update" -AllMatches`

### 2) Implement the rollout backend adapter module

- Add `plugins/training/rollout/modules.py` (`RolloutBackendModule`)
- Keep `plugins/training/modules/__init__.py` as a compatibility re-export (optional)

### 3) Refactor the active GRPO runner to call 4 modules

- Update `plugins/training/runner/grpo_gsm8k.py` to use:
  - `RolloutBackendModule` (rollout; train+eval)
  - `WeightedRewardModule` (reward; train+eval)
  - `GroupIdGRPOAdvantageModule` (advantages; train)
  - `PPOUpdateModule` (update; train)
  - Phase module locations:
    - rollout: `plugins/training/rollout/modules.py`
    - reward: `plugins/training/reward/modules.py`
    - advantage: `plugins/training/advantage/modules.py`
    - update: `plugins/training/update/modules.py`

### 4) Deprecate legacy jit8 code (no edits, move only)

- `git mv plugins/jit8_train deprecated/jit8_train`
- `git mv test_jit8.py deprecated/test_jit8.py`

### 5) Remove active tests that import deprecated jit8 CLI

- Update `tests/test_jit8_schema_and_cli.py` to drop the jit8 CLI import and keep only schema validation tests.

### 6) Run tests

- `python -m pytest -q`

## Expected Result

- The active runner’s training loop is structurally `rollout → reward → advantage → update` via module interfaces.
- `python -m pytest -q` exits `0` (in this run: `14 passed`).

## References

- `plugins/training/api/interfaces.py`
- `plugins/training/rollout/modules.py`
- `plugins/training/reward/modules.py`
- `plugins/training/advantage/modules.py`
- `plugins/training/update/modules.py`
- `plugins/training/runner/grpo_gsm8k.py`
- `memory/20260123_rl-four-phase-interfaces/README.md`
