# Task
- Validate the current 4-phase RL refactor on a fresh `tpu-v6e-8` run and confirm it behaves the same as before (no errors; training loop runs end-to-end).
- Modularize the optimizer so it can be configured/passed in (instead of being hardcoded in `training2.get_state`).

# Plan
1) Create memory folder and log.
2) Run local pytest sanity check.
3) Commit and push four-phase changes.
4) Create v6e-8 TPU VM.
5) Bootstrap conda and git-sync.
6) Install TPU Python requirements.
7) Launch TPU smoke training job.
8) Monitor TPU log and exitcode.
9) Design optimizer config schema.
10) Implement optimizer builder module.
11) Wire optimizer into state init.
12) Run tests and update SOP.
13) Commit and push optimizer changes.
14) Run TPU smoke new commit.
15) Delete TPU VM after validation.

## Step 1 - Create memory folder and log
Completion criteria: `memory/20260123_tpuv6e-validate-optimizer/README.md` exists and `memory/README.md` is updated.
Evidence:
- Added `memory/20260123_tpuv6e-validate-optimizer/README.md`.
- Updated `memory/README.md` with new task entry.

## Step 2 - Run local pytest sanity check
Completion criteria: local test suite passes with exit code 0.
Evidence:
- Command (exit 0): `python -m pytest -q`
- Output: `14 passed in 0.85s`
