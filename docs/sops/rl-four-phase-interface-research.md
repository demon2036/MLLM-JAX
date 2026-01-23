# SOP: Research 4-phase RL interfaces (Tunix/AReaL/VERL/MaxText)

- **Title**: SOP: Clone and inspect Tunix/AReaL/VERL/MaxText to design rollout/reward/adv/update interfaces
  **Prereqs**: Windows PowerShell; `git`; outbound network access; repo checkout
  **Environment (verified)**:
  - OS: Windows `11` (`Windows-11-10.0.22631-SP0`)
  - Python: `3.13.9`
  - git: `2.52.0.windows.1`

## Steps (commands actually used in this repo)

### 1) Clone reference repos into `workdir/` (gitignored)

From repo root:

- `New-Item -ItemType Directory -Force workdir | Out-Null`
- `git clone --depth 1 https://github.com/google/tunix.git workdir/tunix`
- `git clone --depth 1 https://github.com/inclusionAI/AReaL.git workdir/areal`
- `git clone --depth 1 https://github.com/volcengine/verl.git workdir/verl`
- `git clone --depth 1 https://github.com/google/maxtext.git workdir/maxtext`

### 2) Record the inspected revisions

- `git -C workdir/tunix rev-parse --short HEAD; git -C workdir/tunix remote -v`
- `git -C workdir/areal rev-parse --short HEAD; git -C workdir/areal remote -v`
- `git -C workdir/verl rev-parse --short HEAD; git -C workdir/verl remote -v`
- `git -C workdir/maxtext rev-parse --short HEAD; git -C workdir/maxtext remote -v`

### 3) Inspect interface touchpoints (minimal file list)

- Tunix:
  - `Get-Content workdir/tunix/tunix/rl/rollout/base_rollout.py | Select-Object -First 260`
  - `Get-Content workdir/tunix/tunix/rl/rl_cluster.py | Select-Object -First 260`
  - `Get-Content workdir/tunix/tunix/rl/common.py | Select-Object -First 260`
  - `Get-Content workdir/tunix/tunix/rl/function_registry.py | Select-Object -First 260`
- AReaL:
  - `Get-Content workdir/areal/areal/api/workflow_api.py | Select-Object -First 260`
  - `Get-Content workdir/areal/areal/core/workflow_executor.py | Select-Object -First 260`
  - `Get-Content workdir/areal/areal/core/staleness_manager.py | Select-Object -First 260`
- VERL:
  - `Get-Content workdir/verl/verl/protocol.py | Select-Object -Skip 300 -First 220`
  - `Get-Content workdir/verl/verl/workers/rollout/base.py | Select-Object -First 260`
  - `Get-Content workdir/verl/verl/trainer/ppo/core_algos.py | Select-Object -First 260`
- MaxText:
  - `Get-Content workdir/maxtext/src/MaxText/rl/train_rl.py | Select-Object -Skip 340 -First 220`
  - `Get-Content workdir/maxtext/src/MaxText/integration/tunix/tunix_adapter.py | Select-Object -First 260`

### 4) Validate local repo tests still pass

Note: `pytest.ini` excludes `workdir/`, so these clones should not affect tests.

- `python -m pytest -q`

## Expected Result

- `workdir/` contains the cloned repos (not tracked by git).
- You can map each repo’s boundaries to a 4-phase RL pipeline design.
- `python -m pytest -q` exits `0`.

## References

- `answers/rl-four-phase-interface-proposal.md`
- `memory/20260123_rl-four-phase-interfaces/README.md`

