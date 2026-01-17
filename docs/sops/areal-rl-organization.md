# AReaL RL Organization Notes (for modularizing MLLM-JAX training)

- **Title**: SOP: Clone and inspect AReaL to learn how it modularizes asynchronous RL (and what we can borrow)
  **Prereqs**: Ubuntu Linux; `git`; outbound network access
  **Environment (verified)**: Repo `/home/john/github/MLLM-JAX`; Python `3.12.2`

## Goal

- Clone AReaL into an isolated, gitignored workdir.
- Identify AReaL’s modular RL boundaries (API ↔ backend ↔ workflow ↔ algo).
- Extract concrete design patterns we can borrow for MLLM-JAX (plugins-first, non-invasive).

## Steps (commands actually used in this repo)

### 1) Clone into repo-local `workdir/` (gitignored)

- `cd /home/john/github/MLLM-JAX`
- `mkdir -p workdir`
- `git clone --depth 1 https://github.com/inclusionAI/AReaL.git workdir/areal`
- Record the inspected revision:
  - `git -C workdir/areal rev-parse HEAD`
  - `git -C workdir/areal remote -v`

### 2) Identify the architectural layers (AReaL-lite design doc)

- `ls -la workdir/areal/areal | head -n 80`
- `sed -n '1,220p' workdir/areal/areal/README.md`
- (Optional) inspect deps/packaging:
  - `sed -n '1,220p' workdir/areal/pyproject.toml`

### 3) Inspect workflow contract + async rollout primitives

- Workflow contract:
  - `sed -n '1,260p' workdir/areal/areal/api/workflow_api.py`
- Concrete single-turn workflow:
  - `sed -n '1,260p' workdir/areal/areal/workflow/rlvr.py`
- Async rollout scheduler core (submit / wait / prepare_batch):
  - `sed -n '739,940p' workdir/areal/areal/core/workflow_executor.py`
  - `sed -n '1120,1320p' workdir/areal/areal/core/workflow_executor.py`
- Staleness manager (off-policy capacity control):
  - `sed -n '1,260p' workdir/areal/areal/core/staleness_manager.py`

### 4) Inspect training/inference engine contracts + PPO actor

- Engine API contracts:
  - `sed -n '1,260p' workdir/areal/areal/api/engine_api.py`
- PPO/GRPO actor (advantage + loss + update patterns):
  - `sed -n '1,260p' workdir/areal/areal/engine/ppo/actor.py`

## Expected Result

- You can point to:
  - Interfaces/contracts: `workdir/areal/areal/api/*`
  - Workflow implementation examples: `workdir/areal/areal/workflow/*`
  - Async rollout executor: `workdir/areal/areal/core/workflow_executor.py`
  - Staleness/capacity logic: `workdir/areal/areal/core/staleness_manager.py`
  - PPO/GRPO algorithm core: `workdir/areal/areal/engine/ppo/actor.py`

## References

- AReaL repo: https://github.com/inclusionAI/AReaL
- AReaL docs: https://inclusionai.github.io/AReaL/
