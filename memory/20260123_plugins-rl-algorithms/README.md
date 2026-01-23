# Task

- User request: implement multiple RL algorithms under `plugins/` (REINFORCE, PPO, GRPO, DAPO, RLOO, REINFORCE++) referencing known projects (AReaL/Tunix/VERL patterns), then run an end-to-end 100-step training on TPU v4-8 with W&B online so results are inspectable.
- Constraints:
  - Non-invasive: custom code under `plugins/` (avoid patching upstream `MLLM_JAX/`).
  - Config-driven: new YAML configs instead of env-var hyperparameter overrides.
  - Delivery gates: local tests must pass; TPU run must exit 0; no tracebacks; W&B online at least once.
  - TPU workflow: git commit/push locally → TPU `git fetch`/`git checkout` (no code scp).

# Plan

1) Inspect the existing 4-phase runner (`rollout → reward → advantage → update`) and identify minimal extension points.
2) Add an algorithm registry/factory that composes phase modules, primarily swapping advantage estimators.
3) Implement advantage estimators:
   - `reinforce`: global baseline + whitening
   - `grpo`: group-id normalization (existing)
   - `rloo`: leave-one-out baseline per prompt-group
   - `dapo`: GRPO + global mix knob
   - `reinforce++`: RLOO baseline + whitening (+ optional clipping)
   - `ppo`: shipped as a config preset (multi-epoch update) paired with a chosen advantage estimator
4) Add unit tests for advantage math and algorithm factory wiring.
5) Add TPU v4-8 YAML configs for at least one 100-step W&B-online run (plus additional configs for other algorithms).
6) Run local `pytest` (must be green).
7) Push commits to GitHub and run the 100-step TPU job (W&B online), capturing:
   - TPU log file path
   - W&B run URL
   - commit SHA used
8) Update SOP(s) with the exact commands used and append evidence here.

# Completion criteria

- Code:
  - Algorithms are implemented under `plugins/` and selectable via YAML config.
  - Local `python -m pytest -q` exits `0`.
- TPU:
  - On a v4-8 TPU VM, a 100-step run completes with exit code `0`.
  - W&B run exists (online) and contains expected metrics + config (including algo selection).
- Documentation:
  - New/updated SOP for TPU v4-8 100-step run with exact commands used.
  - This memory file contains evidence (commands + exit codes + links).

# Evidence

## Repo state at start

- `git rev-parse --short HEAD`: (fill after changes)
- `git status -sb`: (fill after changes)

## Local validation

- `python -m pytest -q`: (pending)

## TPU validation (v4-8, W&B online, 100 steps)

- TPU VM: `mllm-jax-v4-8-260122100610` (READY/HEALTHY at investigation time)
- W&B run URL: (pending)
- Remote log path: (pending)
- Exit status: (pending)

