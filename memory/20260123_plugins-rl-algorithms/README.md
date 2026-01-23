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

- `git rev-parse --short HEAD`: `c12631c`
- `git status -sb`: dirty working tree (pre-existing docs/memory changes)

## Repo state used for TPU run

- Local commit (pushed): `f8e7cd0` (`feat: add rl algorithm registry`)
- TPU checkout: `f8e7cd0` (detached HEAD on `/root/MLLM-JAX`)

## Local validation

- Command (exit 0): `python -m pytest -q`
- Output (summary): `19 passed in 0.39s`

## TPU validation (v4-8, W&B online, 100 steps)

- TPU VM: `mllm-jax-v4-8-260122100610` (READY/HEALTHY at investigation time)
- Config: `plugins/training/configs/rl_gsm8k_qwen25_3b_v4_8_reinforcepp_steps100.yaml`
- Remote log symlink: `/root/MLLM-JAX/logs/rl_gsm8k_qwen25_3b_v4_8_reinforcepp_steps100_latest.log`
- Remote log file: `/root/MLLM-JAX/logs/rl_gsm8k_qwen25_3b_v4_8_reinforcepp_steps100_f8e7cd0_20260123_162452.log`
- W&B run URL: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/biv1ho6f`
- Runner printed: `algo=reinforce++`
- Runner printed: `step=99 ...` (100 steps)
- Wrapper printed: `exit_status=0`
- Log grep: `traceback_found=0`

## SOP updated

- `docs/sops/tpu-vm-v4-8-rl-gsm8k-reinforcepp-wandb-100steps.md`
