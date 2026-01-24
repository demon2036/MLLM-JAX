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

## TPU validation (v4-8, W&B online, 100 steps, **bs128 default config values**, algo-only swap)

- User “跑通” requirement satisfied here:
  - Base hyperparams are exactly from `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml` (no 缩水).
  - Only difference is `algo.name` (verified by `diff -u ... | grep '^[+-][^+-]'`).
- Diff evidence (local):
  - `diff -u plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml plugins/training/configs/rl_gsm8k_qwen25_3b_bs128_steps100_reinforcepp.yaml | grep -E '^[+-][^+-]'`
  - Output:
    - `-  name: grpo`
    - `+  name: reinforce++`

### Run identity

- TPU VM: `mllm-jax-v4-8-260122100610` (`v4-8`, zone `us-central2-b`, project `civil-rarity-482610-s5`)
- TPU repo HEAD (run commit): `04f5097` (detached HEAD)
- Config used: `plugins/training/configs/rl_gsm8k_qwen25_3b_bs128_steps100_reinforcepp.yaml`
- Base config (same hyperparams): `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml`
- W&B run URL: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/84sik048`

### Logs + key output

- Remote log symlink: `/root/MLLM-JAX/logs/rl_gsm8k_qwen25_3b_bs128_steps100_reinforcepp_latest.log`
- Remote log file: `/root/MLLM-JAX/logs/rl_gsm8k_qwen25_3b_bs128_steps100_reinforcepp_04f5097_20260124_013914.log`
- Key log lines (grep outputs recorded):
  - `2:config_path: plugins/training/configs/rl_gsm8k_qwen25_3b_bs128_steps100_reinforcepp.yaml`
  - `62:algo=reinforce++`
  - `61:wandb: 🚀 View run at https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/84sik048`
  - Final steps:
    - `474:step=95 ... dt=41.76s`
    - `478:step=96 ... dt=46.63s`
    - `482:step=97 ... dt=44.16s`
    - `486:step=98 ... dt=70.01s`
    - `490:step=99 ... dt=43.97s`

### Clean-finish evidence

- No traceback in log:
  - `grep -n "Traceback" "$LOG" | wc -l` → `0`
- No remaining training processes (after completion):
  - `pgrep -af "[r]un_grpo_gsm8k_training.py"` → no output
- W&B backend state confirms normal completion (queried from TPU with WANDB_API_KEY loaded):
  - Output: `wandb_state finished`

## SOP updated

- `docs/sops/tpu-vm-v4-8-rl-gsm8k-reinforcepp-wandb-100steps.md`

## PPO vs REINFORCE 曲线一致性核查（20 steps, algorithm_test）

- Branch/commit: `algorithm` @ `1f6a42d`
- TPU VM: `mllm-jax-v4-8-260122100610` (zone `us-central2-b`, project `civil-rarity-482610-s5`)

### Reinforce 20-step run

- W&B run URL: `https://wandb.ai/johntitordemon2036/algorithm_test/runs/d5zv5dat`
- W&B run name: `rl_gsm8k_qwen25_3b_bs128_steps20_reinforce_algtest_1f6a42d_20260124_045430`
- Log (latest): `/root/MLLM-JAX/logs/nohup_rl_gsm8k_qwen25_3b_bs128_steps20_reinforce_algtest_latest.log`
- Log evidence:
  - `2:config_path: plugins/training/configs/rl_gsm8k_qwen25_3b_bs128_steps20_reinforce_algtest.yaml`
  - `62:algo=reinforce`
  - `146:step=19 loss=0.180230 entropy=0.4590 reward_mean=1.7090 dt=51.49s`

### PPO 20-step run

- W&B run URL: `https://wandb.ai/johntitordemon2036/algorithm_test/runs/pm1e4vof`
- W&B run name: `rl_gsm8k_qwen25_3b_bs128_steps20_ppo_algtest_1f6a42d_20260124_052244`
- Log (latest): `/root/MLLM-JAX/logs/nohup_rl_gsm8k_qwen25_3b_bs128_steps20_ppo_algtest_latest.log`
- Log evidence:
  - `2:config_path: plugins/training/configs/rl_gsm8k_qwen25_3b_bs128_steps20_ppo_algtest.yaml`
  - `62:algo=ppo`
  - `146:step=19 loss=0.180230 entropy=0.4590 reward_mean=1.7090 dt=51.53s`

### W&B status verification

- `wandb.Api()` states: `reinforce finished`, `ppo finished`

### Root-cause note (why curves can match)

- `plugins/training/algorithms/__init__.py` states the repo shares a single PPO-style update path.
- `plugins/training/algorithms/factory.py` maps both `ppo` and `reinforce` to `GlobalNormAdvantageModule`.
- `plugins/training/runner/grpo_gsm8k.py` uses `cfg.train.ppo_epochs` as `ppo_steps`.
- Base config keeps `ppo_epochs: 1`, so with identical seeds/inputs, `ppo` and `reinforce` can produce identical metrics when only `algo.name` changes.

### In-flight (started before interruption)

- RLOO 20-step run is currently active:
  - W&B run URL: `https://wandb.ai/johntitordemon2036/algorithm_test/runs/f2iek3u7`
  - Last observed line: `72:step=1 ... dt=76.13s`
  - Exit file not yet written (status unknown at note time)

### Local validation (this pass)

- `pytest -q` → `19 passed in 0.30s`
