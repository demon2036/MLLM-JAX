# GRPO Pallas kernel: multi-device (shard_map) + TPU 100-step A/B train

## Goal

- Upgrade the existing logits-level GRPO Pallas kernel so it:
  - runs correctly on **single device** (baseline correctness),
  - runs correctly and efficiently on **multi-device** (TPU v6e-8) via `jax.experimental.shard_map`,
  - is faster and uses less peak memory than the pure-JAX baseline (`log_softmax`/`softmax` paths),
  - can be enabled/disabled via **YAML config** (W&B reproducible).
- Validate end-to-end on TPU `mllm-jax-v6e-8-spot-260124132428` with:
  - GRPO kernel gradcheck (W&B online, exit 0)
  - GRPO/GSM8K training 100 steps A/B (baseline vs kernel), W&B online

## Completion criteria

- TPU kernel gradcheck exits `0` with diffs within tolerances; W&B run URL recorded.
- TPU baseline 100-step run exits `0`; W&B run URL recorded.
- TPU kernel 100-step run exits `0`; W&B run URL recorded.
- Metrics are consistent between baseline and kernel runs (within expected noise), and kernel run shows improved:
  - `time/train/step_avg_last10_s` (lower is better)
  - peak memory (as available via logs/JAX profile/TPU metrics)
- SOPs updated with **exact commands actually run**.

## Work log (evidence)

### Reference repos (gitignored clones under `workdir/`)

- `workdir/jax` @ `a0aedf1` (`https://github.com/jax-ml/jax.git`)
- `workdir/maxtext` @ `b646a53` (`https://github.com/google/maxtext.git`)
- `workdir/liger-kernel` @ `9eb9a1e` (`https://github.com/linkedin/Liger-Kernel.git`)
- `workdir/unsloth` @ `4cb7229` (`https://github.com/unslothai/unsloth.git`)

### Notes to carry into implementation

- Current gradcheck scripts force `device0` because Mosaic kernels are not SPMD auto-partitionable.
- Target solution: wrap kernels with `jax.experimental.shard_map` (MaxText/Splash pattern) so multi-device works.
- Liger GRPO Triton kernel patterns to mirror:
  - streaming logsumexp over vocab tiles
  - unclipped-vs-clipped branch for gradients
  - optional KL term (beta)

### GRPO training call chain (repo)

- Entry: `scripts/run_grpo_gsm8k_training.py` → `plugins/training/runner/grpo_gsm8k.py:run_grpo_gsm8k()`
- State construction: `plugins/training/runner/grpo_gsm8k.py` imports `training2.get_state()`
- Baseline loss module: `training2.py:get_state()` hardcodes `MLLM_JAX.train_modules.TrainGRPOModule` (pure JAX ops)
- Update step: `plugins/training/update/train_step.py:training_step()` calls `state.apply_fn(...)`
- Implication: to A/B compare kernel vs baseline, we must add a **config-driven switch** in the state/module wiring:
  - baseline: `TrainGRPOModule` (existing behavior)
  - kernel: `plugins.training.grpo.TrainGRPOModulePallas` (multi-device shard_map wrapper around Pallas kernel)

### Commands / runs (to fill as executed)

- Local:
  - `python -m pytest -q`
- TPU:
  - `<git-sync command>`
  - `<env sync command>`
  - `<gradcheck command>`
  - `<baseline train command>`
  - `<kernel train command>`
