# SOP Index

## Quick tasks (task-based)

- Repo setup: `docs/sops/repo-setup.md`
- Network checks: `docs/sops/network-checks.md`
- Update repo via `git pull`: `docs/sops/git-pull-update.md`
- Inspect diffs via `git diff`: `docs/sops/git-diff-inspect.md`
- Commit and push to GitHub: `docs/sops/github-push.md`
- Backup `main`, replace it with `john`, and delete `join`: `docs/sops/git-main-copy-and-switch-to-john.md`
- Reinstall Electerm (AppImage): `docs/sops/electerm-reinstall-appimage.md`
- Slim repo (move unused code to `deprecated/`): `docs/sops/repo-slim-deprecated.md`
- TPU alive check: `docs/sops/tpu-alive-check.md`
- TPU repo sync via Git (no SCP): `docs/sops/tpu-vm-repo-sync.md`
- TPU v4-8 GRPO runner smoke (`rollout.backend=naive`): `docs/sops/tpu-vm-v4-8-grpo-gsm8k-rollout-backend-naive-smoke.md`
- TPU v6e-8 GRPO train (100 steps, bs=128 seq, W&B): `docs/sops/tpu-vm-v6e-8-grpo-gsm8k-bs128-steps100.md`
- TPU v6e-8 rollout speed debug (Qwen2.5-3B): `docs/sops/tpu-vm-v6e-8-grpo-gsm8k-rollout-speed-debug-3b.md`
- TPU v6e-8 full test-set eval sweep (Qwen2.5-3B, W&B): `docs/sops/tpu-vm-v6e-8-grpo-gsm8k-full-test-eval-sweep.md`
- TPU v6e-8 vs v6e-16 speed gap debug (multihost + mesh): `docs/sops/tpu-vm-v6e-8-v6e-16-grpo-gsm8k-speed-gap-debug.md`
- TPU v6e-16 rollout sharding style bench (legacy vs maxtext): `docs/sops/tpu-vm-v6e-16-grpo-gsm8k-rollout-sharding-style-bench.md`
- TPU bf16 attention-score precision (dot_general fp32 output): `docs/sops/tpu-bf16-attention-score-f32.md`
- JAX SPMD all-gather vs all-reduce (sharded matmul): `docs/sops/jax-spmd-allgather-vs-allreduce.md`
- TPU fsdp all-gather memory bench (Qwen2.5-3B proxy): `docs/sops/tpu-vm-fsdp-allgather-memory-bench-qwen25-3b.md`
- TPU v4-8 timing (len=1024, 20 steps, avg dt steps 10–19): `docs/sops/tpu-vm-v4-8-grpo-gsm8k-len1024-20steps-timing.md`
- TPU v4-8 timing (bs=32, len=1024, k=1, micro_batch=4, avg dt steps 10–19): `docs/sops/tpu-vm-v4-8-grpo-gsm8k-bs32-len1024-k1-mb4-20steps-timing.md`
- TPU v4-16 GRPO train (20 steps, W&B): `docs/sops/tpu-vm-v4-16-grpo-gsm8k-wandb-20steps.md`
- TPU v4-16 GRPO train (100 steps, W&B): `docs/sops/tpu-vm-v4-16-grpo-gsm8k-wandb-100steps.md`
- TPU v4-16 GRPO micro-batch smoke run (1 step): `docs/sops/tpu-vm-v4-16-grpo-gsm8k-microbatch-smoke.md`
- TPU v4-16 OOM sweep (len=2048, W&B): `docs/sops/tpu-vm-v4-16-grpo-gsm8k-wandb-oom-sweep-len2048.md`
- jit8 GRPO/GSM8K YAML config (deprecated): `docs/sops/grpo-gsm8k-jit8-yaml-config.md`
- GRPO runner batch size semantics: `docs/sops/grpo-gsm8k-runner-batch-size.md`
- GRPO runner metrics: `docs/sops/grpo-gsm8k-metrics.md`
- GRPO length + eval knobs: `docs/sops/grpo-gsm8k-length-and-eval.md`
- Training modularization plan (plugins-first): `docs/sops/training-modularization-plan.md`
- GRPO rollout backend abstraction (naive): `docs/sops/grpo-rollout-backend-abstraction-naive.md`
- AReaL RL organization notes (for modularizing training): `docs/sops/areal-rl-organization.md`
- AReaL logging system notes (StatsLogger/StatsTracker/PerfTracer): `docs/sops/areal-logging-system.md`
- 4-phase RL interface research (Tunix/AReaL/VERL/MaxText): `docs/sops/rl-four-phase-interface-research.md`
- MaxText sharding system notes: `docs/sops/maxtext-sharding-system.md`
- 4-phase RL interface implementation (GRPO runner modules): `docs/sops/rl-four-phase-interface-implementation.md`
- RL phase-folder layout (remove `grpo/`): `docs/sops/rl-phase-folder-layout.md`
- RL pluggable optimizer (`train.optimizer`): `docs/sops/rl-pluggable-optimizer.md`

## Browse by area (component-based)

### Git

- `docs/sops/git-pull-update.md`
- `docs/sops/git-diff-inspect.md`
- `docs/sops/git-main-copy-and-switch-to-john.md`
- `docs/sops/git-worktrees.md`
- `docs/sops/github-push.md`
- `docs/sops/repo-slim-deprecated.md`

### TPU

- `docs/sops/tpu-alive-check.md`
- `docs/sops/tpu-vm-create-v4-8-or-v6e-8.md`
- `docs/sops/tpu-vm-v4-8-grpo-gsm8k-rollout-backend-naive-smoke.md`
- `docs/sops/tpu-vm-v6e-8-grpo-gsm8k-bs128-steps100.md`
- `docs/sops/tpu-vm-v6e-8-grpo-gsm8k-rollout-speed-debug-3b.md`
- `docs/sops/tpu-vm-v6e-8-v6e-16-grpo-gsm8k-speed-gap-debug.md`
- `docs/sops/tpu-vm-v6e-16-grpo-gsm8k-rollout-sharding-style-bench.md`
- `docs/sops/tpu-bf16-attention-score-f32.md`
- `docs/sops/jax-spmd-allgather-vs-allreduce.md`
- `docs/sops/tpu-vm-fsdp-allgather-memory-bench-qwen25-3b.md`
- `docs/sops/tpu-vm-v4-8-grpo-gsm8k-len1024-20steps-timing.md`
- `docs/sops/tpu-vm-v4-8-grpo-gsm8k-bs32-len1024-k1-mb4-20steps-timing.md`
- `docs/sops/tpu-vm-bootstrap.md`
- `docs/sops/tpu-vm-delete-all.md`
- `docs/sops/tpu-vm-lifecycle.md`
- `docs/sops/tpu-vm-repo-sync.md`
- `docs/sops/tpu-vm-multihost-smoke-train.md`
- `docs/sops/tpu-vm-v4-16-grpo-gsm8k-wandb-20steps.md`
- `docs/sops/tpu-vm-v4-16-grpo-gsm8k-wandb-100steps.md`
- `docs/sops/tpu-vm-v4-16-grpo-gsm8k-microbatch-smoke.md`
- `docs/sops/tpu-vm-v4-16-grpo-gsm8k-wandb-oom-sweep-len2048.md`
- `docs/sops/tpu-vm-runtime.md`

### Training

- `docs/sops/training-modularization-plan.md`
- `docs/sops/grpo-rollout-backend-abstraction-naive.md`
- `docs/sops/areal-rl-organization.md`
- `docs/sops/areal-logging-system.md`
- `docs/sops/rl-four-phase-interface-research.md`
- `docs/sops/maxtext-sharding-system.md`
- `docs/sops/rl-four-phase-interface-implementation.md`
- `docs/sops/rl-phase-folder-layout.md`
- `docs/sops/rl-pluggable-optimizer.md`
- `docs/sops/grpo-gsm8k-jit8-yaml-config.md`
- `docs/sops/grpo-gsm8k-runner-batch-size.md`
- `docs/sops/grpo-gsm8k-metrics.md`
- `docs/sops/grpo-gsm8k-length-and-eval.md`
- `docs/sops/tpu-vm-v4-16-grpo-gsm8k-microbatch-smoke.md`
- `docs/sops/tpu-vm-v6e-8-grpo-gsm8k-rollout-speed-debug-3b.md`

### Docs

- `docs/sops/docs-maintenance.md`

### Workstation

- `docs/sops/electerm-reinstall-appimage.md`

### Metadata

- `docs/sops/codex-juice.md`
- `docs/sops/prompt-detailed-plans.md`

## Search recipes (grep-first)

- List SOP files: `find docs/sops -maxdepth 1 -type f -name '*.md' -print | sort`
- Find SOP titles: `rg -n '^- \\*\\*Title\\*\\*:' docs/sops`
- Find "juice": `rg -n '\\bjuice\\b' docs/sops`
