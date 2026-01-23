# Memory Index

- 20260123_rollout-batch-size-semantics: Fix rollout.batch_size semantics comment + SOP alignment.
- 20260123_rl-four-phase-interfaces: Analyze Tunix/AReaL/VERL/MaxText RL interfaces; propose 4-phase contracts (rollout/reward/advantage/update).
- 20260123_clone-workdir-repos: Clone Tunix/AReaL/VERL/MaxText/Slime into repo-local gitignored `workdir/`.
- 20260123_plugins-rl-algorithms: Implement multiple RL algorithms (REINFORCE/PPO/GRPO/DAPO/RLOO/REINFORCE++) under `plugins/`, validate on TPU v4-8 (100 steps, W&B).
- 20260123_tpuv6e-validate-optimizer: Validate 4-phase refactor on TPU v6e-8, then modularize optimizer config/passthrough.
- 20260123_default-config-and-launcher: Keep only one default YAML; simplify launch scripts to `script.sh --config ...`; make wandb mode config-driven; re-validate on TPU v6e-8.
