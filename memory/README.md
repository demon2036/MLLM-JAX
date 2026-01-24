# Memory Index

- 20260123_rollout-batch-size-semantics: Fix rollout.batch_size semantics comment + SOP alignment.
- 20260123_rl-four-phase-interfaces: Analyze Tunix/AReaL/VERL/MaxText RL interfaces; propose 4-phase contracts (rollout/reward/advantage/update).
- 20260123_tpuv6e-validate-optimizer: Validate 4-phase refactor on TPU v6e-8, then modularize optimizer config/passthrough.
- 20260123_default-config-and-launcher: Keep only one default YAML; simplify launch scripts to `script.sh --config ...`; make wandb mode config-driven; re-validate on TPU v6e-8.
- 20260123_v6e8-v6e16-grpo-multihost-speedup: Investigate v6e-8 vs v6e-16 GRPO speed gap; add multihost/mesh optimizations; validate on W&B.
- 20260123_clone-minionerec-to-workdir1: Clone this repo into `/home/john/workdir1/minionerec` as a separate working copy.
- 20260123_clone-akalikong-minionerec: Clone https://github.com/AkaliKong/MiniOneRec into `/home/john/workdir/minionerec/workdir/MiniOneRec` and verify the checkout.
- 20260123_sid-sft-plugins: Implement MiniOneRec SID SFT + HR/NDCG eval under `plugins/sft/`.
- 20260124_minionerec-tpu-sft: Re-implement MiniOneRec SFT+eval in JAX for TPU (no HF Trainer); validate HR/NDCG matches upstream `calc.py`.
