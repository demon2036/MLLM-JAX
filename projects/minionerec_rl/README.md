# MiniOneRec RL (JAX/TPU)

This project runs the **MiniOneRec recommendation-oriented RL** phase (GRPO-style)
on TPU, starting from a MiniOneRec **SID SFT** checkpoint.

The core goal is to reproduce the upstream `rl.py` pipeline in a JAX-friendly way:
- constrained SID beam search for rollouts
- rank-aware rewards (binary hit + rank penalties)
- GRPO-style advantage normalization within prompt groups
- optional KL penalty to a frozen reference policy (beta)

## Entry point

- `projects/minionerec_rl/scripts/run_train.py`

## Configs

Configs live under `projects/minionerec_rl/configs/` and are intended to be
tracked end-to-end in W&B.

