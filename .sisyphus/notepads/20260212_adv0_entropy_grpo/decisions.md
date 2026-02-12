## 2026-02-12T15:57:46Z Task: adv0-entropy-grpo
- Implement conditional entropy regularization inside `MLLM_JAX.train_modules.TrainGRPOModule` gated by `advantages==0` (or abs<=eps).
- Configure via YAML under `algo.update.kwargs` (e.g. `adv0_entropy_coef`, `adv0_entropy_eps`) and pass through `projects/gsm8k_grpo/jax/train.py -> training2.get_state`.
- Add W&B metrics in train loop for advantage sign buckets: adv==0, adv>0, adv<0 (counts + fractions).
- Treat `eval_full_sweep: true` + `eval_every_steps>0` as “run full-split eval every N steps” (not just once at end).
