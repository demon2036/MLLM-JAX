# SOP: Deep-dive VERL ReMax implementation

- **Title**: SOP: Clone `volcengine/verl` and audit how it implements ReMax (baseline generation + KL shaping + advantage + update)
  **Prereqs**: Ubuntu Linux; `git`; outbound network access
  **Scope**: `workdir/verl` (external reference repo)

## Goal

- Clone VERL into repo-local `workdir/`.
- Locate ReMax wiring points (trainer loop) and core computations (KL penalty, advantage/return, policy update).
- Provide file/line anchors for quick comparison with this repo’s JAX ReMax.

## Steps (commands used)

### 1) Clone VERL into `workdir/`

```bash
GIT_TERMINAL_PROMPT=0 git ls-remote --heads https://github.com/volcengine/verl.git | head
git clone --depth 1 https://github.com/volcengine/verl.git workdir/verl
git -C workdir/verl rev-parse --short HEAD
git -C workdir/verl remote -v
```

Observed (this run): `HEAD=ec123e6`.

### 2) Inspect ReMax advantage estimator

```bash
sed -n '580,760p' workdir/verl/verl/trainer/ppo/core_algos.py
```

Look for:

- `AdvantageEstimator.REMAX = "remax"`
- `compute_remax_outcome_advantage(...)` (reverse-cumsum returns + baseline subtraction)

### 3) Inspect trainer wiring: greedy baseline generation + baseline reward extraction

```bash
sed -n '1280,1405p' workdir/verl/verl/trainer/ppo/ray_trainer.py
```

Look for:

- `if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:`
- `gen_baseline_batch.meta_info["do_sample"] = False`
- `reward_baselines = batch.batch["rm_scores"].sum(dim=-1)`

### 4) Inspect KL-in-reward shaping path

```bash
sed -n '1,260p' workdir/verl/verl/trainer/ppo/ray_trainer.py
sed -n '1410,1520p' workdir/verl/verl/trainer/ppo/ray_trainer.py
```

Look for:

- `apply_kl_penalty(...)` (computes token-level KL and sets `token_level_rewards = token_level_scores - beta * kld`)
- where `token_level_scores` is set from `rm_scores` and `token_level_rewards` is selected based on `algorithm.use_kl_in_reward`

### 5) Inspect how policy update consumes advantages

```bash
sed -n '502,720p' workdir/verl/verl/workers/actor/dp_actor.py
sed -n '1120,1285p' workdir/verl/verl/trainer/ppo/core_algos.py
```

Look for:

- `DataParallelPPOActor.update_policy()` selecting `advantages` and computing `log_prob`
- policy loss default `loss_mode=vanilla` → `compute_policy_loss_vanilla()` (PPO clip objective)

### 6) Check example ReMax launch config

```bash
sed -n '1,220p' workdir/verl/examples/remax_trainer/run_qwen2.5-3b_seq_balance.sh
```

## Expected result

- You can point to where VERL:
  - generates greedy baseline sequences (trainer loop)
  - stores baseline reward as `reward_baselines` and repeats it across samples
  - applies KL penalty inside `token_level_rewards` when `algorithm.use_kl_in_reward=true`
  - computes ReMax advantages/returns via `compute_remax_outcome_advantage`
  - performs policy updates via PPO infrastructure consuming `advantages`

## Notes (differences vs this repo’s JAX ReMax)

- VERL does **not** keep baseline sequences in the training batch; it only keeps the baseline scalar reward (`reward_baselines`).
- VERL’s ReMax estimator is implemented as a token-level advantage/return function in `core_algos.py` and then fed through the actor’s PPO-style policy loss.

## References

- VERL core algos: `workdir/verl/verl/trainer/ppo/core_algos.py`
- VERL trainer loop: `workdir/verl/verl/trainer/ppo/ray_trainer.py`
- VERL actor update: `workdir/verl/verl/workers/actor/dp_actor.py`
- This repo’s JAX ReMax deep-dive: `docs/sops/remax-jax-implementation-deepdive.md`
