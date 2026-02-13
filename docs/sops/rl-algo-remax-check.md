# Check whether ReMax RL algorithm is implemented

- **Title**: SOP: Verify if this repo implements the `ReMax` RL algorithm (vs `maxrl`)
  **Prereqs**: None
  **Scope**: `plugins/training/rl/remax/`, `plugins/training/rl/algorithms/*`, `plugins/training/rl/rollout/backends/*`

## Goal

Answer the question: “Does this repo implement `remax`?”

## Steps (commands used)

Search for any mention of `remax`:

```bash
rg -n --ignore-case "remax" docs memory plugins projects tests | head -n 200
rg -n --ignore-case "remax" -S . | head -n 200
```

Inspect the RL algorithm registry / normalization:

```bash
rg -n "SUPPORTED_ALGOS|normalize_algo_name\(" plugins/training/rl/algorithms/config.py plugins/training/rl/algorithms/factory.py
sed -n '1,120p' plugins/training/rl/algorithms/factory.py
sed -n '1,120p' plugins/training/rl/algorithms/config.py
```

## Expected result

- `rg` finds occurrences of `remax` across `plugins/`, `projects/`, and `tests/`.
- `SUPPORTED_ALGOS` includes `remax`.
- There is a dedicated JAX implementation under `plugins/training/rl/remax/`.

## Current status (as of these commands)

- ✅ `remax` is implemented in this repo.
- Registry / config normalization:
  - Algo registry: `plugins/training/rl/algorithms/factory.py`
  - Normalization + cross-field constraints (`estimator==update==remax`): `plugins/training/rl/algorithms/config.py`
- Core algorithm components:
  - ReMax loss (policy gradient; configurable returns_style=official|verl): `plugins/training/rl/remax/module.py`
  - ReMax train state + ref params wiring: `plugins/training/rl/remax/state.py`
  - ReMax rollout backend (inject 1 greedy baseline per group, mask baseline labels): `plugins/training/rl/rollout/backends/remax_mixed_naive.py`
  - ReMax advantages (per-group greedy baseline subtraction): `plugins/training/rl/advantage/estimators.py` (`compute_remax_advantages_by_group_id`)
- Runner wiring + safety checks:
  - GSM8K runner selects `get_remax_state` when `algo.update.name=remax`: `projects/gsm8k_grpo/jax/train.py`
  - Config parser enforces required rollout backend / rollout.n / ppo_epochs / dynamic_sampling compatibility: `projects/gsm8k_grpo/scripts/run_train.py`
- Tests:
  - Factory wiring: `tests/test_rl_algorithm_factory_remax.py`
  - Loss semantics (official + verl returns_style): `tests/test_remax_policy_gradient_module.py`

## Notes

If the intent was “ReMax” but you meant **MaxRL**, the config name in this repo is:

- `algo.name: maxrl` (and/or `algo.estimator.name: maxrl`)

## References

- `plugins/training/rl/algorithms/factory.py`
- `plugins/training/rl/algorithms/config.py`
- `docs/sops/remax-jax-implementation-deepdive.md`
