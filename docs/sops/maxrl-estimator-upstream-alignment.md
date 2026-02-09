# SOP: Align MaxRL advantage estimator with upstream `tajwarfahim/maxrl`

- **Title**: SOP: Compare this repo’s MaxRL estimator math against upstream `tajwarfahim/maxrl` and patch mismatches
  **Prereqs**: Ubuntu Linux; `git`; outbound network access; this repo on a writable branch
  **Environment (verified)**:
  - Date: 2026-02-09
  - Repo: `/home/john/workdir/wt-max-rl` (branch `max-rl`)
  - Upstream ref clone path (gitignored): `workdir/maxrl_official`
  - Local tests: `pytest`

## Goal

- Clone upstream `https://github.com/tajwarfahim/maxrl`.
- Compare the **MaxRL advantage estimate** implementation against this repo.
- Apply minimal changes so this repo matches upstream behavior.

## Steps (commands actually used)

### 1) Clone upstream into `workdir/` (gitignored)

From repo root:

```bash
cd /home/john/workdir/wt-max-rl
git clone --depth 1 https://github.com/tajwarfahim/maxrl workdir/maxrl_official
git -C workdir/maxrl_official rev-parse --short HEAD
```

Observed (this run):

- Upstream HEAD: `91118ea`

### 2) Locate upstream estimator

File:

- `workdir/maxrl_official/verl/trainer/ppo/core_algos.py`

Function:

- `compute_maxrl_outcome_advantage(...)` (decorated with `@register_adv_est(AdvantageEstimator.MAXRL)`)

Core math (per prompt-group mean baseline):

- `adv = (score - mean) / (mean + epsilon)`
- special-case: if group size is `1`, upstream sets `mean=0`

### 3) Locate this repo estimator

File:

- `plugins/training/rl/advantage/estimators.py`

Function:

- `compute_maxrl_advantages_by_group_id(...)`

### 4) Patch mismatch(es)

Mismatch (before fix):

- Upstream uses denominator `(mean + eps)`
- This repo used `max(mean, eps)` (missing the `+eps` when `mean > eps`)

Fix applied:

- Make this repo use `(mean + eps)` and match upstream singleton-group behavior.

Commit (this repo):

- `98427f8` on branch `max-rl`

### 5) Run local unit checks

```bash
pytest -q tests/test_grpo_training_print_config_cli.py \
  tests/test_rl_algorithm_factory_maxrl.py \
  tests/test_advantage_estimators.py
```

Observed:

- `15 passed`

## Expected Result

- This repo’s MaxRL advantage estimator matches upstream math.
- Local tests pass.

## References

- Upstream: `https://github.com/tajwarfahim/maxrl`
- This repo: `plugins/training/rl/advantage/estimators.py`
- Memory: `memory/20260209_compare_estimate_maxrl_official/README.md`

