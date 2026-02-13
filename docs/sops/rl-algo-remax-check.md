# Check whether ReMax RL algorithm is implemented

- **Title**: SOP: Verify if this repo implements the `ReMax` RL algorithm (vs `maxrl`)
  **Prereqs**: None
  **Scope**: `plugins/training/rl/algorithms/config.py`, `plugins/training/rl/algorithms/factory.py`

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

- `rg` finds **no** occurrences of `remax`.
- `SUPPORTED_ALGOS` does **not** include `remax`.

## Current status (as of these commands)

- Implemented algorithm identifiers include: `grpo`, `reinforce`, `reinforce++`, `rloo`, `dapo`, `ppo`, `maxrl`.
- There is **no** `remax` implementation or config alias in this repo.

## Notes

If the intent was “ReMax” but you meant **MaxRL**, the config name in this repo is:

- `algo.name: maxrl` (and/or `algo.estimator.name: maxrl`)

## References

- `plugins/training/rl/algorithms/factory.py`
- `plugins/training/rl/algorithms/config.py`

