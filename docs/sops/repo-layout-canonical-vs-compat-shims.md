# Repo Layout: Canonical Tree (no shims)

- **Title**: SOP: Identify canonical module locations (compat shims removed)
  **Prereqs**: Repo checkout; basic Python package import knowledge
  **Environment (verified)**: Ubuntu Linux; repo state as of 2026-01-30

## Goal

This SOP prevents layout confusion by defining **canonical** module locations.

As of 2026-01-30, legacy compatibility shims (e.g. `plugins.common`, `plugins.training.api`,
`plugins.training.ppo`, and top-level `plugins.training.{rollout,reward,update,...}`) were removed.
Old import paths will fail and must be updated.

## Steps (commands actually used)

- Inspect the plugin directory layout:
  - `find plugins -maxdepth 3 -type d | sort`

- Find legacy references that should be migrated:
  - `rg -n "plugins\\.training\\.(api|ppo|advantage|reward|rollout|update|runner|algorithms|configs)|plugins\\.common" -S`

## Canonical tree (what to use)

```
plugins/
  api/                 # contracts only (Protocol/schema)
    training/
    sample/
  training/             # implementations only
    core/
    rl/
    sft/
  sample/               # generation/decoding/backends
projects/               # task/dataset-specific entrypoints + configs
```

## Expected result

- You can answer “which path is canonical?” quickly by checking the shim `__init__.py` docstring.
- New code uses only `plugins.api.*`, `plugins.training.{core,rl,sft}.*`, and `plugins.sample.*`.

## Troubleshooting

- If `find` is unavailable, use:
  - `ls -R plugins | sed -n '1,200p'`

## References

- `docs/sops/training-modularization-plan.md`
- `memory/20260130_repo-layout-tree-refactor-proposal/README.md`
