# Projects

This folder holds **project-scoped** code/configs/notes (e.g., an SFT project) that should stay separated from reusable “functional” components in `plugins/`.

## When to use `projects/`

- End-to-end runnable entrypoints (train/eval/infer) for a specific project.
- Project-specific YAML configs tracked by W&B (recommended: `projects/<project>/configs/`).
- Project notes, dashboards, and validation records.

## Suggested layout

```
projects/<project>/
  README.md
  configs/
  scripts/
```

## Relationship to `plugins/`

- Put reusable implementations (loss/optimizer/sampler/rollout backend/etc.) in `plugins/`.
- `projects/` selects and wires them via config; avoid hardcoding in the main runner.
