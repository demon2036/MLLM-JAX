# SOP Index

## Quick tasks (task-based)

- Repo setup: `docs/sops/repo-setup.md`
- Network checks: `docs/sops/network-checks.md`
- Update repo via `git pull`: `docs/sops/git-pull-update.md`
- Commit and push to GitHub: `docs/sops/github-push.md`
- TPU alive check: `docs/sops/tpu-alive-check.md`

## Browse by area (component-based)

### Git

- `docs/sops/git-pull-update.md`
- `docs/sops/git-worktrees.md`
- `docs/sops/github-push.md`

### TPU

- `docs/sops/tpu-alive-check.md`
- `docs/sops/tpu-vm-bootstrap.md`
- `docs/sops/tpu-vm-delete-all.md`
- `docs/sops/tpu-vm-lifecycle.md`
- `docs/sops/tpu-vm-runtime.md`

### Docs

- `docs/sops/docs-maintenance.md`

### Metadata

- `docs/sops/codex-juice.md`

## Search recipes (grep-first)

- List SOP files: `find docs/sops -maxdepth 1 -type f -name '*.md' -print | sort`
- Find SOP titles: `rg -n '^- \\*\\*Title\\*\\*:' docs/sops`
- Find "juice": `rg -n '\\bjuice\\b' docs/sops`

