# SOP Index

## Quick tasks (task-based)

- Repo setup: `docs/sops/repo-setup.md`
- Network checks: `docs/sops/network-checks.md`
- Update repo via `git pull`: `docs/sops/git-pull-update.md`
- Commit and push to GitHub: `docs/sops/github-push.md`
- Reinstall Electerm (AppImage): `docs/sops/electerm-reinstall-appimage.md`
- Slim repo (move unused code to `deprecated/`): `docs/sops/repo-slim-deprecated.md`
- TPU alive check: `docs/sops/tpu-alive-check.md`
- TPU repo sync via Git (no SCP): `docs/sops/tpu-vm-repo-sync.md`
- TPU smoke GRPO train (3 steps): `docs/sops/tpu-grpo-gsm8k-qwen25-7b-3steps.md`

## Browse by area (component-based)

### Git

- `docs/sops/git-pull-update.md`
- `docs/sops/git-worktrees.md`
- `docs/sops/github-push.md`
- `docs/sops/repo-slim-deprecated.md`

### TPU

- `docs/sops/tpu-alive-check.md`
- `docs/sops/tpu-vm-create-v4-8-or-v6e-8.md`
- `docs/sops/tpu-grpo-gsm8k-qwen25-7b-3steps.md`
- `docs/sops/tpu-vm-bootstrap.md`
- `docs/sops/tpu-vm-delete-all.md`
- `docs/sops/tpu-vm-lifecycle.md`
- `docs/sops/tpu-vm-repo-sync.md`
- `docs/sops/tpu-vm-runtime.md`

### Docs

- `docs/sops/docs-maintenance.md`

### Workstation

- `docs/sops/electerm-reinstall-appimage.md`

### Metadata

- `docs/sops/codex-juice.md`
- `docs/sops/prompt-detailed-plans.md`

## Search recipes (grep-first)

- List SOP files: `find docs/sops -maxdepth 1 -type f -name '*.md' -print | sort`
- Find SOP titles: `rg -n '^- \\*\\*Title\\*\\*:' docs/sops`
- Find "juice": `rg -n '\\bjuice\\b' docs/sops`
