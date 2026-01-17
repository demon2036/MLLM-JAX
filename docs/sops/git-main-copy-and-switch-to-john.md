# Git Branching: backup `main` and work on `john`

- **Title**: SOP: Create a safety branch from `main` (`main--copy`) and switch to `john`
  **Prereqs**: `git` installed; repo cloned at `/home/john/github/MLLM-JAX`
  **Environment (verified)**: Ubuntu Linux; Python `3.12.2`

## Goal

- Keep a backup pointer to the current `main` tip (`main--copy`).
- Switch to the `john` branch as the working base (per project workflow).

## Steps (commands actually used)

- `cd /home/john/github/MLLM-JAX`
- Inspect current state:
  - `git status -sb`
  - `git branch -avv`
  - `git rev-parse HEAD`
- Create the backup branch at the current `main` HEAD:
  - `git branch main--copy`
- If you have local changes that block switching branches, stash them (includes untracked):
  - `git stash push -u -m 'wip: local docs/answers changes before switching to john'`
- Create and switch to a local `john` tracking branch:
  - `git switch -c john origin/john`
- (Optional) Re-apply your stash after switching (may conflict depending on file divergence):
  - `git stash list`
  - `git stash apply stash@{0}`

## Expected Result

- `git branch -avv` shows `main--copy` at the same commit as `main` and `john` tracking `origin/john`.

