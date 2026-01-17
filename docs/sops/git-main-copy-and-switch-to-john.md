# Git Branching: backup `main`, move it to `john`, and delete `join` (if present)

- **Title**: SOP: Create `main--copy`, fast-forward `main` to `john`, and delete `join`
  **Prereqs**: `git` installed; repo cloned at `/home/john/github/MLLM-JAX`; `origin` configured
  **Environment (verified)**: Ubuntu Linux; git `2.48.1`; Python `3.12.2`

## Goal

- Keep a backup pointer to the current `main` tip (`main--copy`).
- Replace `main` with the `john` branch tip (prefer fast-forward).
- Delete `join` (local + `origin/join`) if it exists.
- Push updated refs to GitHub.

## Steps (commands actually used)

- `cd /home/john/github/MLLM-JAX`
- Inspect current state:
  - `git status -sb`
  - `git branch -avv`
  - `git remote -v`
  - `git rev-parse main`
  - `git rev-parse john`
- If you have local changes that block switching branches, stash them (includes untracked):
  - `git stash push -u -m 'wip: local docs/answers changes before switching to john'`
- Ensure you have a local `john` branch (create it if needed):
  - `git switch -c john origin/john`
- Switch to `main` and create the backup pointer (do this BEFORE moving `main`):
  - `git switch main`
  - `git branch main--copy`
  - Verify:
    - `git rev-parse main`
    - `git rev-parse main--copy`
- Move `main` to the same commit as `john` (fast-forward only):
  - `git merge-base --is-ancestor main john`
  - `git merge --ff-only john`
- Delete `join` if present:
  - Local:
    - `if git show-ref --verify --quiet refs/heads/join; then git branch -D join; else echo "local join: not found"; fi`
  - Remote:
    - `if git ls-remote --exit-code --heads origin join >/dev/null 2>&1; then git push origin --delete join; else echo "remote origin/join: not found"; fi`
- Push updated branches:
  - `git push origin main--copy`
  - `git push origin main`
- Verify:
  - `git branch -avv`
  - `git ls-remote --heads origin main main--copy john`

## Expected Result

- `main` points to the same commit as `john`.
- `main--copy` preserves the previous `main`.
- No `join` branch locally or on `origin`.
