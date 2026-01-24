# Git Merge SOPs

- **Title**: SOP: Merge `origin/main` into the current branch (resolve conflicts + run tests)
  **Prereqs**: `git` + `python` available; you have an `origin` remote; prefer a clean working tree (or use `git stash`)
  **Environment (verified)**: Ubuntu kernel `6.14.0-37-generic`; git `2.48.1`; Python (venv) + pytest available
  **Steps**:
  - From repo root, verify state:
    - `git rev-parse --abbrev-ref HEAD`
    - `git status -sb`
  - Update remote refs:
    - `git fetch --all --prune`
  - If you have local edits, stash them:
    - `git stash push -u -m 'wip: before merging origin/main into <branch>'`
  - Merge main into your current branch:
    - `git merge origin/main`
  - If a `modify/delete` conflict happens on an ignored local asset (e.g. `memory/**`), pick the deletion side:
    - `git rm <conflict-path>`
    - `git diff --name-only --diff-filter=U`
  - Finalize the merge:
    - `git commit --no-edit`
  - Run the full local tests:
    - `python -m pytest -q`
  - Clean up the stash (pick one):
    - Drop (discard): `git stash drop stash@{0}`
    - Restore (re-apply): `git stash pop`
  **Expected Result**:
  - Merge commit created; `python -m pytest -q` exits `0`; `git status -sb` is clean.
  **Troubleshooting**:
  - If `git merge` fails due to local modifications, stash first then retry.
  - If you need to keep a `memory/**` file locally, restore it after the merge (it should remain untracked/ignored per `.gitignore`).
  **References**: https://git-scm.com/docs/git-merge ; https://git-scm.com/docs/git-stash ; https://git-scm.com/docs/git-rm
