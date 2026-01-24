# Git Merge SOPs

- **Title**: SOP: Merge `algorithm` into `main` via fast-forward refspec
  **Prereqs**: `git`, `python`, and `pytest` available on PATH
  **Environment (verified)**: Ubuntu kernel `6.14.0-37-generic`; Python `3.12.2`; git `2.48.1`; pytest `7.4.4`
  **Steps**:
  - `cd /home/john/workdir/algorithm`
  - `git status -sb`
  - `git fetch --all --prune`
  - `git log --oneline main..algorithm`
  - `git log --oneline algorithm..main`
  - `git merge-base --is-ancestor main algorithm`
  - `git worktree list`
  - `git -C /home/john/workdir/mllm-jax status -sb`
  - `python -m pytest -q`
  - `git push origin algorithm:main`
  **Expected Result**:
  - `git merge-base --is-ancestor main algorithm` exits `0` (fast-forward possible)
  - `python -m pytest -q` prints `22 passed`
  - `git push` fast-forwards `origin/main`
  **Troubleshooting**:
  - If `algorithm..main` shows commits, merge or rebase before pushing
  - If the `main` worktree is dirty, avoid checkout and use refspec push
  **References**: N/A
