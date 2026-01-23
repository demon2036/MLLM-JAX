# GitHub Push SOPs

- **Title**: SOP: Commit and push local changes to GitHub
  **Prereqs**: `git` and (optional) `gh` configured to authenticate with GitHub (HTTPS or SSH)
  **Environment (verified)**: Ubuntu 6.14; Python 3.13.9; git 2.48.1; gh 2.83.2
  **Steps**:
  - `cd /home/john/github/MLLM-JAX`
  - (Optional) Confirm GitHub auth + remote:
    - `gh auth status`
    - `git remote -v`
  - Check working tree status:
    - `git status -sb`
  - Run local checks (if any):
    - `python -m py_compile training2.py scripts/run_grpo_gsm8k_training.py test_jit8.py`
  - Stage changes:
    - `git add -A`
  - Commit with a Conventional Commit message:
    - `git commit -m "docs: update SOPs"`
  - Push to GitHub:
    - `git push origin main`
  **Expected Result**: `git push` completes without errors and the `origin/main` branch shows the new commit.
  **Troubleshooting**: If push fails with auth errors, ensure your GitHub auth is configured for the remote URL scheme.
  **References**: N/A
