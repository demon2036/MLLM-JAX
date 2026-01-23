# Git Pull Update SOPs

- **Title**: SOP: Update repo with `git pull --ff-only`
  **Prereqs**: OS: Linux (Ubuntu kernel `6.14.0-37-generic`); Git: `2.48.1`; Python: `3.13.9` (only for running the local tests in this SOP)
  **Steps**:
  - `cd /home/john/github/MLLM-JAX`
  - Inspect current branch + cleanliness:
    - `git status -sb`
    - `git remote -v`
  - Fetch then fast-forward only (no merge commits):
    - `git fetch --all --prune`
    - `git pull --ff-only`
  - Minimal sanity check (if tests exist):
    - `python -m py_compile training2.py scripts/run_grpo_gsm8k_training.py test_jit8.py`
  **Expected Result**:
  - `git pull --ff-only` prints `Fast-forward` (or `Already up to date.`)
  - `git status -sb` shows a clean working tree
  - Sanity check completes successfully (exit code 0)
  **Troubleshooting**:
  - If `git pull --ff-only` fails, the branch likely diverged or you have local commits; decide whether to rebase/merge (not covered in this SOP).
  - If `git status -sb` shows local changes, commit or stash them before pulling.
  **References**: N/A
