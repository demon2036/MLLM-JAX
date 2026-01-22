# Git Diff Inspect SOPs

- **Title**: SOP: Inspect working tree + latest commit diff
  **Prereqs**: OS: Windows (PowerShell); Git: `2.52.0.windows.1`
  **Steps**:
  - Go to repo root: `cd <repo>`
  - Confirm current branch + working tree status:
    - `git rev-parse --abbrev-ref HEAD`
    - `git status -sb`
  - Inspect uncommitted (working tree) changes:
    - Summary: `git diff --stat`
    - Patch: `git diff -- <path>`
  - Inspect the latest commit (HEAD):
    - Commit id + subject: `git log -1 --oneline`
    - Summary: `git show --stat HEAD`
    - Patch: `git show HEAD -- <path>`
  **Expected Result**:
  - `git status -sb` shows modified/untracked files
  - `git diff` shows uncommitted hunks; `git show` shows committed hunks
  **Troubleshooting**:
  - If you see `LF will be replaced by CRLF`, check `git config core.autocrlf` and `.gitattributes`.
  - If diffs are huge, start with `--stat` or add `-- <path>`.
  **References**: https://git-scm.com/docs/git-diff ; https://git-scm.com/docs/git-show
