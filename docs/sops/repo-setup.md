# Repo Setup SOPs

- **Title**: SOP: Clone this repo (placeholder)
  **Prereqs**: `git` available
  **Steps**:
  - `git clone <repo-url>`
  - `cd <repo-dir>`
  - `git status -sb`
  **Expected Result**: Repo cloned and git status works in the checkout
  **Troubleshooting**: N/A
  **References**: N/A

- **Title**: SOP: Quick syntax smoke-check (Python)
  **Prereqs**: Python 3.x available
  **Steps**:
  - `python -m py_compile training2.py test_jit9.py`
  **Expected Result**: Exit code 0 with no output
  **Troubleshooting**: If file paths differ, adjust the command for your checkout
  **References**: N/A
