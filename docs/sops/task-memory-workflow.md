# Task Memory SOPs

- **Title**: SOP: Record task progress in `memory/` (reusable)
  **Prereqs**: Repo checkout; write access to repo; plan tool available (e.g. `update_plan`)
  **Steps**:
  - Ensure `memory/` exists (create it if missing)
  - Read `memory/README.md` first; reuse an existing task folder if it matches
  - If no match, create a new folder `memory/YYYYMMDD_<slug>/` and add an index entry to `memory/README.md`
  - Create/initialize `memory/<task>/README.md` (copy/adapt from `memory/_template/README.md`)
  - After you call `update_plan`, copy the finalized plan steps into `memory/<task>/README.md`
  - For each plan step completion:
    - Update the plan status (`pending → in_progress → completed`)
    - Append “completion criteria + evidence” to `memory/<task>/README.md` (commands + exit code, key output summary, files changed, tests/verification passed)
  **Expected Result**: `memory/README.md` indexes the task folder; `memory/<task>/README.md` contains step-by-step evidence that is reusable for similar tasks
  **Troubleshooting**: If you notice duplication, merge by reusing the older folder and add a note in the newer entry pointing to the canonical folder
  **References**: `AGENTS.md`, `memory/README.md`, `memory/_template/README.md`

