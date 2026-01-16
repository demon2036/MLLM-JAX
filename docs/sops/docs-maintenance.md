# Docs Maintenance SOPs

- **Title**: SOP: Update AGENTS assistant execution policy
  **Prereqs**: N/A
  **Steps**:
  - Edit `AGENTS.md` and keep the policy short and action-focused
  - Prefer repo/SOP search and reasonable assumptions; ask only when blocked
  **Expected Result**: `AGENTS.md` guides assistants to implement end-to-end with minimal questions
  **Troubleshooting**: N/A
  **References**: N/A

- **Title**: SOP: Reorganize SOP index and split large TPU VM docs
  **Prereqs**: Ubuntu 25.04; Python 3.12.2; JAX not installed (import failed); CPU: AMD Ryzen 7 H 255 w/ Radeon 780M Graphics; GPU: `nvidia-smi` not found; repo at `/home/john/github/MLLM-JAX`
  **Steps**:
  - `ls -la docs`
  - `rg --files docs`
  - `sed -n '1,200p' docs/sops.md`
  - `wc -l docs/sops/*.md`
  - Split `docs/sops/tpu-vm-infra.md` into `docs/sops/tpu-vm-lifecycle.md` and `docs/sops/tpu-vm-runtime.md`, and add headings for quick scanning
  - Update `docs/sops.md` with grouped sections and new file paths
  **Expected Result**: SOP index is grouped by topic; TPU VM runtime notes live in `docs/sops/tpu-vm-runtime.md`
  **Troubleshooting**: If references still point to `docs/sops/tpu-vm-infra.md`, run `rg -n \"tpu-vm-infra\"` and update remaining mentions
  **References**: N/A

- **Title**: SOP: Require plan tool usage and fallback plan listing in AGENTS policy
  **Prereqs**: repo at `/home/john/github/MLLM-JAX`
  **Steps**:
  - `sed -n '1,200p' docs/sops/docs-maintenance.md`
  - `sed -n '1,200p' AGENTS.md`
  - Update the Assistant Execution Policy to require plan/update_plan usage when available, and list a plan before execution when tools are unavailable
  **Expected Result**: `AGENTS.md` states the plan-tool requirement and the no-confirmation fallback behavior
  **Troubleshooting**: N/A
  **References**: N/A

- **Title**: SOP: Reorganize `docs/sops.md` for fast lookup (quick tasks + browse + grep recipes)
  **Prereqs**: repo at `/home/john/github/MLLM-JAX`; `rg` available
  **Steps**:
  - Review current SOP inventory:
    - `find docs/sops -maxdepth 1 -type f -name '*.md' -print`
    - `wc -l docs/sops/*.md | sort -n`
    - `sed -n '1,200p' docs/sops.md`
  - Update the index structure in `docs/sops.md`:
    - Add **Quick tasks (task-based)** as the top entry point
    - Keep **Browse by area (component-based)** for the full list
    - Add **Search recipes (grep-first)** for common queries
  - Normalize any outlier SOP format (so grep recipes stay reliable):
    - Ensure each SOP entry uses the standard fields: `- **Title**:`, `**Prereqs**:`, `**Steps**:`, `**Expected Result**:`, `**Troubleshooting**:`, `**References**:`
    - Example: convert `docs/sops/git-pull-update.md` from `# SOP: ...` headings into the standard field-based format
    - Sanity-check that the grep recipes work:
      - `rg -n '^- \\*\\*Title\\*\\*:' docs/sops`
      - `rg -n 'gcloud (alpha )?compute tpus tpu-vm' docs/sops`
      - `rg -n 'TPU_NAME=|ZONE=|JAX_COMPILATION_CACHE_DIR' docs/sops`
      - `rg -n '\\*\\*Prereqs\\*\\*:' docs/sops`
      - `rg -n '\\*\\*Expected Result\\*\\*:' docs/sops`
      - `rg -n '\\*\\*Troubleshooting\\*\\*:' docs/sops`
  **Expected Result**: `docs/sops.md` supports (1) task-based lookup, (2) area browsing, and (3) grep-first discovery
  **Troubleshooting**: If `rg` patterns match nothing, confirm SOP entries still use `- **Title**:`, and update the patterns accordingly
  **References**: N/A
