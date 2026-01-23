# SOP: Clone reference repos into repo-local `workdir/`

- **Title**: SOP: Clone Tunix/AReaL/VERL/MaxText/Slime into this repo’s `workdir/` for local inspection
  **Prereqs**: Ubuntu Linux; `git`; outbound network access
  **Environment (verified)**:
  - Repo: `/home/john/workdir/algorithm`
  - OS: Ubuntu (kernel `6.14.0-37-generic`)
  - Python: `3.12.2`
  - git: `2.48.1`

## Goal

- Get local (gitignored) clones under `workdir/` for browsing / referencing code.
- Record each repo’s `HEAD` + `remote -v` for reproducibility.
- Confirm this repo still passes tests after cloning.

## Steps (commands actually used in this repo)

### 1) Create repo-local `workdir/` (gitignored)

From repo root:

- `cd /home/john/workdir/algorithm`
- `mkdir -p workdir`
- (Optional) confirm ignore rule: `rg -n "^workdir/" .gitignore`

### 2) (Optional) Probe remote reachability (pre-clone)

All exit `0` when reachable:

- `GIT_TERMINAL_PROMPT=0 git ls-remote --heads https://github.com/google/tunix.git | head`
- `GIT_TERMINAL_PROMPT=0 git ls-remote --heads https://github.com/inclusionAI/AReaL.git | head`
- `GIT_TERMINAL_PROMPT=0 git ls-remote --heads https://github.com/volcengine/verl.git | head`
- `GIT_TERMINAL_PROMPT=0 git ls-remote --heads https://github.com/google/maxtext.git | head`
- `GIT_TERMINAL_PROMPT=0 git ls-remote --heads https://github.com/THUDM/slime.git | head`

### 3) Clone reference repos into `workdir/` (depth-1)

- `git clone --depth 1 https://github.com/google/tunix.git workdir/tunix`
- `git clone --depth 1 https://github.com/inclusionAI/AReaL.git workdir/areal`
- `git clone --depth 1 https://github.com/volcengine/verl.git workdir/verl`
- `git clone --depth 1 https://github.com/google/maxtext.git workdir/maxtext`
- `git clone --depth 1 https://github.com/THUDM/slime.git workdir/slime`

### 4) Record cloned revisions (commit + remote)

- `for d in tunix areal verl maxtext slime; do printf "%s\\t" "$d"; git -C "workdir/$d" rev-parse --short HEAD; done`
- `for d in tunix areal verl maxtext slime; do echo "--- $d"; git -C "workdir/$d" remote -v; done`

### 5) (Optional) Snapshot disk usage

- `du -sh workdir/* | sort -h`

### 6) Validate this repo still passes tests

Note: `pytest.ini` excludes `workdir/`, so these clones should not affect tests.

- `python -m pytest -q`

## Expected Result

- `workdir/` contains the cloned repos (not tracked by git).
- `python -m pytest -q` exits `0`.

## Troubleshooting

- `git clone` hangs/prompts: run with `GIT_TERMINAL_PROMPT=0` to fail fast on auth issues.
- Need an older commit than the depth-1 fetch includes: `git -C workdir/<name> fetch --unshallow`.
- Network issues: see `docs/sops/network-checks.md`.

## References

- `memory/20260123_clone-workdir-repos/README.md`
- `docs/sops/rl-four-phase-interface-research.md`
