# Task

- User request: clone `verl` / `areal` / `tunix` / `maxtext` / `slime` into this repo’s root `workdir/` folder (gitignored).
- Verify each clone by recording `HEAD` commit + `remote -v`.
- Delivery gate: local `python -m pytest -q` exits `0`.

## Canonical upstream URLs

- Tunix: `https://github.com/google/tunix.git`
- AReaL: `https://github.com/inclusionAI/AReaL.git`
- VERL: `https://github.com/volcengine/verl.git`
- MaxText: `https://github.com/google/maxtext.git`
- Slime: `https://github.com/THUDM/slime.git`

## Target paths (repo-local, gitignored)

- `workdir/tunix`
- `workdir/areal`
- `workdir/verl`
- `workdir/maxtext`
- `workdir/slime`

# Plan

1) Create `workdir/` (if missing) and confirm it’s ignored by git.
2) Create this memory folder and register it in `memory/README.md`.
3) Add/update an SOP with the exact clone + verification commands actually used.
4) Probe remote reachability (`git ls-remote`) and then clone all five repos.
5) Record each repo’s `HEAD` and `remote -v`.
6) Run local `python -m pytest -q`.
7) Confirm `git status` stays clean (no `workdir/` artifacts tracked).

# Completion criteria

- Directories exist: `workdir/{tunix,areal,verl,maxtext,slime}`
- Each repo verification commands exit `0`:
  - `git -C workdir/<name> rev-parse --short HEAD`
  - `git -C workdir/<name> remote -v`
- This repo tests pass: `python -m pytest -q` (exit `0`)
- `git status --porcelain` does not show `workdir/` as untracked/tracked changes

# Evidence

## Setup: create repo-local `workdir/`

- Command (exit 0): `mkdir -p workdir && ls -la workdir`
- Confirm ignore rule (exit 0): `rg -n "^workdir/" .gitignore`
  - Output: `.gitignore:16:workdir/`

## Remote reachability probes (pre-clone)

All exit `0`:

- `GIT_TERMINAL_PROMPT=0 git ls-remote --heads https://github.com/google/tunix.git | head`
- `GIT_TERMINAL_PROMPT=0 git ls-remote --heads https://github.com/inclusionAI/AReaL.git | head`
- `GIT_TERMINAL_PROMPT=0 git ls-remote --heads https://github.com/volcengine/verl.git | head`
- `GIT_TERMINAL_PROMPT=0 git ls-remote --heads https://github.com/google/maxtext.git | head`
- `GIT_TERMINAL_PROMPT=0 git ls-remote --heads https://github.com/THUDM/slime.git | head`

## Clone into `workdir/` (repo-local, gitignored)

Commands (exit 0):

- `git clone --depth 1 https://github.com/google/tunix.git workdir/tunix`
- `git clone --depth 1 https://github.com/inclusionAI/AReaL.git workdir/areal`
- `git clone --depth 1 https://github.com/volcengine/verl.git workdir/verl`
- `git clone --depth 1 https://github.com/google/maxtext.git workdir/maxtext`
- `git clone --depth 1 https://github.com/THUDM/slime.git workdir/slime`

Resulting directories:

- `workdir/areal`
- `workdir/maxtext`
- `workdir/slime`
- `workdir/tunix`
- `workdir/verl`

## Verified revisions (HEAD)

Command (exit 0): `for d in tunix areal verl maxtext slime; do printf "%s\\t" "$d"; git -C "workdir/$d" rev-parse --short HEAD; done`

- Tunix: `b2be968`
- AReaL: `b066584`
- VERL: `f31df34`
- MaxText: `4bcee99`
- Slime: `9bfe152`

## Verified remotes

Command (exit 0): `for d in tunix areal verl maxtext slime; do echo "--- $d"; git -C "workdir/$d" remote -v; done`

- Tunix: `https://github.com/google/tunix.git`
- AReaL: `https://github.com/inclusionAI/AReaL.git`
- VERL: `https://github.com/volcengine/verl.git`
- MaxText: `https://github.com/google/maxtext.git`
- Slime: `https://github.com/THUDM/slime.git`

## Disk usage snapshot

Command (exit 0): `du -sh workdir/* | sort -h`

- `9.5M  workdir/slime`
- `13M   workdir/verl`
- `30M   workdir/areal`
- `78M   workdir/tunix`
- `124M  workdir/maxtext`

## Delivery gates (completed)

- Local tests:
  - Command (exit 0): `python -m pytest -q`
  - Output (summary): `13 passed in 0.29s`
- Git status excludes `workdir/`:
  - Command (exit 0): `git status --porcelain`
  - Output: only shows tracked-doc changes + new docs/memory files; no `workdir/` entries
