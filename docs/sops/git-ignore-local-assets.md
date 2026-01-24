# SOP: Ignore answers/memory/workdir in Git

- **Title**: SOP: Ignore `answers/`, `memory/`, and `workdir/` in Git
- **Prereqs**: Repo checkout; `git` available
- **Environment (verified)**:
  - OS: Linux `6.14.0-37-generic` (Ubuntu)
  - Python: `3.12.2`

## Steps (commands actually used)

From repo root:

1) Inspect current ignore rules and tracked files

```bash
cat .gitignore
git ls-files answers memory workdir
rg -n "^workdir/" .gitignore
```

2) Remove tracked entries from the index (keep files on disk)

```bash
git ls-files -z answers | git update-index --force-remove -z --stdin
git ls-files -z memory | git update-index --force-remove -z --stdin
```

3) Validate index and run tests

```bash
git status -sb
git ls-files answers memory workdir
python -m pytest -q
```

## Expected Result

- `.gitignore` contains `answers/`, `memory/`, and `workdir/`.
- `git ls-files answers memory workdir` prints nothing.
- `python -m pytest -q` exits `0`.

## Troubleshooting

- If files remain tracked, re-run the `git update-index --force-remove` commands.
- If `git status -sb` shows unexpected deletions, re-check the `.gitignore` edits.
