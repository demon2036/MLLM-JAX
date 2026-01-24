# 20260123_clone-minionerec-to-workdir1

## Goal

Clone this repo from `/home/john/workdir/minionerec` into `/home/john/workdir1/minionerec` as a separate working copy.

## Completion Criteria

- `/home/john/workdir1/minionerec` exists and contains a valid `.git/`.
- `git status -sb` works and is clean in the new checkout.
- `origin` remote URL in the new checkout matches the source checkout.

## Evidence (commands actually run)

- `git -C /home/john/workdir/minionerec rev-parse --short HEAD` → `c12631c` (exit 0)
- `git -C /home/john/workdir/minionerec remote get-url origin` → `git@github.com:demon2036/MLLM-JAX.git` (exit 0)
- `mkdir -p /home/john/workdir1` (exit 0)
- `git clone /home/john/workdir/minionerec /home/john/workdir1/minionerec` (exit 0)
- `git -C /home/john/workdir1/minionerec remote set-url origin git@github.com:demon2036/MLLM-JAX.git` (exit 0)
- `git -C /home/john/workdir1/minionerec status -sb` → `## minionerec...origin/minionerec` (exit 0)
- `git -C /home/john/workdir1/minionerec rev-parse --short HEAD` → `c12631c` (exit 0)
- `git -C /home/john/workdir1/minionerec remote -v` (exit 0)
- `git -C /home/john/workdir1/minionerec fsck --no-progress` (exit 0; reports dangling commits only)

## Notes

- Prefer local clone (no network dependency), then normalize the `origin` URL to match the source repo.

## Variant: clone into `./workdir1` (nested working copy)

### Goal

Clone the GitHub repo (same `origin` as the current checkout) into a subfolder `workdir1/` under the current directory.

### Completion Criteria

- `workdir1/.git/` exists (a normal gitdir clone).
- `git -C workdir1 status -sb` works and is clean.
- `workdir1` `origin` points to the expected GitHub URL.

### Evidence (commands actually run)

- `git branch --show-current` → `minionerec` (exit 0)
- `git rev-parse --short HEAD` → `c12631c` (exit 0)
- `git remote get-url origin` → `git@github.com:demon2036/MLLM-JAX.git` (exit 0)
- `ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -T git@github.com </dev/null` → authenticated banner (exit 0)
- `git clone git@github.com:demon2036/MLLM-JAX.git workdir1` (exit 0)
- `test -d workdir1/.git` → `workdir1_has_gitdir` (exit 0)
- `git -C workdir1 status -sb` → `## main...origin/main` (exit 0)
- `git -C workdir1 remote -v` → `origin git@github.com:demon2036/MLLM-JAX.git` (exit 0)
- `git -C workdir1 rev-parse --short HEAD` → `c12631c` (exit 0)
- `git -C workdir1 fsck --no-progress` (exit 0)
