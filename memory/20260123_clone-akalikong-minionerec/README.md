# 20260123_clone-akalikong-minionerec

## Goal

Clone https://github.com/AkaliKong/MiniOneRec into `/home/john/workdir/minionerec/workdir/MiniOneRec` and verify the checkout.

## Completion Criteria

- `/home/john/workdir/minionerec/workdir/MiniOneRec/.git` exists.
- `git status -sb` is clean in the new checkout.
- `origin` remote URL matches `https://github.com/AkaliKong/MiniOneRec`.
- Repository metadata checks run without errors.

## Evidence (commands actually run)

- `git ls-remote https://github.com/AkaliKong/MiniOneRec HEAD` → `ba385c6827177c7b6b849d8b68f6349890099de3	HEAD` (exit 0)
- `mkdir -p /home/john/workdir` (exit 0)
- `git clone https://github.com/AkaliKong/MiniOneRec /home/john/workdir/MiniOneRec` (exit 0)
- `mkdir -p /home/john/workdir/minionerec/workdir` (exit 0)
- `mv /home/john/workdir/MiniOneRec /home/john/workdir/minionerec/workdir/MiniOneRec` (exit 0)
- `test -d /home/john/workdir/minionerec/workdir/MiniOneRec/.git` → `gitdir_present` (exit 0)
- `git -C /home/john/workdir/minionerec/workdir/MiniOneRec status -sb` → `## main...origin/main` (exit 0)
- `git -C /home/john/workdir/minionerec/workdir/MiniOneRec remote -v` → `origin https://github.com/AkaliKong/MiniOneRec (fetch/push)` (exit 0)
- `git -C /home/john/workdir/minionerec/workdir/MiniOneRec rev-parse --short HEAD` → `ba385c6` (exit 0)
- `git -C /home/john/workdir/minionerec/workdir/MiniOneRec fsck --no-progress` (exit 0)
- `ls /home/john/workdir/minionerec/workdir/MiniOneRec` (exit 0)

## Notes

- Use HTTPS to avoid SSH key requirements.
- Final location is under this repo’s `workdir/` as requested.
