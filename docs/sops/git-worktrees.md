# Git Worktree SOPs

- **Title**: SOP: 一台机器同时用多个分支（推荐：`git worktree`）
  **Prereqs**: `git` 支持 `git worktree`；你有权限创建额外目录
  **Steps**:
  - 进入你的主工作目录（worktree A）：`cd /path/to/repo`
  - 确认当前状态：`git status`
  - 新建一个额外工作目录（worktree B），并在里面 checkout 另一个分支：
    - 已有分支：`git worktree add ../<repo>-<branch> <branch>`
    - 新建分支：`git worktree add -b <new-branch> ../<repo>-<new-branch>`
  - 在另一个终端进入 worktree B：`cd ../<repo>-<branch>`，确认分支：`git rev-parse --abbrev-ref HEAD`
  - 结束后清理 worktree B：`git worktree remove ../<repo>-<branch>`
  - 清理残留元数据（可选）：`git worktree prune`
  **Expected Result**: 两个不同目录可同时处于不同分支；互不影响（不会因为 `git switch` 彼此切分支）
  **Troubleshooting**: 若提示分支已被 checkout（branch is already checked out），说明该分支正在别的 worktree 使用；换一个分支名，或用 `--detach` 创建分离头指针 worktree
  **References**: https://git-scm.com/docs/git-worktree

- **Title**: SOP: 两个终端在同一个 repo 目录时的注意事项
  **Prereqs**: N/A
  **Steps**:
  - 同一个目录只有一个 “当前 checkout” 状态；两个终端会共享同一份 `.git` + 工作区文件
  - 在一个终端执行 `git switch <branch>` / `git checkout <branch>` 后，另一个终端看到的文件也会随之变化
  - 若你想并行跑两套代码/两套实验：用 `git worktree`（上条 SOP）或直接 clone 两份到不同目录
  **Expected Result**: 避免在同一目录用两个终端“以为各自不同分支”导致互相覆盖/冲突
  **Troubleshooting**: 需要在同一目录切分支时，先 `git status` 确认无未提交修改；必要时用 `git stash -u` 暂存
  **References**: https://git-scm.com/docs/git-switch ; https://git-scm.com/docs/git-checkout

