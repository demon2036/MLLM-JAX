# Repository Guidelines

## Assistant Execution Policy (Required)

- 默认按用户需求“一口气”完成：实现、完整验证（端到端能跑）、更新 SOP。
- 交付前必须把对应的测试/验证 case 全部跑通（本地能跑的先跑，本地跑不了的在 TPU 上跑）；任何 traceback / 非零退出都视为未完成。
- 接到用户指派任务后：优先检查/创建 `memory/`；先读 `memory/README.md` 看是否已有同任务记录（有则复用对应 folder 并继续追加；没有则新建 `memory/YYYYMMDD_<slug>/`，并在 `memory/README.md` 登记 folder 名字与用途）。
- 无论任务大小，若提供 plan/update_plan（或 start plan）函数，每次任务都必须调用；若工具不可用，执行前以清晰优雅的列表列出计划并直接开始执行，不等待用户确认。
- 一旦开了 plan：允许按批次记录（建议每 5–10 个 step，或每个关键 milestone 记录一次），在 `memory/<task>/README.md` 追加“完成判据 + 证据”（例如：命令+exit code、关键输出摘要、改动文件、通过的测试/验证），便于后续复用。
- 尽量不反问用户：先查 `memory/` 里潜在相关经验（优先看 `memory/README.md`），再查 `docs/` 里潜在相关经验（优先看 `docs/sops.md`；若是分支专用 SOP，再看 `docs/sops/<branch>/`），再在仓库内搜索（如 `rg`/`git`；无 `rg` 时用 `git grep`/`Select-String`），必要时做网络搜索（官方文档/GitHub/PyPI），并把已用链接或命令写入 SOP。
- 仅在搜索后仍无法推进时才提问，并一次问清最少必要信息。
- 保持无侵入开发：自定义代码一律放在 `plugins/`，不要直接改动 `easydel/` 或其他上游仓库。
- 配置语义清晰优先：禁止在 `scripts/` 里用环境变量/拼接参数去“覆盖”训练超参；如果要改配置，必须新建一个新的 YAML config，并用 `*.sh --config <path>.yaml` 启动，让 W&B 配置可追踪、可复现。
- W&B：允许临时 `disabled`（例如本地调试），但在对外回复/交付前必须用 `wandb_mode=online` 跑至少一次验证；推荐整个迭代周期保持 online，便于用户在 W&B 上核对。
- TPU 上的覆盖/替换通过自写 shell 脚本完成（例如同步到覆盖目录 + `PYTHONPATH`），避免改动原始仓库内容。
- TPU 开发/运行采用 Git 同步：本地修改代码 → `git commit`/`git push` 到 GitHub → TPU VM 通过 `scripts/ssh_tpu_vm_root.sh`（或 `gcloud ... tpu-vm ssh`）执行 `git clone`/`git pull` 获取代码并运行；不要用 `gcloud ... scp` / `scp` 手动拷贝代码（见 `docs/sops.md#tpu-vm-repo-sync`）。（例外：同步本地 secret 如 `.env`，用 `scripts/sync_env_to_tpu_vm.sh` 分发到 worker=all。）
- `memory/` 是长期可复用资产，不删除；仍允许使用 repo 根目录的临时目录 `memo/` 记录调查/调试草稿（例如 `memo/<task>.md`），但在**完成任务并准备回复用户之前**必须删除整个 `memo/` 目录，避免把临时笔记作为交付物留在仓库里。

## Project Structure & Module Organization

This repository is focused on validating and documenting `sglang-jax`. The repo is currently minimal; keep the layout simple and documented as it grows. Recommended folders:

- `docs/` for SOPs, validation notes, and references (branch-specific SOPs under `docs/sops/<branch>/`, e.g., A → `docs/sops/A/`, B → `docs/sops/B/`).
- `plugins/` for non-invasive overrides and integration code.
- `scripts/` for repeatable setup or test helpers.
- `tests/` for any local verification scripts.

Keep this file as the primary contributor guide and a quick entry point.

## Build, Test, and Development Commands

Commands are intentionally captured as SOPs so they stay accurate. Do not guess; only record commands you have actually run. Use placeholders until verified.

Example template (replace placeholders with real, validated steps):

```bash
git clone <sglang-jax-repo-url>
cd <sglang-jax-dir>
<install-command>
<test-command>
```

When you add a new command, also note the environment (OS, Python/JAX versions, GPU/CPU).

## Coding Style & Naming Conventions

Until the project adopts a formatter/linter, keep files consistent and readable:

- Use 2-space indentation for config files; 4-space indentation for code unless the language ecosystem dictates otherwise.
- Prefer descriptive file and module names (no abbreviations).
- Keep helper scripts small and single-purpose; name them with verbs (e.g., `setup_env.sh`).

## Testing Guidelines

Testing requirements must be derived from actual `sglang-jax` usage. Record the framework and commands in SOPs once verified. Use stable naming patterns such as `test_*.py` or `*_test.py` depending on the framework.

## Commit & Pull Request Guidelines

Until a project-specific convention is defined, use Conventional Commits (e.g., `feat: add env setup SOP`). Pull requests should include: purpose, scope, and test results (or explain why tests were not run).

## SOP Capture (Required)

Before answering any user question, first review existing SOPs to reuse prior experience. If no SOP applies, create a new one. After work is done, summarize new learnings as an SOP entry so others can reuse them. Each SOP should be brief, deterministic, and easy to follow.

Recommended format:

- **Title**: Short, action-focused (e.g., "SOP: Clone and bootstrap sglang-jax")
- **Prereqs**: OS, Python/JAX versions, hardware notes
- **Steps**: Exact commands run, in order
- **Expected Result**: What success looks like
- **Troubleshooting**: Common errors and fixes
- **References**: Links or commit SHAs used

Add/maintain common/stable SOPs in `docs/sops.md`. If the SOP is branch-specific, create a dedicated folder per branch so they are “错开”: A branch → `docs/sops/A/`, B branch → `docs/sops/B/`.
