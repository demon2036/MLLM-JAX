# Repository Guidelines

## Assistant Execution Policy (Required)

- 默认按用户需求“一口气”完成：实现、最小验证（能跑就跑）、更新 SOP。
- 无论任务大小，若提供 plan/update_plan（或 start plan）函数，每次任务都必须调用；若工具不可用，执行前以清晰优雅的列表列出计划并直接开始执行，不等待用户确认。
- 尽量不反问用户：先查 `docs/` 里潜在相关经验（优先看 `docs/sops/`），再在仓库内搜索（如 `rg`/`git`），必要时做网络搜索（官方文档/GitHub/PyPI），并把已用链接或命令写入 SOP。
- 仅在搜索后仍无法推进时才提问，并一次问清最少必要信息。
- 保持无侵入开发：自定义代码一律放在 `plugins/`，不要直接改动 `easydel/` 或其他上游仓库。
- TPU 上的覆盖/替换通过自写 shell 脚本完成（例如同步到覆盖目录 + `PYTHONPATH`），避免改动原始仓库内容。

## Project Structure & Module Organization

This repository is focused on validating and documenting `sglang-jax`. The repo is currently minimal; keep the layout simple and documented as it grows. Recommended folders:

- `docs/` for SOPs, validation notes, and references.
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

Add SOPs under `docs/sops/` (or add a new module) and keep `docs/sops.md` updated; append a short log here only if the entry is very small.
