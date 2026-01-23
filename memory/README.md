# Memory

这里存放**可长期复用**的“任务过程记录”（和 `memo/` 不同：`memo/` 是临时草稿，交付前必须删除）。

## 规则（执行顺序）

1. 接到任务后，先检查/创建 `memory/`。
2. 先读本文件（`memory/README.md`）确认是否已有同任务记录：
   - 有：进入对应 folder 继续追加记录（不要重复新建）。
   - 无：新建 `memory/YYYYMMDD_<slug>/` 并登记到本文件。
3. 开始执行后若使用 plan（`update_plan`）：每完成一个 step，必须在 `memory/<task>/README.md` 追加该 step 的“完成判据 + 证据”（命令+exit code、关键输出摘要、改动文件、通过的测试/验证）。

## Task Index

- `20260123_v6e8_v6e16_speed_check/`: v6e-8 vs v6e-16 跑 baseline（W&B）并对比 step time / 性能差距来源

按时间追加即可。格式建议：

- `YYYYMMDD_<slug>/`: <这个任务是干什么的>
