# SOP: sglang-jax KV cache 如何设定与分配（JAX）

- **Title**: SOP: Understand & size sglang-jax KV cache (JAX)
- **Scope**: `sglang-jax` 推理引擎（JAX），KV cache 的容量设定、预分配与主要参数

## Prereqs

- 能阅读 `sglang-jax` 代码（本仓库 vendored：`workdir/sglang-jax`）
- （可选）有一份 engine 启动日志，便于对照关键 log

## Behavior（结论）

KV cache 的**容量**是在 engine 启动时（`ModelRunner` 初始化阶段）根据“可用内存 + 参数”计算并**预分配**出来的；运行时（prefill/decode）只是从 pool 里**按需分配 token slot 并写入内容**，不会无限动态扩容。

另外：sglang-jax 的 fused KV buffers 预分配实现里**确实用了 `jax.jit`** 来生成带 sharding 的 `jnp.zeros(...)`（属于一次性的初始化开销，不是每个 request 都做）。

## How it works（代码路径）

1) Engine 启动后，TP worker 会创建 `ModelRunner(...)`：
- `workdir/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py:103`

2) `ModelRunner` 初始化过程中调用 `init_memory_pool(...)`：
- `workdir/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py:151`

3) `init_memory_pool` 会先做 memory profiling 得到 `max_total_num_tokens`，并支持 `max_total_tokens` 上限覆盖：
- profiling：`workdir/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py:294`
- override/cap：`workdir/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py:395`

4) 随后创建 `MHATokenToKVPool(...)`（KV cache pool），其内部会为每一层预分配 fused KV buffer（`jnp.zeros`）：
- pool 创建：`workdir/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py:440`
- 预分配 fused KV buffers：`workdir/sglang-jax/python/sgl_jax/srt/mem_cache/memory_pool.py:300`
- 分配完成 log：`workdir/sglang-jax/python/sgl_jax/srt/mem_cache/memory_pool.py:355`

## Clear vs Free（“清空/释放 token” vs “释放 HBM 内存”）

sglang-jax 里需要区分两件事：

1) **清空/释放 token slot（逻辑层面）**：把当前占用的 token indices 归还给 allocator，并清掉 radix/tree cache。
- Python 入口：`Engine.flush_cache()`（`workdir/sglang-jax/python/sgl_jax/srt/entrypoints/engine.py:342`）
- Scheduler 实现：会 `tree_cache.reset()`、`req_to_token_pool.clear()`、`token_to_kv_pool_allocator.clear()`（`workdir/sglang-jax/python/sgl_jax/srt/managers/scheduler.py:950`）
- allocator 的 `clear()` 只是重置 free list（不会去释放 fused KV buffers 本身）（`workdir/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py:97`）

2) **释放 fused KV buffers 占用的 HBM（物理层面）**：这在 sglang-jax 没有对外暴露“热释放 + 热重建”的稳定 API；通常要靠**重启 Engine 进程**来让 JAX/XLA allocator 彻底回收（即便如此也可能存在 allocator 复用/缓存行为）。
   - 只要 KV buffer 的 shape/dtype/sharding 不变，“模型 forward 的 JIT 编译”通常可以复用；但如果你选择用 `jax.clear_caches()` 或者直接重启进程/重建 Engine，则会更容易触发重新编译（尤其是首次 forward）。

## Rollout 场景建议（实践）

- rollout 一轮结束后，如果你想“回到干净状态”（尤其是启用 radix/tree cache 的情况下），优先用 `engine.flush_cache()`，它不会重新分配 KV buffers，因此也不需要重新做那一套 KV buffer 初始化。
- 如果你做了**热替换权重**（weights hot-swap），强烈建议在切换前后调用一次 `engine.flush_cache()`：因为旧 KV 是用旧权重算出来的，继续复用没有意义且可能造成异常行为（radix/tree cache 也同理）。

## Knobs（你可以调的参数）

这些参数来自 `ServerArgs`，可通过 `Engine(**kwargs)` 传入（`Engine` 会内部构造 `ServerArgs`）：
- 参数入口：`workdir/sglang-jax/python/sgl_jax/srt/entrypoints/engine.py:85`
- 字段定义：`workdir/sglang-jax/python/sgl_jax/srt/server_args.py:54`

常用的 KV cache 相关 knobs：
- `max_total_tokens`: KV pool 的 token 容量上限（更小 => 更省 KV 内存，但并发/长上下文能力下降）
- `mem_fraction_static`: profiling 时给 KV cache 预留的内存比例（值越大通常允许更大的 KV pool；过小可能直接报内存不足）
- `kv_cache_dtype`: KV cache dtype（`auto` 跟随模型 dtype；`bf16` 强制 bfloat16）
- `page_size`: token pool 的 page size（会影响对齐与分配策略；默认 `1`）
- `max_running_requests`: 并发上限（若不设，会从 `max_total_num_tokens` 推导；也会受 attention backend 约束）

## What to look for in logs（如何确认）

Engine 启动时通常会看到类似 log（关键词）：
- `TPU Memory profiling: ... max_tokens=...`（profiling 输出）
- `ModelRunner max_total_num_tokens: ...`（最终 KV 容量）
- `JAX Fused KV Cache allocated. #tokens: ..., Fused KV size: ... GB`（fused KV 预分配完成）

## Troubleshooting

- 报错：`Not enough memory. Please try to increase --mem-fraction-static.`
  - 提高 `mem_fraction_static`
  - 降低 `max_total_tokens`
  - 设 `kv_cache_dtype="bf16"`（若当前模型 dtype 更大）
  - 降低并发（`max_running_requests`）或缩短上下文需求

## References

- 已验证的 TPU 端到端跑通示例（含内存观察与参数热替换）：`docs/sops/tpu-sglang-jax-qwen25-3b-mllm-param-swap-memory.md`
