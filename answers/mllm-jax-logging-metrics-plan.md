# MLLM-JAX（GRPO/GSM8K）：我们应该打哪些 log（以及当前已实现哪些）

> 目标：把训练过程可观测化（可对比、可定位瓶颈、可 debug），对齐 AReaL 的思路：**指标采集**（哪里产生）与 **指标落地**（写到 W&B/TensorBoard）解耦；并在多进程/多 host 下避免重复打点。

当前落地位置（training runner）：

- 训练入口：`scripts/run_grpo_gsm8k_training.py`
- runner：`plugins/training/runner/grpo_gsm8k.py`
- reward funcs：复用 `training2.py` 的 `reward_correct/reward_format/tag_count_reward`

---

## 1) 指标命名规范（推荐）

建议按“语义域”分组，W&B key 用 `/` 层级：

- `train/...`：训练 batch 的统计（rollout → reward → adv → update）
- `eval/...`：验证/评估（rollout + reward；不更新参数）
- `time/train/...`、`time/eval/...`：分阶段耗时（秒）
- `throughput/train/...`：吞吐（tokens/s 等）

原则：

- **train 和 eval 的 key 同构**（方便 W&B 面板直接对比）
- **尽量记录 global（跨 host 聚合后的统计）**，避免 process0 只看到了本地 shard
- **不记录大张量**（如 per-token logps），只记录标量/小向量的聚合结果

---

## 2) 我们这里“具体可以打什么 log”

下面给的是 “最有用、最常看、最能定位问题” 的集合。

### 2.1 奖励（reward）指标（train + eval）

总奖励（加权和）：

- `train/reward_total_mean`
- `train/reward_total_std`
- `train/reward_total_min`
- `train/reward_total_max`

逐 reward function（注意：当前实现记录的是 **加权后的贡献**；如果 weight=0.5，则 mean 也被 0.5 缩放）：

- `train/reward_correct_mean`（本质上就是 correct rate）
- `train/reward_format_mean`
- `train/tag_count_reward_mean`

eval 同构：

- `eval/reward_total_mean/std/min/max`
- `eval/reward_correct_mean` / `eval/reward_format_mean` / `eval/tag_count_reward_mean`

用途：

- 训练是否在“学到正确”（correct_rate 是否上升）
- 是否被格式奖励牵着跑（format/tag reward 很高但 correct 不涨）
- reward 的方差是否异常（std 很大通常意味着 sampling 不稳定或奖励函数异常）

### 2.2 优势（advantages）指标（train）

GRPO 的 advantage（按 `group_ids` 在组内标准化）：

- `train/adv_mean`（理论上应接近 0）
- `train/adv_std`（理论上应接近 1；极端偏离通常表示分组或 reward 异常）
- `train/adv_min`
- `train/adv_max`

用途：

- 快速发现 group_ids/分组错乱（adv 分布会很怪）
- 快速发现 reward 全部相同（adv std 接近 0）

### 2.3 update（优化）指标（train）

来自 `training2.training_step(...)`：

- `train/loss`
- `train/entropy`

用途：

- loss 是否下降、entropy 是否塌缩（塌缩常见于奖励过强/采样温度过低等）

### 2.4 序列长度 / token 统计（train）

用 `attention_mask`（总长度）与 `labels`（completion mask）推导：

- `train/seq_prompt_len_mean` / `train/seq_prompt_len_max`
- `train/seq_completion_len_mean` / `train/seq_completion_len_max`
- `train/seq_total_len_mean` / `train/seq_total_len_max`
- `train/total_valid_token_count`（全局：`labels[:,1:].sum()` 的 host 侧聚合）

用途：

- completion 是否在变长（长度突然变短可能是模型退化或 EOS 过早）
- prompt/total 是否超出你设定的桶（定位 prefill/global_length 配置问题）

### 2.5 分阶段耗时（train + eval）

训练步拆分（秒）：

- `time/train/rollout_s`
- `time/train/reward_s`
- `time/train/advantages_s`
- `time/train/shard_s`（把 host numpy batch 变成 global sharded array）
- `time/train/update_s`
- `time/train/step_s`

eval 同构：

- `time/eval/rollout_s`
- `time/eval/reward_s`
- `time/eval/step_s`

用途：

- 直接定位瓶颈：rollout 慢/更新慢/数据搬运慢
- 观察 compile 影响：第 0 步 `update_s` 往往更大（JIT compile）

### 2.6 吞吐（throughput）指标（train）

- `throughput/train/valid_tokens_per_s`（按 `time/train/step_s` 计算）
- `throughput/train/valid_tokens_per_s_update`（按 `time/train/update_s` 计算）

用途：

- 对比不同 mesh/参数/num_pre_q 的总体效率

---

## 3) 当前已在代码里落地（W&B）

实现位置：`plugins/training/runner/grpo_gsm8k.py`

- train：每步都会打 `train/*`、`time/train/*`、`throughput/train/*`
- eval：通过 config 控制，默认关闭：
  - `eval_every_steps: 0`（0=禁用）
  - `eval_batches: 1`
  - `eval_split: test`
  配置入口：`plugins/training/configs/grpo_gsm8k_default.yaml`

多 host 聚合策略：

- reward/advantages/长度等都用 `jax.experimental.multihost_utils.process_allgather` 聚合后再算 mean/std/min/max（保证 W&B 看到的是 global 统计）
- W&B 只由 `jax.process_index()==0` 写入（避免重复 run）

---

## 4) 下一步（如果你要更像 AReaL）

AReaL 还有两类对排障很有帮助、但我们当前还没落的 log：

1) **版本/权重同步相关**（当 rollout 与 update 分离成两个 engine 后）
   - `weights/sync_s`, `weights/version`, `weights/bytes` …
2) **session/trace 级别事件**（PerfTracer/SessionTracer）
   - 适合未来做异步 rollout、staleness/version 控制时定位“卡在哪里/哪一步慢”

