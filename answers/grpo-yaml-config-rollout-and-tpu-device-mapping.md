# GRPO/GSM8K：YAML 配置、Rollout 机制、以及 2-host TPU 上的 device/mesh 分配说明

本文回答三个问题：

1) 现在的训练是否支持“像以前一样用 YAML 传参/集中配置”？  
2) 现在的 rollout 是怎么做的（和你之前的 rollout 是否同类）？  
3) rollout 过程中 TPU 的 device/mesh 是怎么分配与参与计算的？

---

## 1) 配置方式：YAML（可控）+ env（兼容/secret）

当前仓库里有两条训练入口路径：

### 1.1 YAML 驱动（你之前那种风格，jit8 插件入口）

- 入口：`test_jit8.py` → `plugins/jit8_train/run.py:cli_main`
- 配置文件：`plugins/jit8_train/configs/gsm8k_default.yaml`
- 加载逻辑：`plugins/jit8_train/config.py:load_config`
- 用法（示例）：
  - `python test_jit8.py --config plugins/jit8_train/configs/gsm8k_default.yaml --set training_steps=20 --set num_pre_q=8`

这条路径是“老风格 YAML 配置 + --set 覆盖”的典型实现，依然保留（兼容）。

### 1.2 training runner（新增）：同样支持 YAML 配置（并保留你之前的 env 方式）

- 入口：`scripts/run_grpo_gsm8k_training.py`
- 默认 YAML 配置：`plugins/training/configs/grpo_gsm8k_default.yaml`
- 加载逻辑：`plugins/training/config.py:load_config`
- `--set` 覆盖（repeatable）：`scripts/run_grpo_gsm8k_training.py:main`
- 兼容 env（历史 SOP 用法）：`scripts/run_grpo_gsm8k_training.py:_apply_env_overrides`

用法（示例）：

- 仅查看最终配置（不需要 JAX）：
  - `python scripts/run_grpo_gsm8k_training.py --print-config`
- 指定 config 并覆写少量字段：
  - `python scripts/run_grpo_gsm8k_training.py --config plugins/training/configs/grpo_gsm8k_default.yaml --set steps=20 --set num_pre_q=8`

### 1.3 配置优先级（training runner）

`scripts/run_grpo_gsm8k_training.py` 的实际合并/覆盖顺序是：

1. 内置默认值：`plugins/training/config.py:DEFAULT_CONFIG`
2. YAML 文件：`plugins/training/configs/grpo_gsm8k_default.yaml`
3. CLI 覆盖：`--set key=value`
4. 环境变量覆盖（为了兼容历史 SOP）：如 `STEPS/NUM_PRE_Q/...`（见 `scripts/run_grpo_gsm8k_training.py:_apply_env_overrides`）
5. 派生默认：`max_length_total=None` 时会变成 `max_length_sample + 128`；`wandb_name=None` 时会自动生成（见 `scripts/run_grpo_gsm8k_training.py:_cfg_from_dict`）

### 1.4 `.env` 的角色：只用于 secret（WANDB_API_KEY），不进 Git

- `.env`（gitignored）用于放 `WANDB_API_KEY`：`.env.example`
- entrypoint 会自动尝试加载：
  - `<repo>/.env`
  - `/root/.env`
  （见 `scripts/run_grpo_gsm8k_training.py:_load_dotenv_if_present`）
- 同步到 2-host TPU 的方式：`scripts/sync_env_to_tpu_vm.sh`（worker=all）

---

## 2) Rollout 现在怎么做：同步 on-policy，底层仍是同一套 Sampler（同类）

现在 training runner 的“一步训练”仍然是你熟悉的四阶段同步 pipeline：

`rollout → reward → advantages → update`

端到端编排位置：`plugins/training/runner/grpo_gsm8k.py:run_grpo_gsm8k`

### 2.1 Rollout（高层语义）

在每个 step：

1) 从 GSM8K 取 `batch_size` 个问题（prompt）。  
2) 每个 prompt 复制 `num_pre_q` 次（同一 prompt 多次采样），形成 `B_local = batch_size * num_pre_q` 条 rollout。  
3) 为每条 sample 赋一个显式分组 id：`group_ids`（shape `[B_local]`），同一 prompt 的所有 sample 拥有相同 group_id。  
4) 调用 sampler 生成 completion，得到：
   - `answers`（字符串列表）
   - `batch`（训练用 tensor：`input_ids/attention_mask/labels`）

对应代码（runner 层）：

- prompt repeat + group_ids：`plugins/training/runner/grpo_gsm8k.py:run_grpo_gsm8k`
- rollout 实现：`plugins/training/grpo/sampling.py:generate_answers_and_training_batch`

### 2.2 Rollout 的具体实现（token 级）

`plugins/training/grpo/sampling.py:generate_answers_and_training_batch` 的关键步骤：

1) **组装 chat prompt**：`build_chat_prompts()` 调用 `tokenizer.apply_chat_template(...)`，system prompt 来自 `prompts/prompts.py:system_prompt`。  
2) **tokenize**：得到 `input_ids/attention_mask`（JAX tensor），并计算 `position_ids`。  
3) **prefill bucket**：根据 `desired_length=max(global_length, prompt_len)`，用 `sampler.find_ceil(...)` 找到一个 prefill bucket（例如 512）。  
4) **pad 到 prefill_length**：构造 `input_ids_pad/pad_attention/pad_position_ids`。  
5) **调用 `sampler.generate(...)`**（核心）：见 `MLLM_JAX/sample/sample_state_right_padding2.py:Sampler.generate`
   - prefill：`model.apply(...)` 填 cache + logits
   - decode：循环调用 `Sampler.infer`（jitted），每步用 `sample_fn`（shard_map 的 top-k sampling）采下一个 token
   - 结束条件：全 batch `eos` 或达到 `max_length_sample`
6) **回收本机数据**：`collect_process_data(...)` 把本机 addressable shards 收回到 host numpy，得到：
   - `local_token_buffer/local_attention_mask/local_sample_step`
7) **构造训练 batch**：把 prompt token + completion token 拼回 `train_input_ids/train_attention_mask`，并用 completion mask 生成 `labels`。

这套 rollout 的“本质”与之前一致：**prefill + decode 的同步采样**，再把采样结果拼成训练 batch。区别在于：现在把实现抽成了可复用模块，并把 batch schema（keys/shapes）显式化。

---

## 3) rollout 期间 TPU 的 device/mesh 如何分配

### 3.1 多主机（2-host）JAX 基本事实

以 v4-16（2-host, megacore）为例（见 `docs/sops/tpu-vm-v4-16-grpo-gsm8k-wandb-20steps.md` 的已验证环境）：

- `jax.process_count() == 2`：两个 TPU VM host 各 1 个进程
- `jax.local_device_count() == 4`：每个 host 4 个 local device
- `jax.device_count() == 8`：全局 8 个 device（跨 2 host）

### 3.2 Mesh 形状与轴名

mesh 构造来自：`MLLM_JAX/utils.py:get_jax_mesh2`，轴名固定为：

- `('dp', 'fsdp', 'tp')`

默认训练 runner 用：

- `mesh_shape = "1,-1,1"`（见 `plugins/training/configs/grpo_gsm8k_default.yaml`）

在 v4-16（device_count=8）上，这会形成：

- `dp=1, fsdp=8, tp=1`（纯 FSDP）

含义：

- **参数（weights）**按 partition rules 主要切在 `fsdp` 轴上（见 `MLLM_JAX/utils.py:get_partition_rules_llama`），所以 8 个 device 共同持有模型分片。
- **batch** 的 shard 也会沿 mesh 进行分片（见下一节），所以 rollout 生成并不是“只用单卡/单 host”，而是整个 mesh 一起参与。

### 3.3 Rollout 输入 batch 是怎么映射到 device 的

Rollout 里把 host numpy / host jax tensor 放到 mesh 的关键函数是：

- `MLLM_JAX/utils.py:_form_global_array`

它做了两件关键事：

1) 设定 global shape：`(jax.process_count() * local_B, ...)`  
2) 沿 axis=0（batch 维）把 **本机** `local_B` 切成 `len(mesh.local_devices)` 份，然后 `device_put` 到本机 4 个 device 上：
   - 这就是为什么 runner 会强制 `local_B % local_device_count == 0`（见 `plugins/training/runner/grpo_gsm8k.py:_ensure_batch_multiple_of_local_devices`）

因此，在 2-host 上你可以把 batch 直观理解为：

- Host0 提供 `local_B` 条 prompt，分给它的 4 个 device
- Host1 提供 `local_B` 条 prompt，分给它的 4 个 device
- 合起来形成 global batch `B_global = 2 * local_B`，被全局 8 个 device 分摊

### 3.4 Rollout 计算（prefill/decode）到底用哪些 device

在 `MLLM_JAX/sample/sample_state_right_padding2.py:Sampler.generate` 中：

- prefill：`model.apply(...)` 在 **global mesh** 上对 sharded batch + sharded params 运行
- decode：`Sampler.infer` 迭代，同样在 **global mesh** 上运行（含 cache），并用 `shard_map` 的 `sample_fn` 采样

结论：

- rollout 不存在“额外单独的 rollout device 池”，就是用同一个 mesh 上的全部 TPU device（跨 host）做生成
- 由于 params 是 FSDP sharded，rollout 的 forward/step 内会发生跨 host 通信（这是模型并行/参数分片带来的必然）

### 3.5 为什么 rollout 结束后每个 host 只拿到本地输出

`Sampler.generate` 最后返回的是 `local_*`：

- `collect_process_data(...)` 只收集 **本机 addressable shards**，所以每个进程拿到的是它本机对应的那部分 sample（`local_B` 条）。
- 如果需要全局统计（例如 reward 的 global mean/std），runner 再显式 `process_allgather`（见 `plugins/training/runner/grpo_gsm8k.py`）。

---

## 4) 你最关心的“是否像以前那样的 rollout？”

是同一类（同步、prefill+decode 的 on-policy 采样），并且底层仍然调用上游的 `Sampler.generate`。

现在的主要变化是工程化层面：

- rollout/reward/adv/update 被拆成清晰模块（`plugins/training/grpo/*`），runner 只做编排（`plugins/training/runner/grpo_gsm8k.py`）。
- advantages 不再依赖 `groups` reshape，而是显式 `group_ids`（见 `plugins/training/grpo/advantages.py`），避免配置错误导致 silent bug。

