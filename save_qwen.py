import jax
import jax.numpy as jnp


def static_top_p_sampling(logits, key, top_p):
    # 确保所有操作保持静态形状
    sorted_indices = jnp.argsort(-logits)  # 降序排列
    sorted_logits = logits[sorted_indices]

    # 计算排序后的概率分布
    sorted_probs = jax.nn.softmax(sorted_logits)

    # 计算累积概率（使用双精度提升数值稳定性）
    cum_probs = jnp.cumsum(sorted_probs)#.astype(sorted_probs.dtype)

    # 动态确定切割点（保持静态形状计算）
    threshold_mask = cum_probs >= top_p
    valid_indices = jnp.argmax(threshold_mask, axis=0)

    # 处理未达到阈值的情况（至少保留一个token）
    valid_indices = jnp.where(jnp.any(threshold_mask, axis=0),
                              valid_indices,
                              sorted_logits.shape[-1] - 1)
    # 创建静态掩码（核心：保持固定长度）
    position_mask = jnp.arange(sorted_logits.shape[-1]) <= valid_indices
    clamped_logits = jnp.where(position_mask, sorted_logits, -jnp.inf)

    # 从修正后的分布采样
    choice_idx = jax.random.categorical(key, clamped_logits)


    return sorted_indices[choice_idx],clamped_logits,choice_idx,valid_indices


# 使用示例
key = jax.random.PRNGKey(42)
logits = jnp.array([10.0, 10.0, -10e9, 9])

# JIT编译版本
jit_sampler = jax.jit(static_top_p_sampling)

# 测试采样
print(jit_sampler(logits, key, 0.9))  # 输出采样结果