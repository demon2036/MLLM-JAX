import jax
import jax.numpy as jnp


def _top_k_sampling_batched(rng, logits, k=50, t=0.9):
  # 对所有样本的 logits 应用温度缩放
  logits = logits / t

  # 定义对单个样本的 top-k 采样函数
  def sample_single(rng, logits_single):
    # 获取单个样本中最大的 k 个 logits 及其索引
    top_logits, top_indices = jax.lax.top_k(logits_single, k)
    # 在 top_logits 上进行 categorical 采样，返回采样到的相对索引
    sampled_relative_idx = jax.random.categorical(rng, top_logits)


    print(logits_single.shape)

    # 根据采样结果，从 top_indices 中取出对应的绝对索引
    return top_indices[jnp.arange(0,logits_single.shape[0]),sampled_relative_idx]  # 修复索引方式

  # 分割 rng 以获得每个 batch 的独立随机数种子
  batch_size = logits.shape[0]
  rngs = jax.random.split(rng, batch_size)
  # 对每个样本使用 vmap 进行采样
  out = jax.vmap(sample_single)(rngs, logits)  # 移除多余的[:,0]索引
  return out


key=jax.random.PRNGKey(3)
x=jax.random.uniform(key,(2,1,128))

print(_top_k_sampling_batched(key,x).shape)