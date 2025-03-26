import jax.numpy as jnp
import jax
import numpy as np
from jax import NamedSharding
from jax._src.mesh import Mesh
from jax._src.partition_spec import PartitionSpec


def _build_global_shape_and_sharding(
        local_shape: tuple[int, ...], global_mesh: Mesh
) -> tuple[tuple[int, ...], NamedSharding]:
  sharding = NamedSharding(global_mesh, PartitionSpec(global_mesh.axis_names))

  global_shape = (jax.process_count() * local_shape[0],) + local_shape[1:]

  return global_shape, sharding


def _form_global_array(path, array: np.ndarray, global_mesh: Mesh) -> jax.Array:
  """Put local sharded array into local devices"""

  global_shape, sharding = _build_global_shape_and_sharding(np.shape(array), global_mesh)
  try:
    local_device_arrays = np.split(array, len(global_mesh.local_devices), axis=0)
  except ValueError as array_split_error:
    raise ValueError(
      f"Unable to put to devices shape {array.shape} with "
      f"local device count {len(global_mesh.local_devices)} "
      f"at {jtu.keystr(path)}"
    ) from array_split_error

  local_device_buffers = jax.device_put(local_device_arrays, global_mesh.local_devices)
  return jax.make_array_from_single_device_arrays(global_shape, sharding, local_device_buffers)


def _top_k_sampling_batched(rng, logits, k=50, t=0.9):
  # 对所有样本的 logits 应用温度缩放
  logits = logits / t

  # 定义对单个样本的 top-k 采样函数
  def sample_single(rng, logits):
    # 提取 logits 中最高的 k 个值及其对应的索引
    top_logits, top_indices = jax.lax.top_k(logits, k)
    # 在 top_logits 上进行 categorical 采样
    sampled_index = jax.random.categorical(rng, top_logits)

    print(sampled_index.shape,top_indices.shape)

    # 返回采样后对应原 logits 的索引
    return jnp.take_along_axis(top_indices,sampled_index,axis=1)

  # 分割 rng 以获得每个 batch 的独立随机数种子
  batch_size = logits.shape[0]
  rngs = jax.random.split(rng, batch_size)
  # 对每个样本使用 vmap 进行采样

  out=jax.vmap(sample_single)(rngs, logits)
  print(out.shape)

  return out

def _temperature_sampling(rng,logits ,t=0.9):
  return jax.random.categorical(rng, logits / t)

def _greedy_sampling(rng, logits, ):
  del rng
  return jnp.argmax(logits, axis=-1)

def _nucleus_sampling(rng,logits ,p: float=0.95, t: float = 0.9, ):
  logits = logits / t
  neg_inf = np.array(-1.0e7)  # Effective negative infinity.
  logits_sorted = jnp.sort(logits, axis=-1, descending=True)
  sorted_cum_probs = jnp.cumsum(
      jax.nn.softmax(logits_sorted, axis=-1), axis=-1)
  cutoff_index = jnp.sum(sorted_cum_probs < p, axis=-1, keepdims=True)
  cutoff_logit = jnp.take_along_axis(logits_sorted, cutoff_index, axis=-1)
  logits = jnp.where(logits < cutoff_logit,
                     jnp.full_like(logits, neg_inf), logits)
  return jax.random.categorical(rng, logits)