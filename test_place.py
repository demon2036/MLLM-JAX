import jax
import jax.numpy as jnp
import torch
from jax import NamedSharding
import numpy as np
from jax._src.mesh import Mesh
from jax._src.partition_spec import PartitionSpec

from MLLM_JAX.utils import get_jax_mesh2
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_num_cpu_devices', 4)

print(-0.7 * float(np.finfo(np.dtype("float32")).max))
print(torch.finfo(torch.bfloat16).min)

def reconstruct_from_slices(subarray_info):
  """
  根据一组 (slices, subarray) 自动重构完整数组。

  参数:
    subarray_info: list of (slices, subarray)
        slices: 一个描述子数组在全局数组中位置的 slice 对象的元组
        subarray: 对应的局部数组，可以是 numpy 或 jax 数组

  返回:
    完整的重构后的 numpy 数组
  """
  if not subarray_info:
    raise ValueError("子数组信息列表为空")

  ndim = len(subarray_info[0][0])
  # 初始化每个维度的最小起点和最大终点
  min_starts = [np.inf] * ndim
  max_stops = [0] * ndim

  # 遍历所有切片，确定全局数组的尺寸
  for slices, sub in subarray_info:
    for i, s in enumerate(slices):
      # 如果 s.start 或 s.stop 为 None，则默认起点为 0，终点为子数组该维度大小
      start = s.start if s.start is not None else 0
      # 如果 stop 为 None，则取 start + 子数组对应维度的大小
      stop = s.stop if s.stop is not None else start + sub.shape[i]
      min_starts[i] = min(min_starts[i], start)
      max_stops[i] = max(max_stops[i], stop)

  # 计算全局数组形状（假设各维度连续无间断）
  global_shape = tuple(int(max_stop - min_start) for min_start, max_stop in zip(min_starts, max_stops))
  print(min_starts,max_stops,global_shape)

  # 创建全局数组，注意这里选择了 np.empty，你也可以选择 np.zeros 等方法
  full_array = np.empty(global_shape, dtype=np.result_type(subarray_info[0][1]))

  # 将每个子数组赋值到全局数组中对应的位置
  for slices, sub in subarray_info:
    # 调整切片，使其相对于全局数组的起点（可能不是 0）
    adjusted_slices = tuple(
      slice((s.start if s.start is not None else 0) - int(min_starts[i]),
            (s.stop if s.stop is not None else (s.start if s.start is not None else 0) + sub.shape[i]) - int(
              min_starts[i]),
            s.step)
      for i, s in enumerate(slices)
    )
    # 如果 sub 是 JAX 数组，则先转换为 numpy 数组
    sub_np = np.array(sub)
    full_array[adjusted_slices] = sub_np

  return full_array


def collect_process_data2(data: jnp.ndarray, ):
  s = []
  local_devices = jax.local_devices()
  for shard in data.addressable_shards:
    device = shard.device
    if device in local_devices:
        s.append((shard.index, shard.data))
  return reconstruct_from_slices(s)



mesh= get_jax_mesh2("2,1,-1")

data_sharding = jax.sharding.NamedSharding(mesh, PartitionSpec(['dp', 'fsdp'] ,None, 'tp' ))

def init_data(data)->jnp.ndarray:
  return data


jit_init_data = jax.jit(init_data, out_shardings=data_sharding, )

b,h,n,d=12,4,1024,128
x=jnp.zeros((b,h,n,d))
x=jit_init_data(x)

print(x.shape)

s=[]
for shard in x.addressable_shards:
  s.append((shard.index,shard.data))

print(reconstruct_from_slices(s).shape)


# print(collect_process_data(x).shape)







