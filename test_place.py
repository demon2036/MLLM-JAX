import jax
import jax.numpy as jnp
from jax import NamedSharding
import numpy as np
from jax._src.mesh import Mesh
from jax._src.partition_spec import PartitionSpec

from MLLM_JAX.utils import get_jax_mesh2
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_num_cpu_devices', 4)


def collect_process_data(data: jnp.ndarray):
  local_devices = jax.local_devices()
  shard_infos = []

  # 收集每个本地 shard 的 index 和数据
  for shard in data.addressable_shards:
    if shard.device in local_devices:
      shard_infos.append((shard.index, np.array(shard.data)))

  if not shard_infos:
    raise ValueError("未发现本地设备上的 shard 数据")

  n_dims = len(shard_infos[0][0])  # 数据的维度
  sharded_axes = []  # 记录所有被分片的轴
  axis_slices = {}  # 记录每个轴的切片范围

  # 找出所有被分片的轴
  for axis in range(n_dims):
    slices = [info[0][axis] for info in shard_infos]
    unique_slices = {(s.start, s.stop) for s in slices}

    if len(unique_slices) > 1:
      sharded_axes.append(axis)
      axis_slices[axis] = sorted(unique_slices)

  if not sharded_axes:
    raise ValueError("未能检测到任何分片轴")

  # **按照多个轴依次拼接**
  restored_data = {}
  for axis in sorted(sharded_axes):
    shard_infos.sort(key=lambda x: (x[0][axis].start or 0))  # 先排序
    data_list = [info[1] for info in shard_infos]
    restored_data[axis] = np.concatenate(data_list, axis=axis)  # 逐步拼接

  # **确保最终数据形状与原始数据形状匹配**
  final_data = list(restored_data.values())[-1]  # 取最后一个拼接后的数据
  return final_data


mesh= get_jax_mesh2("2,1,-1")

data_sharding = jax.sharding.NamedSharding(mesh, PartitionSpec(['dp', 'fsdp'] ,'tp' ))

def init_data(data):
  return data


jit_init_data = jax.jit(init_data, out_shardings=data_sharding, )

b,h,n,d=12,4,1024,128
x=jnp.zeros((b,h,n,d))
x=jit_init_data(x)

print(x.shape)

print(collect_process_data(x).shape)







