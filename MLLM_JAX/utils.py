# Copyright 2024 Jungwoo Park (affjljoo3581)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import threading
from collections import defaultdict
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import webdataset as wds
import yaml
from chex import Array, ArrayTree
from jax import NamedSharding
from jax._src.partition_spec import PartitionSpec
from jax.experimental import mesh_utils
from jax.tree_util import DictKey
from jax.sharding import Mesh,PartitionSpec as PS


class AverageMeter:
    def __init__(self, use_latest: list[str] = []):
        self.buffer = defaultdict(list)
        self.use_latest = use_latest

    def update(self, **kwargs: float):
        for k, v in kwargs.items():
            self.buffer[k].append(v)

    def summary(self, prefix: str = "") -> dict[str, float]:
        buffer = {k: np.array(v) for k, v in self.buffer.items()}
        self.buffer.clear()

        return {
            f"{prefix}{k}": v[-1] if k in self.use_latest else np.mean(v)
            for k, v in buffer.items()
        }


def save_checkpoint_in_background(
        filename, params_bytes: bytes, postfix: str = "ema"
):
    filename = f"{filename}-{postfix}.msgpack"
    def thread_fn():
        with wds.gopen(filename, "wb") as fp:
            fp.write(params_bytes)

    threading.Thread(target=thread_fn).start()



def save_checkpoint_in_background2(
        output_dir,name, params_bytes: bytes, postfix: str = "last"
):
    def thread_fn():
        filename = os.path.join(output_dir, f"{name}-{postfix}.msgpack")
        with wds.gopen(filename, "wb") as fp:
            fp.write(params_bytes)

    threading.Thread(target=thread_fn).start()






def get_layer_index_fn(path: tuple[DictKey, ...], _: Any, num_layers: int = 12) -> int:
    if path[0].key == "model" and path[1].key.startswith("layer_"):
        return int(re.match(r"layer_(\d+)", path[1].key).group(1)) + 1
    if path[0].key == "model" and path[1].key == "embed":
        return 0
    return num_layers


def load_pretrained_params(args: argparse.Namespace, params: ArrayTree) -> ArrayTree:
    with wds.gopen(args.pretrained_ckpt) as fp:
        new_params = flax.serialization.msgpack_restore(fp.read())

    # The positional embeddings will be resized when there is a difference in image
    # resolutions between pretraining and finetuning stage.
    if (
            args.posemb == "learnable"
            and new_params["model"]["embed"]["wpe"].shape
            != params["model"]["embed"]["wpe"].shape
    ):
        new_params["model"]["embed"]["wpe"] = jax.image.resize(
            new_params["model"]["embed"]["wpe"],
            params["model"]["embed"]["wpe"].shape,
            method="bicubic",
        )

    # Reinitialize the classifier head if the model was pretrained on different dataset
    # and `args.label_mapping` is not specified.
    if (
            "head" not in new_params["model"]
            or args.label_mapping is None
            and new_params["model"]["head"]["kernel"].shape
            != params["model"]["head"]["kernel"].shape
    ):
        new_params["model"]["head"] = params["model"]["head"]

    # If `args.label_mapping` is specified, then the same labels will automatically
    # replaced with the pretrained ones.
    if args.label_mapping:
        with wds.gopen(args.label_mapping) as fp:
            label_mapping = json.load(fp)
            src, dst = label_mapping["src"], label_mapping["dst"]

        kernel = np.zeros_like(params["model"]["head"]["kernel"])
        kernel[:, dst] = new_params["model"]["head"]["kernel"][:, src]

        bias = np.full_like(params["model"]["head"]["bias"], fill_value=-10.0)
        bias[dst] = new_params["model"]["head"]["bias"][src]

        new_params["model"]["head"] = {"kernel": kernel, "bias": bias}
    return new_params


def read_yaml(config_path):
    with open(config_path, 'r') as f:
        res = yaml.safe_load(f, )
        return res


def get_obj_from_str(string: str):
    module, cls = string.rsplit('.', 1)
    return getattr(importlib.import_module(module), cls)




def tree_path_to_string(path, sep=None):
    keys = []
    for key in path:
        if isinstance(key, jax.tree_util.SequenceKey):
            keys.append(str(key.idx))
        elif isinstance(key, jax.tree_util.DictKey):
            keys.append(str(key.key))
        elif isinstance(key, jax.tree_util.GetAttrKey):
            keys.append(str(key.name))
        elif isinstance(key, jax.tree_util.FlattenedIndexKey):
            keys.append(str(key.key))
        else:
            keys.append(str(key))
    if sep is None:
        return tuple(keys)
    return sep.join(keys)

def named_tree_map(f, tree, *rest, is_leaf=None, sep=None):
    """ An extended version of jax.tree_util.tree_map, where the mapped function
        f takes both the name (path) and the tree leaf as input.
    """
    return jax.tree_util.tree_map_with_path(
        lambda path, x, *r: f(tree_path_to_string(path, sep=sep), x, *r),
        tree, *rest,
        is_leaf=is_leaf
    )

def match_partition_rules(rules, params):
    """ Returns a pytree of PartitionSpec according to rules. Supports handling
        Flax TrainState and Optax optimizer state.
    """

    def get_partition_spec(name, leaf):
        # print(name,)
        if len(leaf.shape) == 0 or np.prod(leaf.shape) == 1:
            """ Don't partition scalar values. """
            return PS()
        for rule, ps in rules:
            if re.search(rule, name) is not None:
                return ps
        raise ValueError(f'Partition rule not found for param: {name}')

    return named_tree_map(get_partition_spec, params, sep='/')





def replace_env_variables(text):

    if isinstance(text,str):
        # 匹配 $VAR_NAME 或 ${VAR_NAME} 格式的环境变量
        pattern = re.compile(r'\$(\w+|\{(\w+)\})')
        # 查找并替换所有环境变量
        def replace_match(match):
            var_name = match.group(1) if match.group(1) else match.group(2)
            # 获取环境变量值，如果不存在则返回空字符串
            return os.environ.get(var_name, '')

        # 使用正则表达式替换所有匹配项

        text=pattern.sub(replace_match, text)

        try:
            text=eval(text)
        except Exception as e:
            pass


        return text
    else:
        return text


def preprocess_config(yaml):
    yaml=jax.tree_util.tree_map(replace_env_variables,yaml)
    return yaml


def get_partition_rules_llama():
    return (
        ('.*/self_attn/q_proj/kernel', PS('fsdp', 'tp')),
        ('.*/self_attn/k_proj/kernel', PS('fsdp', 'tp')),
        ('.*/self_attn/v_proj/kernel', PS('fsdp', 'tp')),
        ('.*/self_attn/o_proj/kernel', PS( 'tp', 'fsdp', )),


        ('.*/mlp/gate_proj/kernel', PS('tp', 'fsdp')),
        ('.*/mlp/up_proj/kernel', PS('tp', 'fsdp')),

        ('.*/mlp/down_proj/kernel', PS('fsdp', 'tp')),

        ('embed_tokens/embedding', PS('fsdp', 'tp')),
        ('lm_head/kernel', PS('fsdp', 'tp')),
        # ('lm_head/kernel', PS('tp','fsdp', )),

        ('scale',PS('tp')),
        # ('bias', PS('fsdp')),

        ('.*', PS(None)),
    )



def get_jax_mesh(axis_dims, names,devices=None):
    if axis_dims.startswith('!'):
        # Allow splitting a physical mesh axis if needed
        mesh_axis_splitting = True
        axis_dims = axis_dims[1:]
    else:
        mesh_axis_splitting = False

    if ':' in axis_dims:
        dims = []
        dim_names = []
        for axis in axis_dims.split(','):
            # print(axis)
            name, dim = axis.split(':')
            assert name in names
            dims.append(int(dim))
            dim_names.append(name)
        assert(set(dim_names) == set(names))
    else:
        dims = [int(x) for x in axis_dims.split(',')]
        dim_names = names
    assert len(dims) == len(names)

    if devices is not None:
        mesh_shape = np.arange(len(devices)).reshape(dims).shape
        physical_mesh = mesh_utils.create_device_mesh(mesh_shape=mesh_shape, devices=devices)
    else:
        mesh_shape = np.arange(jax.device_count()).reshape(dims).shape
        if mesh_axis_splitting:
            physical_mesh = np.array(jax.devices()).reshape(mesh_shape)
        else:
            physical_mesh = mesh_utils.create_device_mesh(mesh_shape)
    return Mesh(physical_mesh, dim_names)

# mesh_dim='dp:2,fsdp:-1,mp:1'
def get_jax_mesh2(axis_dims,devices=None):
    return get_jax_mesh(axis_dims, ('dp', 'fsdp', 'tp') ,devices=devices )









def _build_global_shape_and_sharding(
        local_shape: tuple[int, ...], global_mesh: Mesh
) -> tuple[tuple[int, ...], NamedSharding]:
    sharding = NamedSharding(global_mesh, PartitionSpec(global_mesh.axis_names))

    global_shape = (jax.process_count() * local_shape[0],) + local_shape[1:]

    return global_shape, sharding


def _form_global_array(path, array: np.ndarray, global_mesh: Mesh,sharding=None) -> jax.Array:
    """Put local sharded array into local devices"""


    if sharding is not None:
        # global_shape, _ = _build_global_shape_and_sharding(np.shape(array), global_mesh)
        b,h,n,d=array.shape
        global_shape = (b,h*4,n,d)
        # global_shape=
        print(array.shape,global_shape,sharding)

    else:
        global_shape, sharding = _build_global_shape_and_sharding(np.shape(array), global_mesh)
    try:
        local_device_arrays = np.split(array, len(global_mesh.local_devices), axis=0)
    except ValueError as array_split_error:
        raise ValueError(
            f"Unable to put to devices shape {array.shape} with "
            f"local device count {len(global_mesh.local_devices)} "
            f"at {jax.tree_util.keystr(path)}"
        ) from array_split_error

    local_device_buffers = jax.device_put(local_device_arrays, global_mesh.local_devices)
    return jax.make_array_from_single_device_arrays(global_shape, sharding, local_device_buffers)








def collect_process_data(data:jnp.ndarray,axis=0):
    local_data = []
    local_devices = jax.local_devices()

    for shard in data.addressable_shards:
        device = shard.device
        local_shard = shard.data
        if device in local_devices:
            # if jax.process_index()==0:
            #     print(device)

            local_data.append(np.array(local_shard))
    local_data = np.concatenate(local_data, axis=axis)
    return local_data

















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
  # print(min_starts,max_stops,global_shape)

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

        if jax.process_index()==0:
            print(shard.index)

  out=reconstruct_from_slices(s)


  print(out.shape,len(s))

  return out