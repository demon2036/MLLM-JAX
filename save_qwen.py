import functools
from functools import partial

import jax
import jax.numpy as jnp
from jax import NamedSharding
import numpy as np
from jax._src.mesh import Mesh
from jax._src.partition_spec import PartitionSpec
from jax.experimental.shard_map import shard_map

from MLLM_JAX.utils import get_jax_mesh2
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_num_cpu_devices', 8)





mesh= get_jax_mesh2("4,1,-1")


def inner_fn(x):
    W = jnp.ones((x.shape[-1], x.shape[-1]))  # Dummy weight matrix
    return x @ W  # Sharded matrix multiplication





def tensor_parallel_fn(x):
    print(x.shape)

    def inner_fn(x):
        print(x.shape)

    inner_fn = shard_map(inner_fn, mesh=mesh, in_specs=PartitionSpec(None,'tp'), out_specs=PartitionSpec(None,'tp'))
    inner_fn = jax.jit(inner_fn)

    return inner_fn(x)


# Outer function with Data Parallelism (DP)

# def data_parallel_fn(x):
#     return tensor_parallel_fn(x)  # Calls the inner TP-sharded function



data_parallel_fn=shard_map(tensor_parallel_fn,
    in_specs=(PartitionSpec('dp')),  # Shard input over DP axis
    out_specs=PartitionSpec('dp'),  # Output is also sharded
    mesh=mesh,auto=frozenset({'tp'}),check_rep=False
)

data_parallel_fn=jax.jit(data_parallel_fn)



# Example Input: A batch of data
batch_size = 8  # Ensure it matches the number of DP devices
feature_size = 8  # Example feature size
x = jnp.ones((batch_size, feature_size))

# Run the function
output = data_parallel_fn(x)
