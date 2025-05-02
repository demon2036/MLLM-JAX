from typing import Any

import flax
import numpy as np
import torch
from transformers import AutoConfig

from MLLM_JAX.language.qwen3_moe.configuration_qwen3 import Qwen3MoeConfig
from MLLM_JAX.language.qwen3_moe.qwen3_torch import Qwen3MoeSparseMoeBlock as Qwen3MoeSparseMoeBlockTorch
from flax.traverse_util import unflatten_dict,flatten_dict
import flax.linen as nn
import jax
import jax.numpy as jnp
jax.config.update('jax_platform_name', 'cpu')

def get_activation_fn(activation_name):
    if activation_name == 'gelu':
        return lambda x: jax.nn.gelu(x)
    elif activation_name == 'relu':
        return jax.nn.relu
    elif activation_name == 'silu' or activation_name == 'swish':
        return jax.nn.silu
    else:
        raise ValueError(f"Activation function {activation_name} not supported")


# class Qwen3MoeMLP(nn.Module):
#     config: Any
#     intermediate_size: Optional[int] = None
#
#     def setup(self):
#         self.hidden_size = self.config.hidden_size
#         self._intermediate_size = self.intermediate_size if self.intermediate_size is not None else self.config.intermediate_size
#         self.gate_proj = nn.Dense(
#             features=self._intermediate_size,
#             use_bias=False,
#             name="gate_proj"
#         )
#         self.up_proj = nn.Dense(
#             features=self._intermediate_size,
#             use_bias=False,
#             name="up_proj"
#         )
#         self.down_proj = nn.Dense(
#             features=self.hidden_size,
#             use_bias=False,
#             name="down_proj"
#         )
#         self.act_fn = get_activation_fn(self.config.hidden_act)
#
#     def __call__(self, x):
#         return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3MoeSparseMoeBlock(nn.Module):
    config: Qwen3MoeConfig

    def setup(self):
        self.num_experts = self.config.num_experts
        self.top_k = self.config.num_experts_per_tok
        self.norm_topk_prob = self.config.norm_topk_prob
        self.hidden_size=self.config.hidden_size
        self.moe_intermediate_size=self.config.moe_intermediate_size
        kernel_init=flax.linen.initializers.truncated_normal()
        # gating
        self.gate = nn.Dense(
            features=self.config.num_experts,
            use_bias=False,
            name="gate"
        )

        self.gate_proj = self.param(
            "gate_proj",
            kernel_init,
            # nn.with_logical_partitioning(kernel_init, self.wi_kernel_axes),
            (self.num_experts, self.hidden_size, self.moe_intermediate_size),
            # self.weight_dtype,
        )

        self.up_proj = self.param(
            "up_proj",
            kernel_init,
            # nn.with_logical_partitioning(kernel_init, self.wi_kernel_axes),
            (self.num_experts, self.hidden_size, self.moe_intermediate_size),
            # self.weight_dtype,
        )

        self.down_proj = self.param(
            "down_proj",
            kernel_init,
            # nn.with_logical_partitioning(kernel_init, self.wo_kernel_axes),
            (self.num_experts,self.moe_intermediate_size, self.hidden_size, ),
            # self.weight_dtype,
        )

    def get_topk(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_dim)

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        # In JAX, we use softmax with specific dtype
        routing_weights = jax.nn.softmax(router_logits, axis=1)

        # Get top-k experts and their weights
        routing_weights, selected_experts = jax.lax.top_k(routing_weights, self.top_k)

        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights = routing_weights / jnp.sum(routing_weights, axis=-1, keepdims=True)
        return routing_weights, selected_experts

    def permute(self, inputs):
        """Permute tokens to group by expert to fit gmm call."""
        # reshape inputs (batch, sequence, emb) to (batch * sequence, emb)
        inputs_shape = inputs.shape
        bsz_times_seq_len = inputs_shape[0] * inputs_shape[1]
        inputs_2d = jnp.reshape(inputs, (bsz_times_seq_len, inputs_shape[2]))
        routing_weights, selected_experts = self.get_topk(inputs)

        flatten_selected_experts = jnp.ravel(selected_experts)
        sorted_selected_experts = jnp.argsort(flatten_selected_experts)
        sorted_indices = sorted_selected_experts // self.top_k
        # sort inputs for number of selected experts
        sorted_inputs = jnp.take(inputs_2d, indices=sorted_indices, axis=0)#.astype(self.dtype)
        group_size = jnp.bincount(flatten_selected_experts, length=self.num_experts)
        return sorted_inputs, sorted_selected_experts, routing_weights, group_size

    def unpermute(self, intermediate, sorted_selected_experts, weights, batch_size, sequence_length):
        """Unpermute tokens to original order and combine weights."""

        unsort_intermediate = jnp.take(intermediate, indices=jnp.argsort(sorted_selected_experts), axis=0)
        print(f'{unsort_intermediate.shape=}')

        reshaped_weights = jnp.reshape(weights, (-1, self.top_k))
        reshaped_intermediate = jnp.reshape(
            unsort_intermediate,
            (reshaped_weights.shape[0], self.top_k, -1),
        )


        with jax.named_scope("weight_sum"):
            # matmul_precision = jax.lax.Precision(self.config.matmul_precision)
            matmul_precision = jax.lax.Precision('default')

            output = jnp.einsum(
                "BKE,BK -> BE",
                reshaped_intermediate.astype(jnp.float32),
                reshaped_weights.astype(jnp.float32),
                precision=matmul_precision,
            )
        return output.reshape(batch_size, sequence_length, -1)#.astype(self.dtype)



    def __call__(self, hidden_states):
        print(f'{hidden_states.shape=}')
        batch_size, sequence_length, _ = hidden_states.shape
        x, sorted_selected_experts, weights, group_sizes = self.permute(hidden_states)

        def gmm(inputs, kernel, group_sizes):
            print(f'{inputs.shape=}')
            hs_shape = inputs.shape
            # pad length is the 1st dimension of tiling size in gmm call
            pad_length = 512
            if hs_shape[0] % pad_length:
                pad_length = pad_length - hs_shape[0] % pad_length
                inputs = jax.lax.pad(inputs.astype(jnp.float32), 0.0, [(0, pad_length, 0), (0, 0, 0)])

            # inputs = inputs.astype(self.dtype)
            # kernel = kernel.astype(self.dtype)
            lhs_quantize_dtype, rhs_quantize_dtype = None, None
            megablox=False

            if megablox:
                pass
                # m, k, n = inputs.shape[0], inputs.shape[1], kernel.shape[2]
                # output = mblx.gmm(
                #     lhs=inputs,
                #     rhs=kernel,
                #     group_sizes=group_sizes,
                #     preferred_element_type=jnp.bfloat16,
                #     tiling=(min(tile_size[0], m), min(tile_size[1], k), min(tile_size[2], n)),
                #     lhs_quantize_dtype=lhs_quantize_dtype,
                #     rhs_quantize_dtype=rhs_quantize_dtype,
                # )
            else:
                output = jax.lax.ragged_dot(
                    lhs=inputs,
                    rhs=kernel,
                    group_sizes=group_sizes,
                    # preferred_element_type=jnp.bfloat16,
                    preferred_element_type=jnp.float32,
                )
            if hs_shape[0] % pad_length:
                output = output[: hs_shape[0]]
            return output

        layer_w0 = gmm(x, self.gate_proj, group_sizes)
        layer_w1 = gmm(x, self.up_proj, group_sizes)
        layer_act=nn.silu(layer_w0)
        intermediate_layer = jnp.multiply(layer_act, layer_w1)
        intermediate_output = gmm(intermediate_layer, self.down_proj, group_sizes)

        print(layer_w0.shape,intermediate_layer.shape,intermediate_output.shape)

        output = self.unpermute(
            intermediate_output, sorted_selected_experts, weights, batch_size=batch_size,
            sequence_length=sequence_length
        )
        return output







def convert_torch_to_flax_sparse_moe_block(state_dict):
    params = dict()
    params['gate.kernel']=state_dict['gate.weight'].transpose(1, 0)

    j=0
    gate_proj_experts=[]
    up_proj_experts=[]
    down_proj_experts=[]


    while f'experts.{j}.gate_proj.weight' in state_dict:
        gate_proj_experts.append(state_dict[f'experts.{j}.gate_proj.weight'].transpose(1, 0))
        up_proj_experts.append(state_dict[f'experts.{j}.up_proj.weight'].transpose(1, 0))
        down_proj_experts.append(state_dict[f'experts.{j}.down_proj.weight'].transpose(1, 0))
        j+=1

    gate_proj_experts=np.asarray(gate_proj_experts)
    up_proj_experts=np.asarray(up_proj_experts)
    down_proj_experts=np.asarray(down_proj_experts)
    params['gate_proj'] = gate_proj_experts
    params['up_proj'] = up_proj_experts
    params['down_proj']=down_proj_experts
    return flax.traverse_util.unflatten_dict(params, sep='.')



config=AutoConfig.from_pretrained('Qwen/Qwen3-30B-A3B')
config.num_experts=16
model_torch=Qwen3MoeSparseMoeBlockTorch(config=config)
state_dict=model_torch.state_dict()
flatten_state_dict=unflatten_dict(state_dict,sep='.')
print(flatten_state_dict.keys())
print(flatten_state_dict['experts'].keys())
print(state_dict['gate.weight'].shape)
print(state_dict.keys())

b,n,d=shape=(1,128,2048)
x=jnp.ones(shape,dtype=jnp.float32)

x_torch=torch.from_numpy(np.array(x))

rng=jax.random.PRNGKey(1)
model_jax=Qwen3MoeSparseMoeBlock(config=config)

params=jax.jit(model_jax.init)(rng,x)['params']
# print(params['down_proj'].shape)

params=convert_torch_to_flax_sparse_moe_block(jax.tree_util.tree_map( lambda x:x.numpy(),    state_dict))
# print(params['down_proj'].shape)
output=model_jax.apply({'params':params},x)
# print(routing_weights.shape,selected_experts)
output_torch=model_torch(x_torch)
print(output_torch.detach().numpy()-np.array(output))
"""
"""