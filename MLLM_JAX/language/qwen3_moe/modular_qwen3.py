import functools
import math
from enum import Enum, auto
from typing import Any

import einops
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel, splash_attention_mask
from jax.experimental.shard_map import shard_map

from .configuration_qwen3 import Qwen3MoeConfig
from ..llama.llama import LlamaMLP, LlamaAttention, LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding, \
    apply_rotary_pos_emb, repeat_kv
from jax.sharding import PartitionSpec as P

from MLLM_JAX.kernels import megablox as mblx


def get_all_to_all_params(all_shards_group_sizes, local_expert_size, num_expert_shards):
    class TransformStrategy(Enum):
        INPUT_OFFSET = auto()
        SEND_SIZE = auto()
        OUTPUT_OFFSET = auto()
        RECV_SIZE = auto()

    def transform_array(input_array, shard_id, strategy):
        """This function transforms the input array based on the specified strategy,
        preparing it for the usage with `ragged_all_to_all` API. The transformation
        determines how data is sent and received between shards.
        """
        if strategy == TransformStrategy.INPUT_OFFSET:
            # Index of input array for the send
            local_array = input_array[shard_id]
            return jnp.concatenate((jnp.array([0]), jnp.cumsum(local_array)[:-1]))
        elif strategy == TransformStrategy.SEND_SIZE:
            # Size of input array for the send
            return input_array[shard_id]
        elif strategy == TransformStrategy.OUTPUT_OFFSET:
            # Received index in the target output
            zero_row = jnp.zeros((1,) + input_array.shape[1:], dtype=input_array.dtype)
            array_with_zeros = jnp.concatenate((zero_row, input_array), axis=0)
            cumulated_array = jnp.cumsum(array_with_zeros, axis=0, dtype=input_array.dtype)
            return cumulated_array[shard_id]
        elif strategy == TransformStrategy.RECV_SIZE:
            # Received size in the traget output
            return input_array[:, shard_id]
        else:
            raise ValueError(f"Unknown tranform array strategy: {strategy}")

    # local_id = jax.lax.axis_index("exp")
    local_id = jax.lax.axis_index("tp")
    input_offsets = transform_array(all_shards_group_sizes, local_id, TransformStrategy.INPUT_OFFSET)
    send_sizes = transform_array(all_shards_group_sizes, local_id, TransformStrategy.SEND_SIZE)
    output_offsets = transform_array(all_shards_group_sizes, local_id, TransformStrategy.OUTPUT_OFFSET)
    recv_sizes = transform_array(all_shards_group_sizes, local_id, TransformStrategy.RECV_SIZE)
    return input_offsets, send_sizes, output_offsets, recv_sizes


def local_permute(inputs, global_group_sizes, local_expert_size):
    """Sort inputs by expert within each shard."""
    # local_id = jax.lax.axis_index("exp")
    local_id = jax.lax.axis_index("tp")
    local_sizes = jax.lax.dynamic_slice_in_dim(
        global_group_sizes, local_id * local_expert_size, local_expert_size, axis=1
    ).reshape(-1)
    base_indices = jnp.mod(jnp.arange(local_sizes.shape[0]), local_expert_size)
    expert_indices = jnp.repeat(base_indices, local_sizes, total_repeat_length=inputs.shape[0])
    sorted_indices = jnp.argsort(expert_indices)
    sorted_inputs = jnp.take(inputs, indices=sorted_indices, axis=0)
    group_size = jnp.bincount(expert_indices, length=local_expert_size)
    return sorted_inputs, sorted_indices, group_size


def gmm(inputs, kernel, group_sizes):
        tile_size = (8, 1024, 1024)  # (m, k, n)
        hs_shape = inputs.shape
        # pad length is the 1st dimension of tiling size in gmm call
        pad_length = 8
        if hs_shape[0] % pad_length:
            pad_length = pad_length - hs_shape[0] % pad_length
            inputs = jax.lax.pad(inputs.astype(jnp.float32), 0.0, [(0, pad_length, 0), (0, 0, 0)])

        inputs = inputs.astype(jnp.bfloat16)
        kernel = kernel.astype(jnp.bfloat16)
        lhs_quantize_dtype, rhs_quantize_dtype = None, None
        megablox=True




        if megablox:
            m, k, n = inputs.shape[0], inputs.shape[1], kernel.shape[2]
            print((min(tile_size[0], m), min(tile_size[1], k), min(tile_size[2], n)))
            output = mblx.gmm(
                lhs=inputs,
                rhs=kernel,
                group_sizes=group_sizes,
                preferred_element_type=jnp.bfloat16,
                tiling=(min(tile_size[0], m), min(tile_size[1], k), min(tile_size[2], n)),
                lhs_quantize_dtype=lhs_quantize_dtype,
                rhs_quantize_dtype=rhs_quantize_dtype,
            )
        else:
            output = jax.lax.ragged_dot(
                lhs=inputs,
                rhs=kernel,
                group_sizes=group_sizes,
                preferred_element_type=jnp.bfloat16,
                # preferred_element_type=jnp.float32,
            )
        if hs_shape[0] % pad_length:
            output = output[: hs_shape[0]]
        return output






class Qwen3MLP(LlamaMLP):
    config: Qwen3MoeConfig
    jax_config: Any = None


    def setup(self) -> None:
        dtype=self.jax_config.dtype
        param_dtype=self.jax_config.param_dtype
        self.gate_proj = nn.Dense(self.config.intermediate_size, use_bias=False,dtype=dtype,param_dtype=param_dtype)
        self.up_proj = nn.Dense(self.config.intermediate_size, use_bias=False,dtype=dtype,param_dtype=param_dtype)
        self.down_proj = nn.Dense(self.config.hidden_size, use_bias=False,dtype=dtype,param_dtype=param_dtype)


class Qwen3Attention(LlamaAttention):
    config: Qwen3MoeConfig
    jax_config: Any = None

    def setup(self) -> None:
        dtype=self.jax_config.dtype
        param_dtype=self.jax_config.param_dtype
        config = self.config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.q_proj = nn.Dense(self.num_heads * self.head_dim, use_bias=self.config.attention_bias,dtype=dtype,param_dtype=param_dtype)
        self.k_proj = nn.Dense(self.num_key_value_heads * self.head_dim, use_bias=self.config.attention_bias,dtype=dtype,param_dtype=param_dtype)
        self.v_proj = nn.Dense(self.num_key_value_heads * self.head_dim, use_bias=self.config.attention_bias,dtype=dtype,param_dtype=param_dtype)
        self.o_proj = nn.Dense(self.hidden_size, use_bias=self.config.attention_bias,dtype=dtype,param_dtype=param_dtype)

        self.q_norm = LlamaRMSNorm(eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = LlamaRMSNorm(eps=config.rms_norm_eps)  # thus post q_norm does not need reshape

    def __call__(
            self,
            x: jax.Array,
            input_ids: jax.Array,
            cache: None,
            attn_mask: jax.Array,
            position_embeddings: jax.Array,
    ) -> tuple[  None, jax.Array]:
        bsz, q_len, _ = x.shape

        query_states = self.q_proj(x)
        key_states = self.k_proj(x)
        value_states = self.v_proj(x)

        query_states = einops.rearrange(query_states, 'b n (h d)->b h n  d ', d=self.head_dim)
        key_states = einops.rearrange(key_states, 'b n (h d)->b h n  d ', d=self.head_dim)
        value_states = einops.rearrange(value_states, 'b n (h d)->b h n  d ', d=self.head_dim)

        query_states=self.q_norm(query_states)
        key_states=self.k_norm(key_states)

        dtype = x.dtype
        cos, sin = position_embeddings
        query_states=query_states.astype(jnp.float32)
        key_states = key_states.astype(dtype)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states = query_states.astype(dtype)
        key_states = key_states.astype(dtype)
        # slice_indices = (0, 0, end_index % cache['v'].shape[2], 0)
        # jax.debug.print( "end_index % cache['v'].shape[2]=   {bar}",bar=slice_indices)
        if cache is not None:
            end_index = cache['end_index'][0]
            slice_indices = (0, 0, end_index , 0)
            value_states = jax.lax.dynamic_update_slice(
                cache['v'],
                value_states.astype(cache['v'].dtype),
                slice_indices,
            )
            key_states = jax.lax.dynamic_update_slice(
                cache['k'], key_states.astype(cache['k'].dtype), slice_indices
            )
            new_cache = {
                'v': value_states,
                'k': key_states,
                'end_index': cache['end_index'] + q_len,
            }

            value_states=value_states.astype(dtype)
            key_states=key_states.astype(dtype)

        else:
            new_cache = None

        value_states=repeat_kv(value_states,self.num_key_value_groups)
        key_states=repeat_kv(key_states,self.num_key_value_groups)

        if q_len%128==0 and value_states.shape[-1]%128==0 :
            @functools.partial(
                shard_map,
                mesh=self.jax_config.mesh,
                in_specs=P(['dp','fsdp'],'tp',None,None),
                out_specs=P(['dp','fsdp'],'tp',None,None),
                check_rep=False,
            )
            def wrap_splash_attention(query_states, key_states, value_states):
                mask = splash_attention_mask.CausalMask(shape=(key_states.shape[2], key_states.shape[2]))
                multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(mask,) * value_states.shape[1])

                block_sizes = splash_attention_kernel.BlockSizes(
                    block_q=min(512, query_states.shape[2]),
                    block_kv_compute=min(512, key_states.shape[2]),
                    block_kv=min(512, key_states.shape[2]),
                    block_q_dkv=min(512, query_states.shape[2]),
                    block_kv_dkv=min(512, key_states.shape[2]),
                    block_kv_dkv_compute=min(512, query_states.shape[2]),
                    block_q_dq=min(512, query_states.shape[2]),
                    block_kv_dq=min(512, query_states.shape[2]),
                )

                splash_kernel = splash_attention_kernel.make_splash_mha(
                    mask=multi_head_mask,
                    head_shards=1,
                    q_seq_shards=1,
                    # block_sizes=block_sizes
                )

                attn_output = jax.vmap(splash_kernel)(query_states , key_states, value_states)
                return attn_output

            attn_output=wrap_splash_attention(query_states/ math.sqrt(self.head_dim), key_states, value_states).astype(jnp.bfloat16)

        else:
            attn_weights = (query_states.astype(jnp.float32) @ key_states.swapaxes(2, 3).astype(jnp.float32)) / math.sqrt(self.head_dim)
            if attn_mask is not None:  # no matter the length, we just slice it
                causal_mask = attn_mask
                attn_weights = attn_weights.astype(jnp.float32) + causal_mask

            attn_weights = nn.softmax(attn_weights.astype(jnp.float32), axis=-1, )#.astype(attn_weights.dtype)
            attn_output = attn_weights @ value_states.astype(jnp.float32)
            attn_output=attn_output.astype(dtype)


        attn_output = einops.rearrange(attn_output, 'b h n d-> b n (h d)')
        attn_output = self.o_proj(attn_output)
        return new_cache, attn_output.astype(dtype)





class Qwen3MoeSparseMoeBlock(nn.Module):
    config: Qwen3MoeConfig
    jax_config: Any = None

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
        sorted_inputs = jnp.take(inputs_2d, indices=sorted_indices, axis=0).astype(jnp.bfloat16)
        group_size = jnp.bincount(flatten_selected_experts, length=self.num_experts)
        return sorted_inputs, sorted_selected_experts, routing_weights, group_size

    def unpermute(self, intermediate, sorted_selected_experts, weights, batch_size, sequence_length):
        """Unpermute tokens to original order and combine weights."""

        unsort_intermediate = jnp.take(intermediate, indices=jnp.argsort(sorted_selected_experts), axis=0)
        # print(f'{unsort_intermediate.shape=}')

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
        return output.reshape(batch_size, sequence_length, -1).astype(jnp.bfloat16)

    def get_expert_parallelism_size(self):
        return self.jax_config.mesh.shape["tp"]
        # return self.jax_config.mesh.shape["exp"]

    def get_tensor_parallelism_size(self):
        return self.jax_config.mesh.shape["tp"]

    def __call__(self, hidden_states):
        # print(f'{hidden_states.shape=}')
        batch_size, sequence_length, _ = hidden_states.shape
        # x, sorted_selected_experts, weights, group_sizes = self.permute(hidden_states)

        if self.get_expert_parallelism_size() > 1:
            pass

        input_partition_pspec=P('dp',None)
        # gate_pspec=P("exp", None, 'tp')
        # up_pspec=P("exp", None, 'tp')
        # down_pspec = P("exp", 'tp', None)

        gate_pspec=P("tp", None, None)
        up_pspec=P("tp", None, None)
        down_pspec = P("tp", None, None)

        # out_specs=P('dp',None,'tp')
        out_specs=P('dp',None,None)



        @functools.partial(
            shard_map,
            mesh=self.jax_config.mesh,
            in_specs=(input_partition_pspec,  gate_pspec, up_pspec, down_pspec ),
            out_specs=out_specs,
            check_rep=False,
        )
        def wrapper(hidden_states, gate_proj, up_proj, down_proj  ):

            x, sorted_selected_experts, weights, group_sizes = self.permute(hidden_states)
            if self.get_expert_parallelism_size() > 1:
                # axis_name = "exp"
                axis_name = "tp"
                # get group sizes for all shards
                local_expert_size = self.config.num_experts // self.get_expert_parallelism_size()
                reshaped_group_sizes = jnp.sum(group_sizes.reshape(-1, local_expert_size), axis=1)
                all_shards_group_sizes = jax.lax.all_gather(reshaped_group_sizes, axis_name=axis_name)
                # calculate offsets and sizes for ragged_all_to_all operation
                input_offsets, send_sizes, output_offsets, recv_sizes = get_all_to_all_params(
                    all_shards_group_sizes, local_expert_size, self.get_expert_parallelism_size()
                )
                # TODO(ranran): For better performance, we could update output buffer to a smaller
                # size to replace self.get_expert_parallelism_size() for efficiency,
                # Or we could apply capacity_factor for excessive experts.
                # Note: Reducing buffer increase the risk of token dropping under unbalanced distribution.
                buffer_size = int(
                    self.get_expert_parallelism_size()
                    *1
                    # * self.config.per_device_batch_size
                    # * self.config.max_target_length
                    *sequence_length
                    * self.config.num_experts_per_tok
                )
                output_shape = jnp.zeros((buffer_size, self.config.hidden_size), dtype=x.dtype)
                x = jax.lax.ragged_all_to_all(
                    x,
                    output_shape,
                    input_offsets,
                    send_sizes,
                    output_offsets,
                    recv_sizes,
                    axis_name=axis_name,
                )
                print(f'{x.shape=} {output_shape.shape=}')
                global_group_sizes = jax.lax.all_gather(group_sizes, axis_name=axis_name)
                x, local_sorted_indices, group_sizes = local_permute(x, global_group_sizes, local_expert_size)


            layer_w0 = gmm(x, gate_proj, group_sizes)
            layer_w1 = gmm(x, up_proj, group_sizes)
            layer_act=nn.silu(layer_w0)
            intermediate_layer = jnp.multiply(layer_act, layer_w1)
            intermediate_output = gmm(intermediate_layer, down_proj, group_sizes)

            # if self.get_tensor_parallelism_size() > 1:
            #     intermediate_output = jax.lax.psum_scatter(intermediate_output, "tp", scatter_dimension=1,
            #                                                tiled=True)

            if self.get_expert_parallelism_size() > 1:
                # axis_name = "exp"
                axis_name = "tp"
                # locally unpermute back to the original order
                local_output = jnp.take(intermediate_output, indices=jnp.argsort(local_sorted_indices), axis=0)
                original_inputs_first_dim = batch_size * sequence_length * self.config.num_experts_per_tok
                output_shape = jnp.zeros((original_inputs_first_dim, self.config.hidden_size), dtype=local_output.dtype)
                input_offsets, send_sizes, output_offsets, recv_sizes = get_all_to_all_params(
                    jnp.transpose(all_shards_group_sizes), local_expert_size, self.get_expert_parallelism_size()
                )
                intermediate_output = jax.lax.ragged_all_to_all(
                    local_output,
                    output_shape,
                    input_offsets,
                    send_sizes,
                    output_offsets,
                    recv_sizes,
                    axis_name=axis_name,
                )


            output = self.unpermute(
                intermediate_output, sorted_selected_experts, weights, batch_size=batch_size,
                sequence_length=sequence_length
            )

            print(output.shape)
            return output



        return  wrapper(hidden_states, self.gate_proj, self.up_proj, self.down_proj)





class Qwen3MoeDecoderLayer(LlamaDecoderLayer):
    config: Qwen3MoeConfig
    jax_config: Any = None

    def setup(self) -> None:
        self.hidden_size = self.config.hidden_size
        self.self_attn = Qwen3Attention(config=self.config, jax_config=self.jax_config )
        # self.mlp = Qwen3MLP(self.config, self.jax_config)

        self.mlp = Qwen3MoeSparseMoeBlock(self.config, self.jax_config)

        self.input_layernorm = LlamaRMSNorm(eps=self.config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(eps=self.config.rms_norm_eps)


class Qwen3MoeModel(nn.Module):
    config: Qwen3MoeConfig
    jax_config: Any = None

    def setup(self) -> None:
        dtype=self.jax_config.dtype
        param_dtype=self.jax_config.param_dtype
        self.embed_tokens = nn.Embed(self.config.vocab_size, self.config.hidden_size,dtype=dtype,param_dtype=param_dtype )
        self.layers = [Qwen3MoeDecoderLayer(self.config, self.jax_config) for layer_idx in
                       range(self.config.num_hidden_layers)]
        self.norm = LlamaRMSNorm(eps=self.config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def __call__(
            self,
            input_ids,
            attention_mask,
            position_ids,
            inputs_embeds,
            cache,
    ):

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        b, n, d = inputs_embeds.shape

        if n > 1:

            # mask=attention_mask
            # mask=mask[:,:,None] * mask[:,None,:]
            #
            # attention_mask = jnp.full(
            #     (attention_mask.shape[1], attention_mask.shape[1]), -1e37
            # )
            # attention_mask = jnp.triu(attention_mask, 1)[...]
            # attention_mask=jnp.where(mask,attention_mask,-1e37 )[:,None,...]
            #

            attention_mask = jnp.full(
                (attention_mask.shape[1], attention_mask.shape[1]), -1e37#-0.7 * float(np.finfo(np.dtype("float32")).max)#-1e37
            )
            attention_mask = jnp.triu(attention_mask, 1)[...]

            pass
        else:
            attention_mask = jnp.where(attention_mask, 0, -0.7 * float(np.finfo(np.dtype("float32")).max)#-1e37
                                       )[:,None,None,...]

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        for i, layer in enumerate(self.layers):
            layer_name = f'layer_{i}'
            layer_cache = cache[layer_name] if cache else None
            layer_cache, inputs_embeds = layer(
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                cache=layer_cache,
                attn_mask=attention_mask,
                position_embeddings=position_embeddings
            )
            if cache is not None:
                cache[layer_name] = layer_cache  # pytype: disable=container-type-mismatch

        hidden_states = self.norm(inputs_embeds)

        return hidden_states, cache


class Qwen3MoeForCausalLM(nn.Module):
    config: Qwen3MoeConfig
    jax_config: Any = None

    def setup(self) -> None:
        dtype=self.jax_config.dtype
        param_dtype=self.jax_config.param_dtype
        # self.model = Qwen2Model(self.config, jax_config=self.jax_config)
        # self.lm_head = nn.Dense(self.config.vocab_size, use_bias=False)
        self.model = nn.remat(Qwen3MoeModel)(self.config, jax_config=self.jax_config)
        self.lm_head = nn.remat(nn.Dense)(self.config.vocab_size, use_bias=False,dtype=dtype,param_dtype=param_dtype)



    def __call__(
            self,
            input_ids: jax.Array,  # [B, L]
            position_ids: jax.Array = None,  # [B, L]
            cache=None,  # (sequence length L')
            attention_mask: jax.Array = None,  # [B, L, L']
            inputs_embeds: jax.Array | None = None,
            true_length=None
    ):
        b, n = input_ids.shape

        if position_ids is None:
            position_ids = jnp.arange(n).astype(jnp.int32)[None, :]

        outputs, cache = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            cache=cache
            # past_key_values=past_key_values,
            # use_cache=use_cache,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
            # cache_position=cache_position,
            # **kwargs,
        )

        hidden_states = outputs

        if true_length is not None:
            hidden_states = hidden_states[:, true_length]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states)
        return logits, cache






def convert_torch_to_flax_qwen3_moe(state_dict):
    params = {}
    i = 0
    while f'model.layers.{i}.self_attn.q_proj.weight' in state_dict:
        params[f'model.layers_{i}.input_layernorm.scale'] = state_dict[f'model.layers.{i}.input_layernorm.weight']

        params[f'model.layers_{i}.mlp.gate.kernel'] = state_dict[
            f'model.layers.{i}.mlp.gate.weight'].transpose(1, 0)

        j = 0
        gate_proj_experts = []
        up_proj_experts = []
        down_proj_experts = []
        while f'model.layers.{i}.mlp.experts.{j}.gate_proj.weight' in state_dict:
            gate_proj_experts.append(state_dict[f'model.layers.{i}.mlp.experts.{j}.gate_proj.weight'].transpose(1, 0))
            up_proj_experts.append(state_dict[f'model.layers.{i}.mlp.experts.{j}.up_proj.weight'].transpose(1, 0))
            down_proj_experts.append(state_dict[f'model.layers.{i}.mlp.experts.{j}.down_proj.weight'].transpose(1, 0))
            j += 1

        params[f'model.layers_{i}.mlp.gate_proj'] =  np.asarray(gate_proj_experts)
        params[f'model.layers_{i}.mlp.up_proj'] = np.asarray(up_proj_experts)
        params[f'model.layers_{i}.mlp.down_proj'] = np.asarray(down_proj_experts)


        params[f'model.layers_{i}.post_attention_layernorm.scale'] = state_dict[
            f'model.layers.{i}.post_attention_layernorm.weight']
        params[f'model.layers_{i}.self_attn.q_proj.kernel'] = state_dict[
            f'model.layers.{i}.self_attn.q_proj.weight'].transpose(1, 0)
        params[f'model.layers_{i}.self_attn.k_proj.kernel'] = state_dict[
            f'model.layers.{i}.self_attn.k_proj.weight'].transpose(1, 0)
        params[f'model.layers_{i}.self_attn.v_proj.kernel'] = state_dict[
            f'model.layers.{i}.self_attn.v_proj.weight'].transpose(1, 0)
        params[f'model.layers_{i}.self_attn.o_proj.kernel'] = state_dict[
            f'model.layers.{i}.self_attn.o_proj.weight'].transpose(1, 0)

        if f'model.layers.{i}.self_attn.q_proj.bias' in state_dict:
            params[f'model.layers_{i}.self_attn.q_proj.bias'] = state_dict[
                f'model.layers.{i}.self_attn.q_proj.bias']
            params[f'model.layers_{i}.self_attn.k_proj.bias'] = state_dict[
                f'model.layers.{i}.self_attn.k_proj.bias']
            params[f'model.layers_{i}.self_attn.v_proj.bias'] = state_dict[
                f'model.layers.{i}.self_attn.v_proj.bias']

        if f'model.layers.{i}.self_attn.q_norm.weight' in state_dict:
            params[f'model.layers_{i}.self_attn.q_norm.scale'] = state_dict[
                f'model.layers.{i}.self_attn.q_norm.weight']
            params[f'model.layers_{i}.self_attn.k_norm.scale'] = state_dict[
                f'model.layers.{i}.self_attn.k_norm.weight']


        i += 1
    params['model.norm.scale'] = state_dict['model.norm.weight']
    params['model.embed_tokens.embedding'] = state_dict['model.embed_tokens.weight']  #.transpose(1,0)
    # params['lm_head.kernel'] = params['model.embed_tokens.embedding'].transpose(1, 0)
    params['lm_head.kernel'] = state_dict['lm_head.weight'].transpose(1, 0)

    return flax.traverse_util.unflatten_dict(params, sep='.')
