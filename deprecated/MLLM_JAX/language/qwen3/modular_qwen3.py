import functools
import math
from typing import Any

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel, splash_attention_mask
from jax.experimental.shard_map import shard_map

from .configuration_qwen3 import Qwen3Config
from ..llama.llama import LlamaMLP, LlamaAttention, LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding, \
    apply_rotary_pos_emb, repeat_kv
from jax.sharding import PartitionSpec as P

class Qwen3MLP(LlamaMLP):
    config: Qwen3Config
    jax_config: Any = None


    def setup(self) -> None:
        dtype=self.jax_config.dtype
        param_dtype=self.jax_config.param_dtype
        self.gate_proj = nn.Dense(self.config.intermediate_size, use_bias=False,dtype=dtype,param_dtype=param_dtype)
        self.up_proj = nn.Dense(self.config.intermediate_size, use_bias=False,dtype=dtype,param_dtype=param_dtype)
        self.down_proj = nn.Dense(self.config.hidden_size, use_bias=False,dtype=dtype,param_dtype=param_dtype)


class Qwen3Attention(LlamaAttention):
    config: Qwen3Config
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

                # block_sizes = splash_attention_kernel.BlockSizes(
                #     block_q=min(512, query_states.shape[2]),
                #     block_kv_compute=min(512, key_states.shape[2]),
                #     block_kv=min(512, key_states.shape[2]),
                #     block_q_dkv=min(512, query_states.shape[2]),
                #     block_kv_dkv=min(512, key_states.shape[2]),
                #     block_kv_dkv_compute=min(512, query_states.shape[2]),
                #     block_q_dq=min(512, query_states.shape[2]),
                #     block_kv_dq=min(512, query_states.shape[2]),
                # )

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
                    block_sizes=block_sizes
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


class Qwen3DecoderLayer(LlamaDecoderLayer):
    config: Qwen3Config
    jax_config: Any = None

    def setup(self) -> None:
        self.hidden_size = self.config.hidden_size
        self.self_attn = Qwen3Attention(config=self.config, jax_config=self.jax_config )
        self.mlp = Qwen3MLP(self.config, self.jax_config)
        self.input_layernorm = LlamaRMSNorm(eps=self.config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(eps=self.config.rms_norm_eps)


class Qwen3Model(nn.Module):
    config: Qwen3Config
    jax_config: Any = None

    def setup(self) -> None:
        dtype=self.jax_config.dtype
        param_dtype=self.jax_config.param_dtype
        self.embed_tokens = nn.Embed(self.config.vocab_size, self.config.hidden_size,dtype=dtype,param_dtype=param_dtype )
        self.layers = [Qwen3DecoderLayer(self.config, self.jax_config) for layer_idx in
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


class Qwen3ForCausalLM(nn.Module):
    config: Qwen3Config
    jax_config: Any = None

    def setup(self) -> None:
        dtype=self.jax_config.dtype
        param_dtype=self.jax_config.param_dtype
        # self.model = Qwen2Model(self.config, jax_config=self.jax_config)
        # self.lm_head = nn.Dense(self.config.vocab_size, use_bias=False)
        self.model = nn.remat(Qwen3Model)(self.config, jax_config=self.jax_config)
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
