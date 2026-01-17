import copy
import functools
import math

import einops
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel, splash_attention_mask, \
    MultiHeadMask, NumpyMask
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh,PartitionSpec as PS
from jax.sharding import Mesh,PartitionSpec,NamedSharding
from MLLM_JAX.efficient2 import flash_attention, attention_math
from MLLM_JAX.language.gemma3.configuration_gemma3 import Gemma3TextConfig
from MLLM_JAX.language.llama.llama import LlamaJaxConfig, _compute_default_rope_parameters
from MLLM_JAX.mask import make_causal_bidirectional_attention_mask

# from .configuration_gemma3 import Gemma3TextConfig
# from ..llama.llama import _compute_default_rope_parameters, LlamaJaxConfig

K_MASK = -2.3819763e38  # Set to a large negative number.
LayerCache = dict[str, jax.Array]

Cache = dict[str, LayerCache]



def _create_sliding_mask(
    segment_pos: jnp.ndarray,
    end_index: int,
    cache_len: int,
    sliding_window_size: int,
):
  """Creates mask for sliding window attention."""
  total_tokens = end_index + segment_pos.shape[1]  # cached + processing tokens

  def _reconstruct_rotated_cache_positions():
    cache_positions = jnp.arange(cache_len) + total_tokens - cache_len
    cache_positions = (
        jnp.zeros_like(cache_positions)
        # kv were placed at index (position_id % cache_len) in the cache.
        .at[cache_positions % cache_len].set(cache_positions)
    )
    return cache_positions

  # Reconstruct position_ids for cached kv.
  cache_positions = jax.lax.cond(
      total_tokens <= cache_len,
      lambda: jnp.arange(cache_len),
      _reconstruct_rotated_cache_positions,
  )

  cache_positions = cache_positions[None, None, :]  # [1, 1, cache_len]
  segment_pos = segment_pos[:, :, None]  # [B, seq_len, 1]
  sliding_mask = cache_positions > segment_pos - sliding_window_size
  sliding_mask *= cache_positions < segment_pos + sliding_window_size
  return sliding_mask



class Gemma3MLP(nn.Module):
    config: Gemma3TextConfig
    jax_config: LlamaJaxConfig = None
    def setup(self) -> None:
        self.hidden_size = self.config.hidden_size
        self.intermediate_size = self.config.intermediate_size
        self.gate_proj = nn.Dense(self.intermediate_size, use_bias=False)
        self.up_proj = nn.Dense(self.intermediate_size, use_bias=False)
        self.down_proj = nn.Dense(self.hidden_size, use_bias=False)

    def __call__(self, x):
        x=nn.gelu(self.gate_proj(x),approximate=True) * self.up_proj(x)
        x = self.down_proj(x)
        return x


class Gemma3RMSNorm(nn.Module):
    """RMSNorm layer."""
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        scale = self.param('scale', nn.initializers.zeros_init(), (x.shape[-1]))
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        # Jax.lax.rsqrt is used because it returns different floats than
        # jnp.reciprocal(jnp.sqrt(var + 1e-06))
        normed_inputs = x * jax.lax.rsqrt(var + 1e-06)

        # normed_inputs is a rank-K tensor, K > 1 (K is typically 2 or 3). scale is
        # a rank-1 tensor. To avoid implicit rank-promotion, reshape scale to
        # a (1, ..., 1, D) tensor, so the rank of scale matches normed_inputs.
        scale = jnp.expand_dims(scale, axis=range(len(x.shape) - 1))
        normed_inputs = normed_inputs * (1 + scale)
        return normed_inputs






def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
          cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    """
    cos = jnp.expand_dims(cos, unsqueeze_dim)
    sin = jnp.expand_dims(sin, unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states:jax.Array, n_rep):
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    hidden_states = hidden_states[:, :, None, :, :]
    hidden_states=jnp.broadcast_to(hidden_states,(batch, num_key_value_heads, n_rep, slen, head_dim))
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)




class Gemma3Attention(nn.Module):
    """Attention module."""
    config: Gemma3TextConfig
    jax_config: LlamaJaxConfig = None
    is_sliding :bool=False

    def setup(self) -> None:
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

        self.q_proj = nn.Dense(self.num_heads * self.head_dim, use_bias=config.attention_bias,
                               #kernel_init=None if self.jax_config is None else nn.with_logical_partitioning(self.jax_config.dense_init, ('embed', 'hidden')),
                               )


        self.k_proj = nn.Dense(self.num_key_value_heads * self.head_dim, use_bias=config.attention_bias)
        self.v_proj = nn.Dense(self.num_key_value_heads * self.head_dim, use_bias=config.attention_bias)
        self.o_proj = nn.Dense(self.hidden_size, use_bias=config.attention_bias)


        self.q_norm = Gemma3RMSNorm(eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(eps=config.rms_norm_eps)


    def __call__(
            self,
            x:  jnp.ndarray,
            input_ids:  jnp.ndarray,
            cache:   None,
            attn_mask: jnp.ndarray,
            position_embeddings:  jnp.ndarray,
            segment_pos: jnp.ndarray,
            special_image_mask=None,
    ) -> tuple[  None, jax.Array]:
        bsz, q_len, _ = x.shape

        if q_len > 1:
            attn_mask = make_causal_bidirectional_attention_mask(attn_mask.astype(jnp.bool),
                                                                 bidirectional_mask=special_image_mask)[:, None, :, :]
            # attention_mask=jnp.where(attention_mask,0,-1e37)#[:,None,:,:]
        else:
            attn_mask = jnp.where(attn_mask, True, False)[:, None, None, ...]




        query_states = self.q_proj(x)
        key_states = self.k_proj(x)
        value_states = self.v_proj(x)

        query_states = einops.rearrange(query_states, 'b n (h d)->b h n  d ', d=self.head_dim)
        key_states = einops.rearrange(key_states, 'b n (h d)->b h n  d ', d=self.head_dim)
        value_states = einops.rearrange(value_states, 'b n (h d)->b h n  d ', d=self.head_dim)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        dtype = query_states.dtype
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        query_states = query_states.astype(dtype)
        key_states = key_states.astype(dtype)
        # slice_indices = (0, 0, end_index % cache['v'].shape[2], 0)
        if cache is not None and cache['v'] is not None:
            end_index = cache['end_index'][0]
            slice_indices = (0, 0, end_index % cache['v'].shape[2] , 0)

            value_states = jax.lax.dynamic_update_slice(
                cache['v'],
                value_states,
                slice_indices,
            )
            key_states = jax.lax.dynamic_update_slice(
                cache['k'], key_states, slice_indices
            )
            new_cache = {
                'v': value_states,
                'k': key_states,
                'end_index': cache['end_index'] + q_len,
            }

        else:
            new_cache = None

        value_states=repeat_kv(value_states,self.num_key_value_groups)
        key_states=repeat_kv(key_states,self.num_key_value_groups)


        if q_len%128==0 and value_states.shape[-1]%128==0 and self.jax_config is not None :

            if self.is_sliding:
                sliding_mask = _create_sliding_mask(
                    segment_pos,
                    end_index=cache['end_index'][0] if cache is not None else 0,
                    # Derive cache length from attn_mask shape in case cache is None
                    cache_len=attn_mask.shape[-1],
                    sliding_window_size=self.config.sliding_window,
                )[None, ...]
                attn_mask = jnp.where(sliding_mask, attn_mask, False)

            attn_output=attention_math(query_states/ math.sqrt(self.head_dim),key_states,value_states,attn_mask)

            # attn_mask = jnp.where(attn_mask==1, 0, -1e37)

            # attn_mask=einops.rearrange(attn_mask,'b h n d -> (b h ) n d').astype(jnp.bool)
            # query_states=einops.rearrange(query_states/ math.sqrt(self.head_dim),'b h n d -> (b h ) n d')#.astype(jnp.float32)
            # key_states=einops.rearrange(key_states,'b h n d -> (b h ) n d')#.astype(jnp.float32)
            # value_states=einops.rearrange(value_states,'b h n d -> (b h ) n d')#.astype(jnp.float32)
            #
            # # query_states=jax.lax.with_sharding_constraint(query_states,NamedSharding(self.jax_config.mesh,P('tp')))
            # # key_states=jax.lax.with_sharding_constraint(key_states, NamedSharding(self.jax_config.mesh, P('tp')))
            # # value_states=jax.lax.with_sharding_constraint(value_states, NamedSharding(self.jax_config.mesh, P('tp')))
            # #
            # attn_output = flash_attention(
            #     query_states, key_states, value_states, attn_mask)
            #
            # # attn_output =query_states
            # flash_attn=splash_attention_kernel.make_splash_mha(mask=attn_mask,head_shards=1,
            #             q_seq_shards=1,)
            # # attn_mask=jnp.repeat(attn_mask,4,axis=0)
            # # flash_attention_kernel=functools.partial(flash_attention,)
            # # attn_output=shard_map(flash_attention_kernel,  self.jax_config.mesh,
            # #                       in_specs=(P('tp',None,None),P('tp',None,None),P('tp',None,None),P('tp',None,None)),
            # #                       out_specs=P('tp',None,None),
            # #                         check_rep=False,)(query_states,key_states,value_states,attn_mask)
            # attn_output = einops.rearrange(attn_output, '(b h ) n d->b h n d',b=bsz)
            # jax.debug.inspect_array_sharding(attn_output, callback=print)
        else:
            attn_weights = (query_states @ key_states.swapaxes(2, 3)) / math.sqrt(self.head_dim)
            if attn_mask is not None:
                causal_mask = jnp.where(attn_mask == True, 0, -1e37)

                if self.is_sliding:
                    sliding_mask = _create_sliding_mask(
                        segment_pos,
                        end_index=cache['end_index'][0] if cache is not None else 0,
                        # Derive cache length from attn_mask shape in case cache is None
                        cache_len=attn_mask.shape[-1],
                        sliding_window_size=self.config.sliding_window,
                    )[None, ...]
                    causal_mask = jnp.where(sliding_mask, causal_mask, -1e37)
                attn_weights = attn_weights + causal_mask
            attn_weights = nn.softmax(attn_weights.astype(jnp.float32), axis=-1, ).astype(attn_weights.dtype)
            attn_output = attn_weights @ value_states


        attn_output = einops.rearrange(attn_output, 'b h n d-> b n (h d)')
        attn_output = self.o_proj(attn_output)

        return new_cache, attn_output

    @classmethod
    def init_cache(
            cls,
            cache_size: int,
            num_heads: int,
            head_dim: int,
            batch_size: int,
            dtype: jnp.dtype = jnp.bfloat16,
            shard_method=None
    ):
        del cls  # not used

        cache={
            'v': jnp.zeros(
                (batch_size, num_heads, cache_size, head_dim), dtype=dtype
            ),
            'k': jnp.zeros(
                (batch_size, num_heads, cache_size, head_dim), dtype=dtype
            ),
            # 'v': None,
            # 'k': None,
            'end_index': jnp.zeros((batch_size,), dtype=jnp.int32),
        }
        if shard_method and cache['v'] is not None:
            cache['v']=shard_method(cache['v'])
            cache['k'] = shard_method(cache['k'])

        return cache














class LlamaRotaryEmbedding(nn.Module):
    # dim: int = None
    # max_position_embeddings: int = 2048,
    # base: int = 10000,
    # scaling_factor: float = 1.0,
    # rope_type: str = "default",
    config: Gemma3TextConfig = None

    def setup(self) -> None:
        inv_freq, self.attention_scaling = _compute_default_rope_parameters(self.config, )
        # inv_freq, self.attention_scaling = _compute_llama3_parameters(self.config, )
        self.inv_freq = inv_freq

    def __call__(self, x, position_ids):
        inv_freq_expanded = jnp.tile(self.inv_freq[None, :, None], (position_ids.shape[0], 1, 1))
        # inv_freq_expanded = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].astype(jnp.float32)
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        freqs = (inv_freq_expanded.astype(jnp.float32) @ position_ids_expanded.astype(jnp.float32)).swapaxes(1, 2)
        emb = jnp.concatenate((freqs, freqs), axis=-1)

        cos = jnp.cos(emb)
        sin = jnp.sin(emb)
        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling
        return cos, sin



def _compute_default_rope_parameters(
        config=None,
        **rope_kwargs,
):
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    elif config is not None:
        base = config.rope_theta
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.int32).astype(jnp.float32) / dim))
    return inv_freq, attention_factor









def _compute_linear_scaling_rope_parameters(
    config= None,
    # seq_len: Optional[int] = None,
    **rope_kwargs,
):
    """
    Computes the inverse frequencies with linear scaling. Credits to the Reddit user /u/kaiokendev
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
            f"`_compute_linear_scaling_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
        )
    if len(rope_kwargs) > 0:
        factor = rope_kwargs["factor"]
    elif config is not None:
        factor = config.rope_scaling["factor"]

    # Gets the default RoPE parameters
    inv_freq, attention_factor = _compute_default_rope_parameters(config, **rope_kwargs)

    # Then applies linear scaling to the frequencies.
    # NOTE: originally, scaling was applied to the position_ids. However, we get `embs = inv_freq @ position_ids`, so
    # applying scaling to the inverse frequencies is equivalent.
    inv_freq /= factor
    return inv_freq, attention_factor



class Gemma3DecoderLayer(nn.Module):
    config: Gemma3TextConfig
    jax_config: LlamaJaxConfig = None
    is_sliding:bool=False

    def setup(self) -> None:
        self.hidden_size = self.config.hidden_size
        # self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.self_attn = Gemma3Attention(config=self.config,is_sliding=self.is_sliding ,jax_config=self.jax_config)
        self.mlp = Gemma3MLP(self.config)
        self.input_layernorm = Gemma3RMSNorm(eps=self.config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(eps=self.config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(eps=self.config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(eps=self.config.rms_norm_eps)

    def __call__(
            self,
            inputs_embeds: jax.Array,
            input_ids: jax.Array,
            cache: None,
            attn_mask: jax.Array,
            position_embeddings_local: jax.Array,
            position_embeddings_global: jax.Array,
            segment_pos:jax.Array,
            special_image_mask=None
    ) -> tuple[  None, jax.Array]:
        inputs_normalized = self.input_layernorm(inputs_embeds)

        if self.self_attn.is_sliding:
            position_embeddings = position_embeddings_local
        else:
            position_embeddings = position_embeddings_global

        cache, attn_output = self.self_attn(
            x=inputs_normalized,
            position_embeddings=position_embeddings,
            cache=cache,
            attn_mask=attn_mask,
            input_ids=input_ids,
            segment_pos=segment_pos,
            special_image_mask=special_image_mask

        )
        attn_output = self.post_attention_layernorm(attn_output)
        attn_output += inputs_embeds
        outputs = self.pre_feedforward_layernorm(attn_output)
        outputs = self.mlp(outputs)
        outputs = self.post_feedforward_layernorm(outputs)
        outputs += attn_output
        return cache, outputs


class Gemma3RotaryEmbedding(nn.Module):
    # dim: int = None
    # max_position_embeddings: int = 2048,
    # base: int = 10000,
    # scaling_factor: float = 1.0,
    # rope_type: str = "default",
    config: Gemma3TextConfig = None

    def setup(self) -> None:

        if hasattr(self.config, "rope_scaling") and self.config.rope_scaling is not None:
            self.rope_type = self.config.rope_scaling.get("rope_type", self.config.rope_scaling.get("type"))

            if self.rope_type!="default":

                inv_freq, self.attention_scaling = _compute_linear_scaling_rope_parameters(self.config, )
            else:
                inv_freq, self.attention_scaling = _compute_default_rope_parameters(self.config, )
        else:
            self.rope_type = "default"
            inv_freq, self.attention_scaling = _compute_default_rope_parameters(self.config, )



        self.inv_freq = inv_freq

    def __call__(self, x, position_ids):
        inv_freq_expanded = jnp.tile(self.inv_freq[None, :, None], (position_ids.shape[0], 1, 1))
        # inv_freq_expanded = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].astype(jnp.float32)
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        freqs = (inv_freq_expanded.astype(jnp.float32) @ position_ids_expanded.astype(jnp.float32)).swapaxes(1, 2)
        emb = jnp.concatenate((freqs, freqs), axis=-1)

        cos = jnp.cos(emb)
        sin = jnp.sin(emb)
        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling
        return cos, sin



class Gemma3TextModel(nn.Module):
    config: Gemma3TextConfig
    jax_config: LlamaJaxConfig=None

    def setup(self) -> None:
        self.embed_tokens = nn.Embed(self.config.vocab_size, self.config.hidden_size, )
        self.layers = [Gemma3DecoderLayer(self.config,  jax_config=self.jax_config, is_sliding=bool((layer_idx + 1) % self.config.sliding_window_pattern  , ))
                       for layer_idx in range(self.config.num_hidden_layers)]
        self.norm = Gemma3RMSNorm(eps=self.config.rms_norm_eps)

        self.rotary_emb = Gemma3RotaryEmbedding(config=self.config)

        config = copy.deepcopy(self.config)
        config.rope_theta = config.rope_local_base_freq
        config.rope_scaling = {"rope_type": "default"}
        self.rotary_emb_local = Gemma3RotaryEmbedding(config=config)


    def embedding_inputs_ids(self,input_ids):
        return self.embed_tokens(input_ids)*self.config.hidden_size**0.5



    def __call__(
            self,
            input_ids,
            attention_mask,
            position_ids,
            inputs_embeds,
            cache,
            special_image_mask=None
    ):

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)*self.config.hidden_size**0.5

        b, n, d = inputs_embeds.shape
        # causal_mask = self._update_causal_mask(
        #     attention_mask, inputs_embeds,
        # )

        # if  len(attention_mask.shape)==4:
        #     # print(attention_mask.shape,attention_mask.dtype)
        #     pass
        # else:
        #     if n > 1:
        #         attention_mask = jnp.full(
        #             (attention_mask.shape[1], attention_mask.shape[1]), -1e37
        #         )
        #         attention_mask = jnp.triu(attention_mask, 1)[...]
        #     else:
        #         attention_mask = jnp.where(attention_mask, 0, -1e37)[:, None, None, ...]


        # position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        position_embeddings_global = self.rotary_emb(inputs_embeds, position_ids)
        position_embeddings_local = self.rotary_emb_local(inputs_embeds, position_ids)


        for i, layer in enumerate(self.layers):
            layer_name = f'layer_{i}'
            layer_cache = cache[layer_name] if cache else None
            layer_cache, inputs_embeds = layer(
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                cache=layer_cache,
                attn_mask=attention_mask,
                position_embeddings_local=position_embeddings_local,
                position_embeddings_global=position_embeddings_global,
                segment_pos=position_ids,
                special_image_mask=special_image_mask
            )
            if cache is not None:
                cache[layer_name] = layer_cache  # pytype: disable=container-type-mismatch

        hidden_states = self.norm(inputs_embeds)

        return hidden_states, cache








class GemmaForCausalLM(nn.Module):
    config: Gemma3TextConfig
    jax_config: LlamaJaxConfig=None

    def setup(self) -> None:
        self.model = Gemma3TextModel(self.config,jax_config=self.jax_config)
        # self.vocab_size = config.vocab_size
        self.lm_head = nn.Dense(self.config.vocab_size, use_bias=False)

    def __call__(
            self,
            input_ids: jax.Array,  # [B, L]
            position_ids: jax.Array=None,  # [B, L]
            cache: Cache | None = None,  # (sequence length L')
            attention_mask: jax.Array = None,  # [B, L, L']
            inputs_embeds: jax.Array | None = None,
            special_image_mask=None,
            true_length=None
    ) -> tuple[jax.Array, Cache | None]:

        b,n=input_ids.shape

        if position_ids is None:
            position_ids = jnp.arange(n).astype(jnp.int32)[None, :]



        outputs, cache = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            cache=cache,
            special_image_mask=special_image_mask
            # past_key_values=past_key_values,
            # use_cache=use_cache,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
            # cache_position=cache_position,
            # **kwargs,
        )

        hidden_states = outputs
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        if true_length is not None:
            hidden_states=hidden_states[:,true_length]
            # if cache is not None:
            #     for i in range(len(cache)):
            #         cache[f'layer_{i}']['end_index'] = jnp.zeros_like(cache[f'layer_{i}']['end_index'],dtype=jnp.int32) + true_length

        logits = self.lm_head(hidden_states)
        print(logits.shape)
        return logits, cache




def convert_torch_to_flax_gemma3_text(state_dict):
    params = {}
    i = 0
    while f'model.layers.{i}.self_attn.q_proj.weight' in state_dict:
        params[f'model.layers_{i}.input_layernorm.scale'] = state_dict[f'model.layers.{i}.input_layernorm.weight']
        params[f'model.layers_{i}.post_attention_layernorm.scale'] = state_dict[f'model.layers.{i}.post_attention_layernorm.weight']
        params[f'model.layers_{i}.pre_feedforward_layernorm.scale'] = state_dict[f'model.layers.{i}.pre_feedforward_layernorm.weight']
        params[f'model.layers_{i}.post_feedforward_layernorm.scale'] = state_dict[f'model.layers.{i}.post_feedforward_layernorm.weight']

        params[f'model.layers_{i}.mlp.down_proj.kernel'] = state_dict[
            f'model.layers.{i}.mlp.down_proj.weight'].transpose(1, 0)
        params[f'model.layers_{i}.mlp.gate_proj.kernel'] = state_dict[
            f'model.layers.{i}.mlp.gate_proj.weight'].transpose(1, 0)
        params[f'model.layers_{i}.mlp.up_proj.kernel'] = state_dict[f'model.layers.{i}.mlp.up_proj.weight'].transpose(1,
                                                                                                                      0)

        params[f'model.layers_{i}.self_attn.q_proj.kernel'] = state_dict[
            f'model.layers.{i}.self_attn.q_proj.weight'].transpose(1, 0)
        params[f'model.layers_{i}.self_attn.k_proj.kernel'] = state_dict[
            f'model.layers.{i}.self_attn.k_proj.weight'].transpose(1, 0)
        params[f'model.layers_{i}.self_attn.v_proj.kernel'] = state_dict[
            f'model.layers.{i}.self_attn.v_proj.weight'].transpose(1, 0)
        params[f'model.layers_{i}.self_attn.o_proj.kernel'] = state_dict[
            f'model.layers.{i}.self_attn.o_proj.weight'].transpose(1, 0)

        params[f'model.layers_{i}.self_attn.q_norm.scale'] = state_dict[
            f'model.layers.{i}.self_attn.q_norm.weight']

        params[f'model.layers_{i}.self_attn.k_norm.scale'] = state_dict[
            f'model.layers.{i}.self_attn.k_norm.weight']

        if f'model.layers.{i}.self_attn.q_proj.bias' in state_dict:
            params[f'model.layers_{i}.self_attn.q_proj.bias'] = state_dict[
                f'model.layers.{i}.self_attn.q_proj.bias']
            params[f'model.layers_{i}.self_attn.k_proj.bias'] = state_dict[
                f'model.layers.{i}.self_attn.k_proj.bias']
            params[f'model.layers_{i}.self_attn.v_proj.bias'] = state_dict[
                f'model.layers.{i}.self_attn.v_proj.bias']

        i += 1
    params['model.norm.scale'] = state_dict['model.norm.weight']
    params['model.embed_tokens.embedding'] = state_dict['model.embed_tokens.weight']  #.transpose(1,0)
    # params['lm_head.kernel'] = params['model.embed_tokens.embedding'].transpose(1, 0)
    params['lm_head.kernel'] = state_dict['lm_head.weight'].transpose(1, 0)

    return flax.traverse_util.unflatten_dict(params, sep='.')


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
        ('lm_head/kernel', PS('tp','fsdp', )),

        ('scale',PS('tp')),

        ('.*', PS(None)),
    )



if __name__=='__main__':
    pass
    # from transformers.models.gemma3.modeling_gemma3 import
    # model_torch=

    # model=GemmaForCausalLM()