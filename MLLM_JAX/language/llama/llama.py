import dataclasses
import functools
import math
from typing import Any

import einops
import flax
import jax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
import torch
from flax.linen.spmd import RulesFallback
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask, splash_attention_kernel

from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh,PartitionSpec,NamedSharding
from tensorflow.python.framework.tensor import DenseSpec
from tqdm import tqdm

from jax.sharding import PartitionSpec as P


K_MASK = -2.3819763e38  # Set to a large negative number.
LayerCache = dict[str, jax.Array]

Cache = dict[str, LayerCache]



class LlamaConfig:
    r"""
    This is the configuration class to store the configuration of a [`LlamaModel`]. It is used to instantiate an LLaMA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LLaMA-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LlamaModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Llama 1 supports up to 2048 tokens,
            Llama 2 up to 4096, CodeLlama up to 16384.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/main/perf_train_gpu_many#tensor-parallelism) to
            understand more about it. This value is necessary to ensure exact reproducibility of the pretraining
            results. Please refer to [this issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.
        head_dim (`int`, *optional*):
            The attention head dimension. If None, it will default to hidden_size // num_heads

    ```python
    >>> from transformers import LlamaModel, LlamaConfig

    >>> # Initializing a LLaMA llama-7b style configuration
    >>> configuration = LlamaConfig()

    >>> # Initializing a model from the llama-7b style configuration
    >>> model = LlamaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "llama"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `LlamaModel`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    def __init__(
            self,
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=None,
            hidden_act="silu",
            max_position_embeddings=2048,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            pad_token_id=None,
            bos_token_id=1,
            eos_token_id=2,
            pretraining_tp=1,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias=False,
            attention_dropout=0.0,
            mlp_bias=False,
            head_dim=None,
            max_cache_length=1024,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.max_cache_length = max_cache_length
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.

    def init_cache(
            self,
            batch_size: int,
            max_cache_length:int=1024,
            dtype: jnp.dtype = jnp.bfloat16,
    ) -> Cache:
        """Initializes a new Transformer cache."""
        if self.max_cache_length is None:
            raise ValueError('max_cache_length must be set to initialize cache.')
        cache = {
            f'layer_{i}': LlamaAttention.init_cache(
                self.max_cache_length,
                self.num_key_value_heads,
                self.head_dim,
                batch_size,
                dtype,
            )
            for i in range(self.num_hidden_layers)
        }
        return cache


    @classmethod
    def llama3_8b(cls, ):
        return cls(
            num_key_value_heads=8,
            intermediate_size=14336,
            max_position_embeddings=131072,
            rope_scaling={
                "factor": 8.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3"
            },
            rope_theta=500000.0,
        )





@dataclasses.dataclass
class LlamaJaxConfig:
    dense_init:Any =nn.initializers.truncated_normal()
    mesh: Any = None




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









def _compute_llama3_parameters(
    config, **rope_kwargs
) :
    """
    Computes the inverse frequencies for llama 3.1.

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
        post-processing scaling factor applied to the computed cos/sin.
    """
    # Gets the default RoPE parameters
    inv_freq, attention_factor = _compute_default_rope_parameters(config, **rope_kwargs)

    factor = config.rope_scaling["factor"]  # `8` in the original implementation
    low_freq_factor = config.rope_scaling["low_freq_factor"]  # `1` in the original implementation
    high_freq_factor = config.rope_scaling["high_freq_factor"]  # `4` in the original implementation
    old_context_len = config.rope_scaling["original_max_position_embeddings"]  # `8192` in the original implementation

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = jnp.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = jnp.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama, attention_factor




class LlamaRotaryEmbedding(nn.Module):
    # dim: int = None
    # max_position_embeddings: int = 2048,
    # base: int = 10000,
    # scaling_factor: float = 1.0,
    # rope_type: str = "default",
    config: LlamaConfig = None

    def setup(self) -> None:
        inv_freq, self.attention_scaling = _compute_default_rope_parameters(self.config, )
        # inv_freq, self.attention_scaling = _compute_llama3_parameters(self.config, )
        self.inv_freq = inv_freq

    def __call__(self, x, position_ids):
        # print(self.inv_freq[None, :, None].shape)
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


class LlamaRMSNorm(nn.Module):
    """RMSNorm layer."""
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        scale = self.param('scale', nn.initializers.zeros_init(), (x.shape[-1]))
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)

        # Jax.lax.rsqrt is used because it returns different floats than
        # jnp.reciprocal(jnp.sqrt(var + 1e-06))
        normed_inputs = x * jax.lax.rsqrt(var + self.eps)

        # normed_inputs is a rank-K tensor, K > 1 (K is typically 2 or 3). scale is
        # a rank-1 tensor. To avoid implicit rank-promotion, reshape scale to
        # a (1, ..., 1, D) tensor, so the rank of scale matches normed_inputs.
        scale = jnp.expand_dims(scale, axis=range(len(x.shape) - 1))
        normed_inputs = normed_inputs * scale
        return normed_inputs


class LlamaMLP(nn.Module):
    config: LlamaConfig
    jax_config: LlamaJaxConfig = None

    def setup(self) -> None:
        self.hidden_size = self.config.hidden_size
        self.intermediate_size = self.config.intermediate_size
        self.gate_proj = nn.Dense(self.intermediate_size, use_bias=self.config.mlp_bias)
        self.up_proj = nn.Dense(self.intermediate_size, use_bias=self.config.mlp_bias)
        self.down_proj = nn.Dense(self.hidden_size, use_bias=self.config.mlp_bias)

    def __call__(self, x):
        x=nn.silu(self.gate_proj(x)) * self.up_proj(x)
        # x = nn.with_logical_constraint(x, ("dp", "fsdp", "tp"))
        # x = nn.with_logical_constraint(x, ("dp",  "tp","fsdp"))
        #
        #
        # if isinstance(x,jax._src.interpreters.partial_eval.DynamicJaxprTracer):
        #     # jax.debug.visualize_array_sharding(x[0])
        #     # print(x.shape)
        #     # print(x.sharding)
        #     jax.debug.inspect_array_sharding(x,callback=print)
        # else:
        #     print(type(x))
        if self.jax_config is not None:
            pass
            # print('sharding !')
            # x = nn.with_logical_constraint(x, ('batch',None, 'embed',),fallback=RulesFallback.RAISE_ERROR)
            # x=jax.lax.with_sharding_constraint(x, NamedSharding(self.jax_config.mesh,PartitionSpec('dp', 'fsdp','tp')))
            # x = nn.with_logical_constraint(x, (None, None,None ))

        x = self.down_proj(x)
        return x


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



class LlamaAttention(nn.Module):
    """Attention module."""
    config: LlamaConfig
    jax_config: LlamaJaxConfig = None

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

    def __call__(
            self,
            x: jax.Array,
            input_ids: jax.Array,
            cache: LayerCache | None,
            attn_mask: jax.Array,
            position_embeddings: jax.Array,
    ) -> tuple[LayerCache | None, jax.Array]:
        bsz, q_len, _ = x.shape

        query_states = self.q_proj(x)
        key_states = self.k_proj(x)
        value_states = self.v_proj(x)

        query_states = einops.rearrange(query_states, 'b n (h d)->b h n  d ', d=self.head_dim)
        key_states = einops.rearrange(key_states, 'b n (h d)->b h n  d ', d=self.head_dim)
        value_states = einops.rearrange(value_states, 'b n (h d)->b h n  d ', d=self.head_dim)

        """
        if self.jax_config is not None:
            print('sharding attention !')
            query_states=jax.lax.with_sharding_constraint(query_states, NamedSharding(self.jax_config.mesh,PartitionSpec('dp','tp', 'fsdp',None)))
            key_states = jax.lax.with_sharding_constraint(key_states, NamedSharding(self.jax_config.mesh,
                                                                                        PartitionSpec('dp', 'tp',
                                                                                                      'fsdp', None)))
            # value_states = jax.lax.with_sharding_constraint(value_states, NamedSharding(self.jax_config.mesh,
            #                                                                             PartitionSpec('dp', 'tp',
            #                                                                                           'fsdp', None)))


            value_states=nn.with_logical_constraint(value_states,('batch','embed',None,None),fallback=RulesFallback.RAISE_ERROR)

        """

        dtype = query_states.dtype
        cos, sin = position_embeddings
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


        # print(query_states.shape,value_states.shape)

        value_states=repeat_kv(value_states,self.num_key_value_groups)
        key_states=repeat_kv(key_states,self.num_key_value_groups)

        if q_len%128==0 and value_states.shape[-1]%128==0 and q_len>512:
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
                    mask_value=-1e17,
                    block_sizes=block_sizes
                )

                attn_output = jax.vmap(splash_kernel)(query_states , key_states, value_states)
                return attn_output

            @functools.partial(
                shard_map,
                mesh=self.jax_config.mesh,
                in_specs=P(['dp', 'fsdp'], 'tp', None, None),
                out_specs=P(['dp', 'fsdp'], 'tp', None, None),
                check_rep=False,
            )
            def wrap_flash_attention(query_states, key_states, value_states):
                attn_output = flash_attention(query_states, key_states, value_states,causal=True)
                return attn_output
            attn_output=wrap_splash_attention(query_states/ math.sqrt(self.head_dim), key_states, value_states).astype(jnp.bfloat16)


        else:

            attn_weights = (query_states @ key_states.swapaxes(2, 3)) / math.sqrt(self.head_dim)
            if attn_mask is not None:  # no matter the length, we just slice it
                causal_mask = attn_mask
                attn_weights = attn_weights + causal_mask

            attn_weights = nn.softmax(attn_weights.astype(jnp.float32), axis=-1, ).astype(attn_weights.dtype)
            attn_output = attn_weights @ value_states





        # attn_weights = (query_states @ key_states.swapaxes(2, 3)) / math.sqrt(self.head_dim)
        #
        # if attn_mask is not None:  # no matter the length, we just slice it
        #     # causal_mask = attn_mask[:, :, :, : key_states.shape[-2]]
        #     causal_mask = attn_mask
        #     attn_weights = attn_weights + causal_mask
        #
        # attn_weights = nn.softmax(attn_weights.astype(jnp.float32), axis=-1, ).astype(attn_weights.dtype)
        # attn_output = attn_weights @ value_states


        attn_output = einops.rearrange(attn_output, 'b h n d-> b n (h d)')
        attn_output = self.o_proj(attn_output)

        # if cache is not None:
        #     new_cache = {
        #         'v': value_states_cache,
        #         'k': key_states_cache,
        #         'end_index': cache['end_index'] + q_len,
        #     }
        # else:
        #     new_cache = None

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
    ) -> LayerCache:
        del cls  # not used

        cache={
            'v': jnp.zeros(
                (batch_size, num_heads, cache_size, head_dim), dtype=dtype
            ),
            'k': jnp.zeros(
                (batch_size, num_heads, cache_size, head_dim), dtype=dtype
            ),
            'end_index': jnp.zeros((batch_size,), dtype=jnp.int32),
        }
        if shard_method:
            # cache['v']=shard_method(cache['v'])
            # cache['k'] = shard_method(cache['k'])


            cache['v'] = jax.tree_util.tree_map_with_path(shard_method,cache['v'])
            cache['k'] = jax.tree_util.tree_map_with_path(shard_method,cache['k'])



        return cache


class LlamaDecoderLayer(nn.Module):
    config: LlamaConfig
    jax_config: LlamaJaxConfig = None

    def setup(self) -> None:
        self.hidden_size = self.config.hidden_size
        # self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.self_attn = LlamaAttention(config=self.config, )
        self.mlp = LlamaMLP(self.config)
        self.input_layernorm = LlamaRMSNorm(eps=self.config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(eps=self.config.rms_norm_eps)

    def __call__(
            self,
            inputs_embeds: jax.Array,
            input_ids: jax.Array,
            cache: LayerCache | None,
            attn_mask: jax.Array,
            position_embeddings: jax.Array
    ) -> tuple[LayerCache | None, jax.Array]:
        inputs_normalized = self.input_layernorm(inputs_embeds)
        cache, attn_output = self.self_attn(
            x=inputs_normalized,
            position_embeddings=position_embeddings,
            cache=cache,
            attn_mask=attn_mask,
            input_ids=input_ids

        )

        attn_output += inputs_embeds
        outputs = self.post_attention_layernorm(attn_output)
        outputs = self.mlp(outputs)
        outputs += attn_output
        return cache, outputs


class LlamaModel(nn.Module):
    config: LlamaConfig
    jax_config: LlamaJaxConfig = None

    def setup(self) -> None:
        self.embed_tokens = nn.Embed(self.config.vocab_size, self.config.hidden_size, )
        self.layers = [LlamaDecoderLayer(self.config) for layer_idx in range(self.config.num_hidden_layers)]
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
        # causal_mask = self._update_causal_mask(
        #     attention_mask, inputs_embeds,
        # )

        if n > 1:
            attention_mask = jnp.full(
                (n, attention_mask.shape[1]), K_MASK#float("-inf")
            )
            attention_mask = jnp.triu(attention_mask, 1)[None, None, ...]
        else:
            attention_mask = jnp.where(attention_mask, 0, float("-inf"))


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


class LlamaForCausalLM(nn.Module):
    config: LlamaConfig
    jax_config: LlamaJaxConfig=None

    def setup(self) -> None:
        self.model = LlamaModel(self.config,jax_config=self.jax_config)
        # self.vocab_size = config.vocab_size
        self.lm_head = nn.Dense(self.config.vocab_size, use_bias=False)

    def __call__(
            self,
            input_ids: jax.Array,  # [B, L]
            position_ids: jax.Array=None,  # [B, L]
            cache: Cache | None = None,  # (sequence length L')
            attention_mask: jax.Array = None,  # [B, L, L']
            inputs_embeds: jax.Array | None = None
    ) -> tuple[jax.Array, Cache | None]:

        b,n=input_ids.shape

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
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states)
        return logits, cache




def convert_torch_to_flax_llama(state_dict):
    # print(state_dict.keys())
    params = {}
    i = 0
    while f'model.layers.{i}.self_attn.q_proj.weight' in state_dict:
        params[f'model.layers_{i}.input_layernorm.scale'] = state_dict[f'model.layers.{i}.input_layernorm.weight']
        params[f'model.layers_{i}.mlp.down_proj.kernel'] = state_dict[
            f'model.layers.{i}.mlp.down_proj.weight'].transpose(1, 0)
        params[f'model.layers_{i}.mlp.gate_proj.kernel'] = state_dict[
            f'model.layers.{i}.mlp.gate_proj.weight'].transpose(1, 0)
        params[f'model.layers_{i}.mlp.up_proj.kernel'] = state_dict[f'model.layers.{i}.mlp.up_proj.weight'].transpose(1,
                                                                                                                      0)
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

        i += 1
    params['model.norm.scale'] = state_dict['model.norm.weight']
    params['model.embed_tokens.embedding'] = state_dict['model.embed_tokens.weight']  #.transpose(1,0)
    # params['lm_head.kernel'] = params['model.embed_tokens.embedding'].transpose(1, 0)
    params['lm_head.kernel'] = state_dict['lm_head.weight'].transpose(1, 0)

    return flax.traverse_util.unflatten_dict(params, sep='.')





def test_vicuna(prompt="I'm a language model called Vicuna"):
    max_cache_length=96
    # jax.config.update('jax_platform_name', 'cpu')
    # variables=model.init(rng,x.astype(jnp.int32),x.astype(jnp.int32))
    # params=variables['params']

    from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor, LlamaTokenizerFast, \
        AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
    from transformers.generation import configuration_utils

    # from transformers import BitsAndBytesConfig
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    # )

    # Load the base model with adapters on top
    model = AutoModelForCausalLM.from_pretrained(
        'lmsys/vicuna-7b-v1.5',
        torch_dtype=torch.float16,
        # quantization_config=quantization_config,
    )
    # prompt="I'm a language model called Vicuna, and I was trained by Large Model Systems Organization"
    # prompt="I'm a language model called Vicuna"
    prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hello! ASSISTANT: Hi!</s>USER: Who are you? ASSISTANT: {prompt}"
    prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hello! ASSISTANT: Hi!</s>USER: Who are you? ASSISTANT:"

    device = 'cuda'
    device = 'cpu'
    inputs = tokenizer(prompt, return_tensors="jax")
    # inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # # print(inputs.device)
    # out=model.generate(**inputs,do_sample=False,temperature=1.0,top_k=1,top_p=1.0)
    # # print(out)
    # output=tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # print(output)
    #
    #
    # print('\n'*10)
    # while True:

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    state_dict = model.state_dict()
    params = convert_torch_to_flax_llama(state_dict)

    dtype = jnp.bfloat16
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(np.array(x), dtype=dtype), params)

    pad_attention = jnp.pad(attention_mask, ((0, 0), (0, max_cache_length - attention_mask.sum(axis=1)[0])))

    input_ids = input_ids
    position_ids = jnp.arange(0, input_ids.shape[1])[None, ...]

    llama_config = LlamaConfig(max_cache_length=max_cache_length)

    model = LlamaForCausalLM(llama_config)
    cache = llama_config.init_cache(1, dtype=dtype)

    b, n, d = shape = 1, 1, 768

    jit_infer = jax.jit(model.apply)

    logits, cache = jit_infer({'params': params}, input_ids=input_ids, position_ids=position_ids,
                              attention_mask=pad_attention, cache=cache)
    # select_ids = jnp.argmax(logits[:, -1], axis=1)
    # decoded_token = tokenizer.decode(select_ids)
    # print(decoded_token, end='', flush=True)

    # print(out)
    # print(f'{pad_attention=}  {position_ids=}')

    position_ids = position_ids[:, -1][..., None]
    max_decode_length = 250
    res = []
    # exit_token_ids=tokenizer.encode(tokenizer.st)[0]
    # exit_token_ids = tokenizer
    # print(tokenizer.eos_token, exit_token_ids)

    exit_token_ids = tokenizer.eos_token_id
    print(f'{tokenizer.eos_token=} , {exit_token_ids=}')

    for i in tqdm(range(max_decode_length)):
        select_ids = jnp.argmax(logits[:, -1], axis=1)
        print(select_ids)
        if select_ids[0] == exit_token_ids:
            break

        res.append(select_ids)
        input_ids = select_ids[..., None]
        position_ids += 1
        pad_attention = pad_attention.at[:, attention_mask.sum(axis=1)[0] + i].set(1)

        logits, cache = jit_infer({'params': params}, input_ids=input_ids, position_ids=position_ids,
                                  attention_mask=pad_attention, cache=cache)

        # print(tokenizer.decode(select_ids))

    ans = []
    for t in res:
        decoded_token = tokenizer.decode(t)
        print(decoded_token, end='', flush=True)
        ans.append(decoded_token)
    print('\n' * 10)

    print(np.array(res))
    output = \
        tokenizer.batch_decode(np.array(res).reshape(1, -1), skip_special_tokens=True,
                               clean_up_tokenization_spaces=False)[
            0]
    # output=tokenizer.decode(np.array(res), skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(output)

    return cache


def test_vicuna_torch():
    # jax.config.update('jax_platform_name', 'cpu')
    # variables=model.init(rng,x.astype(jnp.int32),x.astype(jnp.int32))
    # params=variables['params']

    from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor, LlamaTokenizerFast, \
        AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
    from transformers.generation import configuration_utils

    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    )

    # Load the base model with adapters on top
    model = AutoModelForCausalLM.from_pretrained(
        'lmsys/vicuna-7b-v1.5',
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
    )

    prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hello! ASSISTANT: Hi!</s>USER: Who are you? ASSISTANT:"

    device = 'cuda'
    # device = 'cpu'
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out=model.generate(**inputs,do_sample=False,temperature=1.0,top_k=1,top_p=1.0,max_new_tokens=512)
    output=tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(output)
























def test_llama3():
    # model_path='./autodl-tmp/LLM-Research/Meta-Llama-3___1-8B-Instruct'
    model_path = '/home/john/PycharmProjects/test/autodl-tmp/LLM-Research/Meta-Llama-3.1-8B-Instruct'
    # model_path = '/home/john/PycharmProjects/test/autodl-tmp/qwen/Qwen2-7B-Instruct'
    # model_path = '/home/john/PycharmProjects/test/output/llama3_1_instruct_lora/checkpoint-699'
    # jax.config.update('jax_platform_name', 'cpu')
    # variables=model.init(rng,x.astype(jnp.int32),x.astype(jnp.int32))
    # params=variables['params']

    from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor, LlamaTokenizerFast, \
        AutoTokenizer, AutoModelForCausalLM,AutoConfig


    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(config)

    # while True:
    #     pass


    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    from transformers.generation import configuration_utils

    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load the base model with adapters on top
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        trust_remote_code=True
    )

    # prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hello! ASSISTANT: Hi!</s>USER: How are you? ASSISTANT:"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who are you?   "},
    ]

    prompt = "Give me a short introduction to large language model."
    prompt = "Who are you?"
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": prompt}
    # ]

    # messages = [
    #     {"role": "system", "content": "假设你是皇帝身边的女人--甄嬛。"},
    #     {"role": "user", "content": "嬛嬛你怎么了，朕替你打抱不平！"}
    # ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(prompt)

    device = 'cuda'
    # device = 'cpu'
    # inputs = tokenizer(prompt, return_tensors="jax")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out=model.generate(**inputs,do_sample=False,temperature=1.0,top_k=1,top_p=1.0,   max_new_tokens=512)
    # out = model.generate(inputs.input_ids.to(device), max_new_tokens=512)
    output=tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(output)
    print('\n'*10)
    while True:
        pass


    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    state_dict = model.state_dict()
    params = convert_torch_to_flax_llama(state_dict)

    dtype = jnp.bfloat16
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(np.array(x), dtype=dtype), params)

    pad_attention = jnp.pad(attention_mask, ((0, 0), (0, 1024 - attention_mask.sum(axis=1)[0])))

    input_ids = input_ids
    position_ids = jnp.arange(0, input_ids.shape[1])[None, ...]

    llama_config = LlamaConfig(max_cache_length=1024)

    model = LlamaForCausalLM(llama_config)
    cache = llama_config.init_cache(1, dtype=dtype)

    b, n, d = shape = 1, 1, 768

    jit_infer = jax.jit(model.apply)

    logits, cache = jit_infer({'params': params}, input_ids=input_ids, position_ids=position_ids,
                              attention_mask=pad_attention, cache=cache)
    # select_ids = jnp.argmax(logits[:, -1], axis=1)
    # decoded_token = tokenizer.decode(select_ids)
    # print(decoded_token, end='', flush=True)

    # print(out)

    position_ids = position_ids[:, -1][..., None]

    max_decode_length = 25
    res = []
    # exit_token_ids=tokenizer.encode(tokenizer.st)[0]
    exit_token_ids = 2
    print(tokenizer.eos_token, exit_token_ids)
    for i in tqdm(range(max_decode_length)):
        select_ids = jnp.argmax(logits[:, -1], axis=1)
        print(select_ids)
        if select_ids[0] == exit_token_ids:
            break

        res.append(select_ids)
        input_ids = select_ids[..., None]
        position_ids += 1
        pad_attention = pad_attention.at[:, attention_mask.sum(axis=1)[0] + i].set(1)
        logits, cache = jit_infer({'params': params}, input_ids=input_ids, position_ids=position_ids,
                                  attention_mask=pad_attention, cache=cache)

        # print(tokenizer.decode(select_ids))

    ans = []
    for t in res:
        decoded_token = tokenizer.decode(t)
        print(decoded_token, end='', flush=True)
        ans.append(decoded_token)
    print('\n' * 10)

    print(np.array(res))
    output = \
        tokenizer.batch_decode(np.array(res).reshape(1, -1), skip_special_tokens=True,
                               clean_up_tokenization_spaces=False)[
            0]
    # output=tokenizer.decode(np.array(res), skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(output)

    """
    generate_ids = model.generate(inputs.input_ids.to(device), max_new_tokens=1)
    print(generate_ids.shape)
    output=tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    print(output)
    print()

    """

# if __name__ == "__main__":
#     # test_vicuna()
#     # test_llama3()
#     test_qwen2()