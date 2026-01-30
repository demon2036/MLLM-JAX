from typing import Any

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from .configuration_qwen2 import Qwen2Config
from ..llama.llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    _resolve_rope_theta,
)

class Qwen2MLP(LlamaMLP):
    config: Qwen2Config
    jax_config: Any = None


    def setup(self) -> None:
        dtype=self.jax_config.dtype
        param_dtype=self.jax_config.param_dtype
        self.gate_proj = nn.Dense(self.config.intermediate_size, use_bias=False,dtype=dtype,param_dtype=param_dtype)
        self.up_proj = nn.Dense(self.config.intermediate_size, use_bias=False,dtype=dtype,param_dtype=param_dtype)
        self.down_proj = nn.Dense(self.config.hidden_size, use_bias=False,dtype=dtype,param_dtype=param_dtype)


class Qwen2Attention(LlamaAttention):
    config: Qwen2Config
    jax_config: Any = None

    def setup(self) -> None:
        dtype=self.jax_config.dtype
        param_dtype=self.jax_config.param_dtype
        config = self.config
        use_bias = bool(getattr(config, "attention_bias", True))
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = _resolve_rope_theta(config)
        self.is_causal = True
        self.q_proj = nn.Dense(self.num_heads * self.head_dim, use_bias=use_bias,dtype=dtype,param_dtype=param_dtype)
        self.k_proj = nn.Dense(self.num_key_value_heads * self.head_dim, use_bias=use_bias,dtype=dtype,param_dtype=param_dtype)
        self.v_proj = nn.Dense(self.num_key_value_heads * self.head_dim, use_bias=use_bias,dtype=dtype,param_dtype=param_dtype)
        self.o_proj = nn.Dense(self.hidden_size, use_bias=False,dtype=dtype,param_dtype=param_dtype)
    def _jax_attention_use_block_sizes(self) -> bool:
        return False

    def _jax_attention_mask_value(self) -> float | None:
        return None


class Qwen2DecoderLayer(LlamaDecoderLayer):
    config: Qwen2Config
    jax_config: Any = None

    def setup(self) -> None:
        self.hidden_size = self.config.hidden_size
        self.self_attn = Qwen2Attention(config=self.config, jax_config=self.jax_config )
        self.mlp = Qwen2MLP(self.config, self.jax_config)
        self.input_layernorm = LlamaRMSNorm(eps=self.config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(eps=self.config.rms_norm_eps)


class Qwen2Model(nn.Module):
    config: Qwen2Config
    jax_config: Any = None

    def setup(self) -> None:
        dtype=self.jax_config.dtype
        param_dtype=self.jax_config.param_dtype
        self.embed_tokens = nn.Embed(self.config.vocab_size, self.config.hidden_size,dtype=dtype,param_dtype=param_dtype )
        self.layers = [Qwen2DecoderLayer(self.config, self.jax_config) for layer_idx in
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


class Qwen2ForCausalLM(nn.Module):
    config: Qwen2Config
    jax_config: Any = None

    def setup(self) -> None:
        dtype=self.jax_config.dtype
        param_dtype=self.jax_config.param_dtype
        # self.model = Qwen2Model(self.config, jax_config=self.jax_config)
        # self.lm_head = nn.Dense(self.config.vocab_size, use_bias=False)
        self.model = nn.remat(Qwen2Model)(self.config, jax_config=self.jax_config)
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

        # Weight tying (HF-compatible): reuse the input embedding matrix as lm_head when enabled.
        # This is important when adding new SID tokens during SFT/RL.
        if bool(getattr(self.config, "tie_word_embeddings", False)):
            logits = self.model.embed_tokens.attend(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        return logits, cache
