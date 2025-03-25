import math
from typing import Any

import einops
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from transformers import SiglipConfig

from MLLM_JAX.language.llama.llama import LlamaJaxConfig
from MLLM_JAX.vision.siglip.configuration_siglip import SiglipVisionConfig


use_fast_variance=False

class SiglipVisionEmbeddings(nn.Module):
    config: SiglipVisionConfig

    def setup(self):
        self.embed_dim = self.config.hidden_size
        self.image_size = self.config.image_size
        self.patch_size = self.config.patch_size

        self.patch_embedding = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            name="patch_embedding",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embed(
            num_embeddings=self.num_positions,
            features=self.embed_dim,
            name="position_embedding",
        )

        # Create position_ids - in Flax, we handle this in __call__
        self.position_ids = jnp.expand_dims(jnp.arange(self.num_positions), axis=0)

    def interpolate_pos_encoding(
        self, embeddings: jnp.ndarray, height: int, width: int
    ) -> jnp.ndarray:
        num_patches = embeddings.shape[1]
        num_positions = self.num_positions

        if num_patches == num_positions and height == width:
            return self.position_embedding(self.position_ids)

        patch_pos_embed = jnp.expand_dims(self.position_embedding.embedding, axis=0)

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = int(math.sqrt(num_positions))
        patch_pos_embed = patch_pos_embed.reshape(
            1, sqrt_num_positions, sqrt_num_positions, dim
        )
        patch_pos_embed = jnp.transpose(patch_pos_embed, (0, 3, 1, 2))

        patch_pos_embed = jax.image.resize(
            patch_pos_embed,
            shape=(1, dim, new_height, new_width),
            method="bicubic",
            antialias=True,
        )

        patch_pos_embed = jnp.transpose(patch_pos_embed, (0, 2, 3, 1))
        patch_pos_embed = patch_pos_embed.reshape(1, -1, dim)
        return patch_pos_embed

    def __call__(
        self, pixel_values: jnp.ndarray, interpolate_pos_encoding: bool = False
    ) -> jnp.ndarray:
        batch_size, _, height, width = pixel_values.shape
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = jnp.reshape(patch_embeds, (batch_size, -1, self.embed_dim))

        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(
                embeddings, height, width
            )
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings




class SiglipAttention(nn.Module):
    """Attention module."""
    config: SiglipVisionConfig
    jax_config: Any = None

    def setup(self) -> None:
        config = self.config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Dense( self.embed_dim)
        self.v_proj = nn.Dense( self.embed_dim)
        self.q_proj = nn.Dense( self.embed_dim)
        self.out_proj = nn.Dense( self.embed_dim)
        self.scale = self.head_dim**-0.5

    def __call__(
            self,
            x: jax.Array,
    ):
        bsz, q_len, _ = x.shape

        query_states = self.q_proj(x)
        key_states = self.k_proj(x)
        value_states = self.v_proj(x)

        query_states = einops.rearrange(query_states, 'b n (h d)->b h n  d ', d=self.head_dim)
        key_states = einops.rearrange(key_states, 'b n (h d)->b h n  d ', d=self.head_dim)
        value_states = einops.rearrange(value_states, 'b n (h d)->b h n  d ', d=self.head_dim)

        if q_len%128==0 and value_states.shape[-1]%128==0 and self.jax_config is not  None:
            raise NotImplementedError()
        else:
            attn_weights = (query_states @ key_states.swapaxes(2, 3))   * self.scale
            attn_weights = nn.softmax(attn_weights.astype(jnp.float32), axis=-1, ).astype(attn_weights.dtype)
            attn_output = attn_weights @ value_states

        attn_output = einops.rearrange(attn_output, 'b h n d-> b n (h d)')
        attn_output = self.out_proj(attn_output)
        return  attn_output





class SiglipMLP(nn.Module):
    config: SiglipVisionConfig

    def setup(self):
        self.fc1 = nn.Dense(features=self.config.intermediate_size)
        self.fc2 = nn.Dense(features=self.config.hidden_size)

    def __call__(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):
    config: SiglipVisionConfig
    jax_config: Any = None

    def setup(self) -> None:
        config=self.config
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config=config,jax_config=self.jax_config)
        self.layer_norm1 = nn.LayerNorm( epsilon=config.layer_norm_eps,use_fast_variance=use_fast_variance)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm( epsilon=config.layer_norm_eps,use_fast_variance=use_fast_variance)

        # Ignore copy

    def __call__(
            self,
            hidden_states: jax.Array,
    ):
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states
            # hidden_states=hidden_states,
            # attention_mask=attention_mask,
            # output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
        # outputs = (hidden_states,)
        # return outputs


class SiglipEncoder(nn.Module):
    config: SiglipConfig
    jax_config:Any=None

    def setup(self, ):
        super().__init__()
        config=self.config
        self.layers = [SiglipEncoderLayer(config,jax_config=self.jax_config) for _ in range(config.num_hidden_layers)]

    # Ignore copy
    def __call__(
        self,
        inputs_embeds,
    ):
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:

            hidden_states = encoder_layer(
                hidden_states,
                # attention_mask,
                # output_attentions=output_attentions,
            )
        return hidden_states







class SiglipVisionTransformer(nn.Module):
    config: SiglipVisionConfig
    jax_config: LlamaJaxConfig = None
    def setup(self, ):
        config=self.config
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config,jax_config=self.jax_config)
        self.post_layernorm = nn.LayerNorm(epsilon=config.layer_norm_eps,use_fast_variance=use_fast_variance)
    #     self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head
    #     if self.use_head:
    #         self.head = SiglipMultiheadAttentionPoolingHead(config)
    #
    def __call__(
        self,
        pixel_values,
        interpolate_pos_encoding=False
    ):
        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)


        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs

        # last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state
        # pooler_output = self.head(last_hidden_state) if self.use_head else None
        # if not return_dict:
        #     return (last_hidden_state, pooler_output) + encoder_outputs[1:]
        #
        # return BaseModelOutputWithPooling(
        #     last_hidden_state=last_hidden_state,
        #     pooler_output=pooler_output,
        #     hidden_states=encoder_outputs.hidden_states,
        #     attentions=encoder_outputs.attentions,
        # )




class SiglipVisionModel(nn.Module):
    config: SiglipVisionConfig
    config_class = SiglipVisionConfig
    main_input_name = "pixel_values"
    jax_config: LlamaJaxConfig = None

    def setup(self, ):
        self.vision_model = SiglipVisionTransformer(self.config,jax_config=self.jax_config)


    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def __call__(
        self,
        pixel_values,
        interpolate_pos_encoding: bool = False,
    ):

        return self.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )









def convert_torch_to_flax_siglip(state_dict,prefix='vision_model'):
    # 'vision_model.encoder.layers.21.self_attn.out_proj.bias'
    # 'vision_model.encoder.layers_15.layer_norm1.bias'

    state_dict=jax.tree_util.tree_map(lambda x:x.cpu().numpy(),state_dict)


    params = {}
    i = 0
    while f'{prefix}.encoder.layers.{i}.self_attn.q_proj.weight' in state_dict:
        params[f'{prefix}.encoder.layers_{i}.layer_norm1.scale'] = state_dict[f'{prefix}.encoder.layers.{i}.layer_norm1.weight']
        params[f'{prefix}.encoder.layers_{i}.layer_norm2.scale'] = state_dict[f'{prefix}.encoder.layers.{i}.layer_norm2.weight']
        params[f'{prefix}.encoder.layers_{i}.layer_norm1.bias'] = state_dict[f'{prefix}.encoder.layers.{i}.layer_norm1.bias']
        params[f'{prefix}.encoder.layers_{i}.layer_norm2.bias'] = state_dict[f'{prefix}.encoder.layers.{i}.layer_norm2.bias']

        params[f'{prefix}.encoder.layers_{i}.mlp.fc1.kernel'] = state_dict[f'{prefix}.encoder.layers.{i}.mlp.fc1.weight'].transpose(1, 0)
        params[f'{prefix}.encoder.layers_{i}.mlp.fc1.bias'] = state_dict[f'{prefix}.encoder.layers.{i}.mlp.fc1.bias']

        params[f'{prefix}.encoder.layers_{i}.mlp.fc2.kernel'] = state_dict[f'{prefix}.encoder.layers.{i}.mlp.fc2.weight'].transpose(1, 0)
        params[f'{prefix}.encoder.layers_{i}.mlp.fc2.bias'] = state_dict[f'{prefix}.encoder.layers.{i}.mlp.fc2.bias']


        params[f'{prefix}.encoder.layers_{i}.self_attn.q_proj.kernel'] = state_dict[f'{prefix}.encoder.layers.{i}.self_attn.q_proj.weight'].transpose(1, 0)
        params[f'{prefix}.encoder.layers_{i}.self_attn.k_proj.kernel'] = state_dict[f'{prefix}.encoder.layers.{i}.self_attn.k_proj.weight'].transpose(1, 0)
        params[f'{prefix}.encoder.layers_{i}.self_attn.v_proj.kernel'] = state_dict[f'{prefix}.encoder.layers.{i}.self_attn.v_proj.weight'].transpose(1, 0)
        params[f'{prefix}.encoder.layers_{i}.self_attn.out_proj.kernel'] = state_dict[f'{prefix}.encoder.layers.{i}.self_attn.out_proj.weight'].transpose(1, 0)



        params[f'{prefix}.encoder.layers_{i}.self_attn.q_proj.bias'] = state_dict[f'{prefix}.encoder.layers.{i}.self_attn.q_proj.bias']
        params[f'{prefix}.encoder.layers_{i}.self_attn.k_proj.bias'] = state_dict[f'{prefix}.encoder.layers.{i}.self_attn.k_proj.bias']
        params[f'{prefix}.encoder.layers_{i}.self_attn.v_proj.bias'] = state_dict[f'{prefix}.encoder.layers.{i}.self_attn.v_proj.bias']
        params[f'{prefix}.encoder.layers_{i}.self_attn.out_proj.bias'] = state_dict[f'{prefix}.encoder.layers.{i}.self_attn.out_proj.bias']

        i += 1


    params[f'{prefix}.post_layernorm.scale'] = state_dict[f'{prefix}.post_layernorm.weight']
    params[f'{prefix}.post_layernorm.bias'] = state_dict[f'{prefix}.post_layernorm.bias']
    params[f'{prefix}.embeddings.patch_embedding.kernel'] = state_dict[f'{prefix}.embeddings.patch_embedding.weight'].transpose(2, 3, 1, 0)
    params[f'{prefix}.embeddings.patch_embedding.bias'] = state_dict[f'{prefix}.embeddings.patch_embedding.bias']
    params[f'{prefix}.embeddings.position_embedding.embedding'] = state_dict[f'{prefix}.embeddings.position_embedding.weight']



    return flax.traverse_util.unflatten_dict(params, sep='.')

