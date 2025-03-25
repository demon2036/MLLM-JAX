from dataclasses import dataclass
from typing import Optional, Tuple

import einops
import flax
import jax
import jax.numpy as jnp
from flax import linen as nn
import jax.scipy as sp

from .configuration_clip import CLIPVisionConfig



use_fast_variance=False
precision='highest'

@dataclass
class BaseModelOutputWithPooling:
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: jax.Array = None
    pooler_output: jax.Array = None
    hidden_states: Optional[Tuple[jax.Array, ...]] = None
    attentions: Optional[Tuple[jax.Array, ...]] = None


@dataclass
class BaseModelOutput():
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: jax.Array = None
    hidden_states: Optional[Tuple[jax.Array, ...]] = None
    attentions: Optional[Tuple[jax.Array, ...]] = None



class CLIPVisionEmbeddings(nn.Module):
    config: CLIPVisionConfig  # Assuming config is a dataclass with similar attributes to the PyTorch version.

    def setup(self):
        # Initialize the class embedding
        self.class_embedding = self.param('class_embedding', nn.initializers.normal(stddev=0.02),
                                          (self.config.hidden_size,))

        # Define the convolution layer for patch embedding
        self.patch_embedding = nn.Conv(
            features=self.config.hidden_size,
            kernel_size=(self.config.patch_size, self.config.patch_size),
            strides=(self.config.patch_size, self.config.patch_size),
            use_bias=False,precision=precision
        )

        # Position embeddings, including class token
        self.num_patches = (self.config.image_size // self.config.patch_size) ** 2
        self.num_positions = self.num_patches + 1  # +1 for the class token
        self.position_embedding = nn.Embed(
            num_embeddings=self.num_positions,
            features=self.config.hidden_size
        )

    def interpolate_pos_encoding(self, embeddings: jnp.ndarray, height: int, width: int) -> jnp.ndarray:
        num_patches = embeddings.shape[1] - 1
        position_embedding = self.position_embedding.embeddings  # (num_positions, embed_dim)
        num_positions = position_embedding.shape[0] - 1

        # Always interpolate when tracing
        if height == self.config.image_size and width == self.config.image_size:
            return self.position_embedding(embeddings.shape[0])  # Return original position embeddings

        # Split position embeddings into class and patch embeddings
        class_pos_embed = position_embedding[:1, :]  # (1, embed_dim)
        patch_pos_embed = position_embedding[1:, :]  # (num_patches, embed_dim)

        dim = embeddings.shape[-1]
        new_height = height // self.config.patch_size
        new_width = width // self.config.patch_size

        # Reshape and interpolate the patch position embeddings
        sqrt_num_positions = int(num_positions ** 0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = jnp.moveaxis(patch_pos_embed, (0, 3, 1, 2), (0, 2, 3, 1))

        patch_pos_embed = sp.ndimage.zoom(patch_pos_embed,
                                          (1, 1, new_height / sqrt_num_positions, new_width / sqrt_num_positions),
                                          order=1)

        patch_pos_embed = jnp.moveaxis(patch_pos_embed, (0, 2, 3, 1), (0, 1, 2, 3)).reshape(1, -1, dim)

        return jnp.concatenate([class_pos_embed, patch_pos_embed], axis=1)

    def __call__(self, pixel_values: jnp.ndarray, interpolate_pos_encoding: bool = False) -> jnp.ndarray:
        # Check the shape of pixel_values: (batch_size, height, width, channels) for JAX
        batch_size, height, width, _ = pixel_values.shape

        # Apply patch embedding
        patch_embeds = self.patch_embedding(pixel_values)  # Shape: (batch_size, new_height, new_width, embed_dim)
        patch_embeds = einops.rearrange(patch_embeds,'b h w c -> b (h w) c')  # Flatten and transpose



        # Expand class embedding
        # class_embeds = jnp.expand_dims(self.class_embedding, axis=0).repeat(batch_size, axis=0)
        class_embeds=self.class_embedding.reshape(1,1,-1).repeat(batch_size, axis=0)
        print(class_embeds.shape,patch_embeds.shape)


        # Concatenate class embedding and patch embeddings
        embeddings = jnp.concatenate([class_embeds, patch_embeds], axis=1)

        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(jnp.arange(0,self.num_positions).reshape((1,-1)))


        return embeddings


class CLIPAttention(nn.Module):
    config: CLIPVisionConfig  # Assume 'CLIPConfig' is a dataclass similar to the PyTorch one

    def setup(self):
        self.embed_dim = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim ** -0.5
        self.dropout = self.config.attention_dropout

        # Define projection layers for query, key, value
        self.k_proj = nn.Dense(self.embed_dim)
        self.v_proj = nn.Dense(self.embed_dim)
        self.q_proj = nn.Dense(self.embed_dim)
        self.out_proj = nn.Dense(self.embed_dim)

    def _shape(self, tensor: jnp.ndarray, seq_len: int, bsz: int) -> jnp.ndarray:
        return einops.rearrange(tensor,'b n (h d)-> b h n d',h=self.num_heads)

        # return tensor.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def __call__(
            self,
            hidden_states: jnp.ndarray,
            attention_mask: Optional[jnp.ndarray] = None,
            causal_attention_mask: Optional[jnp.ndarray] = None,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, embed_dim = hidden_states.shape

        # Get query, key, and value projections
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).reshape(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.shape[1]
        attn_weights = jnp.matmul(query_states, key_states.transpose(0, 2, 1))

        if attn_weights.shape != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.shape}"
            )

        # Apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.shape != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.shape}"
                )
            attn_weights = attn_weights.reshape(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.reshape(bsz * self.num_heads, tgt_len, src_len)

        # Apply the general attention_mask
        if attention_mask is not None:
            if attention_mask.shape != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.shape}"
                )
            attn_weights = attn_weights.reshape(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.reshape(bsz * self.num_heads, tgt_len, src_len)

        # Apply softmax to attention weights
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        if output_attentions:
            # If outputting attention weights, reshape for proper gradient flow
            attn_weights_reshaped = attn_weights.reshape(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.reshape(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        # Apply dropout to attention probabilities
        # attn_probs = jax.nn.dropout(attn_weights, rate=self.dropout, deterministic=False)
        attn_probs=attn_weights

        # Compute the attention output
        attn_output = jnp.matmul(attn_probs, value_states)

        if attn_output.shape != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"attn_output should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.reshape(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.swapaxes(1, 2)  # Transpose heads and sequence length
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        # Apply final projection layer
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


def quick_gelu(x):
    return x * jax.nn.sigmoid(1.702 * x)

class CLIPMLP(nn.Module):
    config: CLIPVisionConfig  # Assuming config is a dataclass with similar attributes to the PyTorch version.

    def setup(self):
        # Define the first fully connected layer (fc1)
        self.fc1 = nn.Dense(self.config.intermediate_size)
        # Define the second fully connected layer (fc2)
        self.fc2 = nn.Dense(self.config.hidden_size)

        # Map activation function to Flax compatible functions
        self.activation_fn = self.get_activation_fn(self.config.hidden_act)
        print(self.config.hidden_act,self.activation_fn)

    def get_activation_fn(self, activation_name: str):
        # Mapping activation function names to Flax activation functions
        activation_map = {
            'gelu': nn.gelu,
            'relu': nn.relu,
            'tanh': nn.tanh,
            'quick_gelu':quick_gelu,
            # Add more activations as needed
        }
        return activation_map.get(activation_name, None)  # Default to ReLU if not found

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(nn.Module):
    config: CLIPVisionConfig  # Configuration class similar to PyTorch's configuration

    def setup(self):
        # Layer components
        self.embed_dim = self.config.hidden_size
        self.self_attn = CLIPAttention(config=self.config)
        self.layer_norm1 = nn.LayerNorm(epsilon=self.config.layer_norm_eps,use_fast_variance=use_fast_variance)
        self.mlp = CLIPMLP(config=self.config)
        self.layer_norm2 = nn.LayerNorm(epsilon=self.config.layer_norm_eps,use_fast_variance=use_fast_variance)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        causal_attention_mask: jnp.ndarray,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[jnp.ndarray]:
        """
        Args:
            hidden_states (`jnp.ndarray`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`jnp.ndarray`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attention tensors of all attention layers.
        """
        # First residual connection + attention block
        residual = hidden_states

        # Layer normalization (LayerNorm1)
        hidden_states = self.layer_norm1(hidden_states)

        # Attention (Self Attention)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )

        # Add the residual to the output
        hidden_states = residual + hidden_states

        # Second residual connection + MLP block
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)

        # MLP
        hidden_states = self.mlp(hidden_states)

        # Add residual again after MLP
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            # If requested, return attention weights
            outputs += (attn_weights,)

        return outputs






# Assuming CLIPEncoderLayer is already converted to Flax
class CLIPEncoder(nn.Module):
    config: 'CLIPConfig'  # Configuration class

    def setup(self):
        # Initialize layers (equivalent to ModuleList in PyTorch)
        self.layers = [CLIPEncoderLayer(config=self.config) for _ in range(self.config.num_hidden_layers)]
        self.gradient_checkpointing = False

    def __call__(
        self,
        inputs_embeds: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        causal_attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Args:
            inputs_embeds: The embedded inputs of shape `(batch_size, sequence_length, hidden_size)`
            attention_mask: The attention mask of shape `(batch_size, sequence_length)`
            causal_attention_mask: Causal attention mask for causal transformer models
            output_attentions: Whether or not to return attention weights
            output_hidden_states: Whether or not to return hidden states of all layers
            return_dict: Whether to return outputs as a dict (True) or tuple (False)
        """

        # Use config to set default values for some of the parameters
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Initialize the container for hidden states and attentions
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # Initialize hidden states
        hidden_states = inputs_embeds

        # Loop over each encoder layer
        for idx, encoder_layer in enumerate(self.layers):
            # Store hidden states if requested
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            # Apply gradient checkpointing if enabled
            if self.gradient_checkpointing and self.training:
                # Use jax.checkpoint for gradient checkpointing
                layer_outputs = jax.checkpoint(encoder_layer)(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

            # Get the hidden states after this layer
            hidden_states = layer_outputs[0]

            # Store attention weights if requested
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add the final hidden states to encoder states if requested
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        # Return the results
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class CLIPVisionTransformer(nn.Module):
    config: CLIPVisionConfig  # Configuration class

    def setup(self):
        # Initialize layers
        embed_dim = self.config.hidden_size
        self.embeddings = CLIPVisionEmbeddings(config=self.config)
        self.pre_layrnorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps,use_fast_variance=use_fast_variance)
        self.encoder = CLIPEncoder(config=self.config)  # Assuming CLIPEncoder is defined in Flax
        self.post_layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps,use_fast_variance=use_fast_variance)

    def __call__(
        self,
        pixel_values: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
    ) :
        # Ensure that the relevant flags are set
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Raise error if pixel_values is None
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Get the embeddings from the vision model
        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        # Apply pre-layer normalization
        hidden_states = self.pre_layrnorm(hidden_states)

        # Pass the embeddings through the encoder
        encoder_outputs = self.encoder(
            hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        print(type(encoder_outputs))
        # Extract the last hidden state
        # last_hidden_state = encoder_outputs[0]  # Assuming encoder_outputs[0] is the last hidden state
        last_hidden_state = encoder_outputs.last_hidden_state  # Assuming encoder_outputs[0] is the last hidden state
        pooled_output = last_hidden_state[:, 0, :]  # Pool the first token (CLS token)

        # Apply post-layer normalization to pooled output
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            # Return last hidden state and pooled output if return_dict is False
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # Return the outputs in the BaseModelOutputWithPooling format
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CLIPVisionModel(nn.Module):
    config: CLIPVisionConfig  # Configuration class


    def setup(self):
        self.vision_model=CLIPVisionTransformer(self.config)

    def __call__(self, *args, **kwargs):
        return self.vision_model(*args,**kwargs)


def convert_hf_to_flax_clip(state_dict,prefix_torch='model',prefix_flax='model'):
    params = {}
    params[f'{prefix_flax}.embeddings.class_embedding']=state_dict[f'{prefix_torch}.embeddings.class_embedding']
    params[f'{prefix_flax}.embeddings.patch_embedding.kernel'] = state_dict[f'{prefix_torch}.embeddings.patch_embedding.weight'].transpose(2, 3, 1, 0)
    params[f'{prefix_flax}.embeddings.position_embedding.embedding'] = state_dict[f'{prefix_torch}.embeddings.position_embedding.weight']#.transpose(2, 3, 1, 0)

    params[f'{prefix_flax}.pre_layrnorm.scale'] = state_dict[f'{prefix_torch}.pre_layrnorm.weight']
    params[f'{prefix_flax}.pre_layrnorm.bias'] = state_dict[f'{prefix_torch}.pre_layrnorm.bias']

    params[f'{prefix_flax}.post_layernorm.scale'] = state_dict[f'{prefix_torch}.post_layernorm.weight']
    params[f'{prefix_flax}.post_layernorm.bias'] = state_dict[f'{prefix_torch}.post_layernorm.bias']


    i = 0
    while f'{prefix_torch}.encoder.layers.{i}.self_attn.q_proj.weight' in state_dict:
        params[f'{prefix_flax}.encoder.layers_{i}.layer_norm1.scale'] = state_dict[f'{prefix_torch}.encoder.layers.{i}.layer_norm1.weight']
        params[f'{prefix_flax}.encoder.layers_{i}.layer_norm1.bias'] = state_dict[
            f'{prefix_torch}.encoder.layers.{i}.layer_norm1.bias']

        params[f'{prefix_flax}.encoder.layers_{i}.layer_norm2.scale'] = state_dict[f'{prefix_torch}.encoder.layers.{i}.layer_norm2.weight']
        params[f'{prefix_flax}.encoder.layers_{i}.layer_norm2.bias'] = state_dict[f'{prefix_torch}.encoder.layers.{i}.layer_norm2.bias']

        params[f'{prefix_flax}.encoder.layers_{i}.mlp.fc1.kernel'] = state_dict[
            f'{prefix_torch}.encoder.layers.{i}.mlp.fc1.weight'].transpose(1, 0)
        params[f'{prefix_flax}.encoder.layers_{i}.mlp.fc2.kernel'] = state_dict[
            f'{prefix_torch}.encoder.layers.{i}.mlp.fc2.weight'].transpose(1, 0)

        params[f'{prefix_flax}.encoder.layers_{i}.self_attn.q_proj.kernel'] = state_dict[
            f'{prefix_torch}.encoder.layers.{i}.self_attn.q_proj.weight'].transpose(1, 0)
        params[f'{prefix_flax}.encoder.layers_{i}.self_attn.k_proj.kernel'] = state_dict[
            f'{prefix_torch}.encoder.layers.{i}.self_attn.k_proj.weight'].transpose(1, 0)
        params[f'{prefix_flax}.encoder.layers_{i}.self_attn.v_proj.kernel'] = state_dict[
            f'{prefix_torch}.encoder.layers.{i}.self_attn.v_proj.weight'].transpose(1, 0)
        params[f'{prefix_flax}.encoder.layers_{i}.self_attn.out_proj.kernel'] = state_dict[
            f'{prefix_torch}.encoder.layers.{i}.self_attn.out_proj.weight'].transpose(1, 0)

        # if f'{prefix_torch}.encoder.layers.{i}.self_attn.q_proj.bias' in state_dict:
        params[f'{prefix_flax}.encoder.layers_{i}.self_attn.q_proj.bias'] = state_dict[
            f'{prefix_torch}.encoder.layers.{i}.self_attn.q_proj.bias']
        params[f'{prefix_flax}.encoder.layers_{i}.self_attn.k_proj.bias'] = state_dict[
            f'{prefix_torch}.encoder.layers.{i}.self_attn.k_proj.bias']
        params[f'{prefix_flax}.encoder.layers_{i}.self_attn.v_proj.bias'] = state_dict[
            f'{prefix_torch}.encoder.layers.{i}.self_attn.v_proj.bias']
        params[f'{prefix_flax}.encoder.layers_{i}.self_attn.out_proj.bias'] = state_dict[
            f'{prefix_torch}.encoder.layers.{i}.self_attn.out_proj.bias']


        params[f'{prefix_flax}.encoder.layers_{i}.mlp.fc1.bias'] = state_dict[
            f'{prefix_torch}.encoder.layers.{i}.mlp.fc1.bias']
        params[f'{prefix_flax}.encoder.layers_{i}.mlp.fc2.bias'] = state_dict[
            f'{prefix_torch}.encoder.layers.{i}.mlp.fc2.bias']

        i += 1

    return flax.traverse_util.unflatten_dict(params, sep='.')









