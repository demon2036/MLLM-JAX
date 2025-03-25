import copy
import math
from typing import Any

import einops
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel, splash_attention_mask
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from MLLM_JAX.language.gemma3.configuration_gemma3 import Gemma3TextConfig
from MLLM_JAX.language.gemma3.modeling_gemma3 import Gemma3RMSNorm, GemmaForCausalLM, convert_torch_to_flax_gemma3_text
from MLLM_JAX.language.llama.llama import LlamaJaxConfig, _compute_default_rope_parameters
from MLLM_JAX.mask import make_causal_bidirectional_attention_mask
from MLLM_JAX.mutinomial.gemma3.configuration_gemma3 import Gemma3Config
from MLLM_JAX.vision.siglip.modeling_siglip import convert_torch_to_flax_siglip, SiglipVisionModel
from jax.sharding import Mesh,PartitionSpec as PS

class Gemma3MultiModalProjector(nn.Module):
    config: Gemma3Config
    def setup(self, ):
        config=self.config
        self.mm_input_projection_weight = self.param(
            'mm_input_projection_weight',
            flax.linen.initializers.kaiming_normal(),
            (config.vision_config.hidden_size, config.text_config.hidden_size),
        )

        self.mm_soft_emb_norm = Gemma3RMSNorm(
            eps=config.vision_config.layer_norm_eps
        )

        self.patches_per_image = int(config.vision_config.image_size // config.vision_config.patch_size)
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side

    def __call__(self, vision_outputs: jax.Array):
        batch_size, _, seq_length = vision_outputs.shape
        # reshaped_vision_outputs=einops.rearrange(vision_outputs,'b n d ->b d n')
        reshaped_vision_outputs = vision_outputs.reshape(
            batch_size,  self.patches_per_image, self.patches_per_image,seq_length,
        )
        pooled_vision_outputs = nn.avg_pool(reshaped_vision_outputs,window_shape=(self.kernel_size,self.kernel_size),
                                            strides=(self.kernel_size,self.kernel_size))


        pooled_vision_outputs=einops.rearrange(pooled_vision_outputs,'b  h w d -> b  (h w) d')
        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)
        projected_vision_outputs=jnp.einsum('...nd,dk->...nk',normed_vision_outputs,self.mm_input_projection_weight)
        return projected_vision_outputs






class Gemma3ForConditionalGeneration(nn.Module):
    config: Gemma3Config
    jax_config: LlamaJaxConfig
    def setup(self, ):
        config=self.config
        self.vision_tower = SiglipVisionModel(config=config.vision_config)
        self.multi_modal_projector = Gemma3MultiModalProjector(config)
        self.language_model = GemmaForCausalLM(config=config.text_config,jax_config=self.jax_config)

        # self.vision_tower = AutoModel.from_config(config=config.vision_config)
        # self.multi_modal_projector = Gemma3MultiModalProjector(config)
        # self.vocab_size = config.text_config.vocab_size

        # language_model = AutoModelForCausalLM.from_config(config=config.text_config)
        #
        # if language_model._tied_weights_keys is not None:
        #     self._tied_weights_keys = [f"language_model.{k}" for k in language_model._tied_weights_keys]
        # self.language_model = language_model
        #
        # self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        # self.post_init()

    def get_image_features(self, pixel_values: jax.Array):
        """
        Projects the last hidden state from the vision model into language model space.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        vision_outputs = self.vision_tower(pixel_values=pixel_values)#.last_hidden_state
        image_features = self.multi_modal_projector(vision_outputs)
        return image_features

    def __call__(
            self,
            input_ids:jax.Array,
            pixel_values:jax.Array=None,
            attention_mask:jax.Array=None,
            position_ids:jax.Array=None,
            cache:Any=None,
            true_length=None,
            # input_ids: torch.LongTensor = None,
            # pixel_values: torch.FloatTensor = None,
            # attention_mask: Optional[torch.Tensor] = None,
            # position_ids: Optional[torch.LongTensor] = None,
            # past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
            # token_type_ids: Optional[torch.LongTensor] = None,
            # cache_position: Optional[torch.LongTensor] = None,
            # inputs_embeds: Optional[torch.FloatTensor] = None,
            # labels: Optional[torch.LongTensor] = None,
            # use_cache: Optional[bool] = None,
            # output_attentions: Optional[bool] = None,
            # output_hidden_states: Optional[bool] = None,
            # return_dict: Optional[bool] = None,
            # logits_to_keep: Union[int, torch.Tensor] = 0,
            **lm_kwargs,
    ):

        position_ids+=1

        inputs_embeds=self.language_model.model.embedding_inputs_ids(input_ids)
        b,n,d=inputs_embeds.shape

        if pixel_values is not None:

            image_features = self.get_image_features(pixel_values)

            special_image_mask = (input_ids == self.config.image_token_index)
            inputs_embeds=jnp.place(
                inputs_embeds,
                mask=special_image_mask[...,None].repeat(
                 inputs_embeds.shape[-1], axis=-1
            ),
                vals=image_features,
                inplace=False
            )
        else:
            special_image_mask=None

        # if n > 1:
        #     attention_mask=make_causal_bidirectional_attention_mask(attention_mask.astype(jnp.bool),bidirectional_mask=special_image_mask)[:,None,:,:]
        #     # attention_mask=jnp.where(attention_mask,0,-1e37)#[:,None,:,:]
        # else:
        #     attention_mask = jnp.where(attention_mask, 0, -1e37)[:, None, None, ...]

        out=self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            cache=cache,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            special_image_mask=special_image_mask,
            true_length=true_length
        )

        return out







def convert_torch_to_flax_gemma3_multi_projector(state_dict,prefix='multi_modal_projector'):
    params =dict()
    params[f'mm_soft_emb_norm.scale'] = state_dict[f'mm_soft_emb_norm.weight']
    params[f'mm_input_projection_weight'] = state_dict[f'mm_input_projection_weight']
    return flax.traverse_util.unflatten_dict(params, sep='.')





def convert_torch_to_flax_gemma3(state_dict,prefix='model'):
    params =dict()
    state_dict=flax.traverse_util.unflatten_dict(state_dict,sep='.')

    projector=state_dict.pop('multi_modal_projector')
    projector=convert_torch_to_flax_gemma3_multi_projector(flax.traverse_util.flatten_dict(projector,sep='.'))

    vision_tower = state_dict.pop('vision_tower')
    vision_tower = convert_torch_to_flax_siglip(flax.traverse_util.flatten_dict(vision_tower, sep='.'))

    language_model = state_dict.pop('language_model')
    language_model = convert_torch_to_flax_gemma3_text(flax.traverse_util.flatten_dict(language_model, sep='.'))


    params['multi_modal_projector']=projector
    params['vision_tower'] = vision_tower
    params['language_model'] = language_model
    # params.update(flax.traverse_util.flatten_dict(vision_tower, sep='.'))
    # print(params.keys())
    return flax.traverse_util.unflatten_dict(params, sep='.')




def get_partition_rules_gemma3():
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
        # ('lm_head/kernel', PS( 'fsdp','tp' )),

        ('vision_tower/.*/q_proj/kernel', PS('fsdp', 'tp')),
        ('vision_tower/.*/k_proj/kernel', PS('fsdp', 'tp')),
        ('vision_tower/.*/v_proj/kernel', PS('fsdp', 'tp')),
        ('vision_tower/.*/out_proj/kernel', PS('tp', 'fsdp', )),

        ('.*/vision_tower/.*/fc1/kernel', PS('fsdp', 'tp')),
        ('.*/vision_tower/.*/fc2/kernel', PS('tp', 'fsdp')),

        ('mm_input_projection_weight', PS('tp', 'fsdp')),
        # ('language_model/.*/embed_tokens/embedding', PS('fsdp', 'tp')),
        # ('language_model/.*/lm_head/kernel', PS('tp','fsdp', )),

        ('scale',PS('tp')),

        ('.*', PS(None)),
    )




# def get_partition_rules_gemma3():
#     # self.vision_tower = SiglipVisionModel(config=config.vision_config)
#     # self.multi_modal_projector = Gemma3MultiModalProjector(config)
#     # self.language_model = GemmaForCausalLM(config=config.text_config, jax_config=self.jax_config)
#
#     return (
#         ('language_model/.*/self_attn/q_proj/kernel', PS('fsdp', 'tp')),
#         ('language_model/.*/self_attn/k_proj/kernel', PS('fsdp', 'tp')),
#         ('language_model/.*/self_attn/v_proj/kernel', PS('fsdp', 'tp')),
#         ('language_model/.*/self_attn/o_proj/kernel', PS( 'tp', 'fsdp', )),
#
#         ('language_model/.*/mlp/gate_proj/kernel', PS('tp', 'fsdp')),
#         ('language_model/.*/mlp/up_proj/kernel', PS('tp', 'fsdp')),
#         ('language_model/.*/mlp/down_proj/kernel', PS('fsdp', 'tp')),
#         #
#         # ('language_model/.*/embed_tokens/embedding', PS('fsdp', 'tp')),
#         # ('language_model/.*/lm_head/kernel', PS('tp','fsdp', )),
#
#         #
#         # ('vision_tower/.*/q_proj/kernel', PS('fsdp', 'tp')),
#         # ('vision_tower/.*/k_proj/kernel', PS('fsdp', 'tp')),
#         # ('vision_tower/.*/v_proj/kernel', PS('fsdp', 'tp')),
#         # ('vision_tower/.*/out_proj/kernel', PS('tp', 'fsdp', )),
#         #
#         # ('.*/vision_tower/.*/fc1/kernel', PS('fsdp', 'tp')),
#         # ('.*/vision_tower/.*/fc2/kernel', PS('tp', 'fsdp')),
#
#         ('scale',PS('tp')),
#         ('.*', PS(None)),
#     )
