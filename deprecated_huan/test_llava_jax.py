from typing import Optional, List, Union, Tuple

import einops
import jax.random
import numpy as np
from PIL import Image
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast

from MLLM_JAX.mutinomial.llava.modeling_llava import LlavaForConditionalGeneration

import jax.numpy as jnp
import jax
from MLLM_JAX.vision.clip.clip import convert_hf_to_flax_clip
import flax
from transformers import AutoConfig, LlavaForConditionalGeneration as LlavaForConditionalGenerationTorch, AutoProcessor
import torch



class LlavaForConditionalGenerationTorchTest(LlavaForConditionalGenerationTorch):



        def forward(
                self,
                input_ids: torch.LongTensor = None,
                pixel_values: torch.FloatTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                vision_feature_layer: Optional[int] = None,
                vision_feature_select_strategy: Optional[str] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                cache_position: Optional[torch.LongTensor] = None,
                num_logits_to_keep: int = 0,
        ) -> Union[Tuple, LlavaCausalLMOutputWithPast]:
            r"""
            Args:
                labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                    Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                    config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                    (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

                num_logits_to_keep (`int`, *optional*):
                    Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                    `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                    token can save memory, which becomes pretty significant for long sequences or large vocabulary size.


            Returns:

            Example:

            ```python
            >>> from PIL import Image
            >>> import requests
            >>> from transformers import AutoProcessor, LlavaForConditionalGeneration

            >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
            >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

            >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
            >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

            >>> # Generate
            >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
            >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
            ```"""

            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            vision_feature_layer = (
                vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
            )
            vision_feature_select_strategy = (
                vision_feature_select_strategy
                if vision_feature_select_strategy is not None
                else self.config.vision_feature_select_strategy
            )

            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

            if pixel_values is not None and inputs_embeds is not None:
                raise ValueError(
                    "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
                )

            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)

            return inputs_embeds

            # print(f'{pixel_values=}')

            if pixel_values is not None:
                image_features = self.get_image_features(
                    pixel_values=pixel_values,
                    vision_feature_layer=vision_feature_layer,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                )

                n_image_tokens = (input_ids == self.config.image_token_index).sum().item()
                n_image_features = image_features.shape[0] * image_features.shape[1]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
                special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
                image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
                return inputs_embeds





def convert_hf_to_flax_llava_multimodal_projector(state_dict,prefix_torch='model',prefix_flax='model'):
    params = dict()
    print(state_dict.keys())


    params[f'linear_1.kernel'] = state_dict[
        f'linear_1.weight'].transpose(1, 0)
    params[f'linear_2.kernel'] = state_dict[
        f'linear_2.weight'].transpose(1, 0)

    params[f'linear_1.bias'] = state_dict[
        f'linear_1.bias']
    params[f'linear_2.bias'] = state_dict[
        f'linear_2.bias']
    return flax.traverse_util.unflatten_dict(params,'.')



def convert_hf_to_flax_llava(state_dict,prefix_torch='',prefix_flax=''):
    params=dict()


    state_dict = {k: v.cpu().numpy() for k, v in state_dict.items()}
    state_dict = flax.traverse_util.unflatten_dict(state_dict, sep=".")
    #['vision_model']
    vision_model=flax.traverse_util.flatten_dict(state_dict['vision_tower'],sep='.')
    multi_modal_projector = flax.traverse_util.flatten_dict(state_dict['multi_modal_projector'], sep='.')
    # print(vision_model.keys())

    params['vision_tower']=convert_hf_to_flax_clip(vision_model,prefix_torch='vision_model',prefix_flax='vision_model')


    # print(state_dict.keys())
    # while True:
    #     pass

    params['multi_modal_projector']=convert_hf_to_flax_llava_multimodal_projector(multi_modal_projector,prefix_torch='',prefix_flax='')
    return params




def get_image_features(
    llava_model, pixel_values, vision_feature_layer: int=-2, vision_feature_select_strategy: str='default'
):
    """
    Obtains image last hidden states from the vision tower and apply multimodal projection.

    Args:
        pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
           The tensors corresponding to the input images.
        vision_feature_layer (`int`):
            The index of the layer to select the vision feature.
        vision_feature_select_strategy (`str`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`
    Returns:
        image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
    """
    image_outputs = llava_model.vision_tower(pixel_values, output_hidden_states=True)
    # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
    selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
    if vision_feature_select_strategy == "default":
        selected_image_feature = selected_image_feature[:, 1:]
    elif vision_feature_select_strategy == "full":
        selected_image_feature = selected_image_feature
    else:
        raise ValueError(f"Unexpected select feature strategy: {llava_model.config.vision_feature_select_strategy}")

    # return selected_image_feature
    image_features = llava_model.multi_modal_projector(selected_image_feature)
    return image_features




def test1():
    from transformers import  AutoConfig, LlavaForConditionalGeneration as LlavaForConditionalGenerationTorch
    import torch
    import flax
    config = AutoConfig.from_pretrained("llava-hf/llava-1.5-7b-hf")
    print(config)

    model_torch= LlavaForConditionalGenerationTorch.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float32   ,
                                                      device_map="cpu",
                                                      )

    pixel_values = torch.ones((1, 3,336, 336))

    device='cuda'
    model_torch.vision_tower=model_torch.vision_tower.to(device)


    out_torch=model_torch.vision_tower(pixel_values.to(device))
    print(out_torch.keys())
    print(out_torch['last_hidden_state'].shape)


    params=convert_hf_to_flax_llava(model_torch.state_dict())



    # print(params['vision_tower']['vision_model']['embeddings'].keys())

    params=jax.tree_util.tree_map(jnp.array,params)

    pixel_values=jnp.ones((1,336,336,3))
    model=LlavaForConditionalGeneration(config)
    # params=model.init(jax.random.PRNGKey(1),pixel_values)
    out=model.apply({'params':params},pixel_values)
    #.last_hidden_state.shape
    print(out.last_hidden_state.shape)


    # print(np.array(out.last_hidden_state)-out_torch['last_hidden_state'].detach().cpu().numpy())
    diff=np.array(out.last_hidden_state)-out_torch['last_hidden_state'].detach().cpu().numpy()
    # diff = np.array(out.pooler_output) - out_torch['pooler_output'].detach().cpu().numpy()
    print(np.max(np.abs(diff)))
    print('here is different')






def test2():
    from transformers import  AutoConfig, LlavaForConditionalGeneration as LlavaForConditionalGenerationTorch
    import torch
    import flax
    config = AutoConfig.from_pretrained("llava-hf/llava-1.5-7b-hf")
    print(config)

    model_torch= LlavaForConditionalGenerationTorch.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float32   ,
                                                      device_map="cpu",
                                                      )

    del model_torch.language_model

    url = "kane.jpg"
    # image_stop = Image.open(requests.get(url, stream=True).raw)
    image = np.array(Image.open(url).resize((336,336))).reshape((-1,336,336,3))*2 -1

    print(image.shape)





    device='cpu'
    # pixel_values = torch.ones((1, 3,336, 336)).to(device)
    pixel_values = torch.from_numpy(einops.rearrange(image,'b h w c-> b c h w')).to(device)
    # while True:
    #     pass

    # model_torch.vision_tower=model_torch.vision_tower.to(device)


    # out_torch=model_torch.vision_tower(pixel_values.to(device))
    # print(out_torch.keys())
    # print(out_torch['last_hidden_state'].shape)


    # out_torch=get_image_features(model_torch,pixel_values)
    out_torch = model_torch.get_image_features( pixel_values,-2,'default')

    # out_torch = get_image_features(model_torch, pixel_values)


    params=convert_hf_to_flax_llava(model_torch.state_dict())

    # print(params['vision_tower']['vision_model']['embeddings'].keys())

    params=jax.tree_util.tree_map(jnp.array,params)

    # pixel_values=jnp.ones((1,336,336,3))
    pixel_values=jnp.array(image)
    model=LlavaForConditionalGeneration(config)
    # params=model.init(jax.random.PRNGKey(1),pixel_values)
    out=model.apply({'params':params},pixel_values)
    #.last_hidden_state.shape


    diff=np.array(out)-out_torch.detach().cpu().numpy()
    print(np.max(np.abs(diff)),np.mean(np.abs(diff)))
    print('here is different')


def test3():

    import flax
    config = AutoConfig.from_pretrained("llava-hf/llava-1.5-7b-hf")
    print(config)




    url = "kane.jpg"
    image = Image.open(url)

    conversation_2 = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        },
    ]
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    prompt_2 = processor.apply_chat_template(conversation_2, add_generation_prompt=True)

    inputs = processor(images=[image], text=[prompt_2], padding=True, return_tensors="pt")
    inputs_jax = processor(images=[image], text=[prompt_2], padding=True, return_tensors="jax")


    print(inputs_jax.keys())




    model_torch= LlavaForConditionalGenerationTorchTest.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float32   ,
                                                      device_map="cpu",
                                                       )


    # model_torch.forward=LlavaForConditionalGenerationTorchTest.forward

    del model_torch.language_model.model.layers
    out=model_torch(**inputs)


    print(out.shape)

    while True:
        pass




    url = "kane.jpg"
    # image_stop = Image.open(requests.get(url, stream=True).raw)
    image = np.array(Image.open(url).resize((336,336))).reshape((-1,336,336,3))*2 -1

    print(image.shape)
    device='cpu'
    # pixel_values = torch.ones((1, 3,336, 336)).to(device)
    pixel_values = torch.from_numpy(einops.rearrange(image,'b h w c-> b c h w')).to(device)
    out_torch = model_torch.get_image_features( pixel_values,-2,'default')
    params=convert_hf_to_flax_llava(model_torch.state_dict())
    params=jax.tree_util.tree_map(jnp.array,params)
    # pixel_values=jnp.ones((1,336,336,3))
    pixel_values=jnp.array(image)
    model=LlavaForConditionalGeneration(config)
    out=model.apply({'params':params},pixel_values)
    diff=np.array(out)-out_torch.detach().cpu().numpy()
    print(np.max(np.abs(diff)),np.mean(np.abs(diff)))
    print('here is different')


if __name__ == '__main__':
    jax.config.update('jax_platform_name', 'cpu')
    test3()




