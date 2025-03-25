from typing import Optional, Union, List, Tuple

import einops
import flax.traverse_util
import jax.random
import numpy as np
import torch
from transformers import AutoProcessor, AutoConfig, Cache
from transformers.models.gemma3 import Gemma3ForConditionalGeneration as Gemma3ForConditionalGenerationTorch
from transformers.models.gemma3.modeling_gemma3 import Gemma3MultiModalProjector, Gemma3CausalLMOutputWithPast

from MLLM_JAX.multinomial_sample import test_gemma3
from MLLM_JAX.mutinomial.gemma3.modeling_gemma3 import convert_torch_to_flax_gemma3,Gemma3ForConditionalGeneration
from MLLM_JAX.vision.siglip.modeling_siglip import SiglipVisionModel, convert_torch_to_flax_siglip
import jax.numpy as jnp




class Gemma3ForConditionalGenerationTorchMy(Gemma3ForConditionalGenerationTorch):

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **lm_kwargs,
    ) :


        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        is_training = token_type_ids is not None and labels is not None

        # Replace image id woth PAD if the image token if OOV, to avoid index-errors
        if input_ids is not None and self.config.image_token_index >= self.vocab_size:
            special_image_mask = input_ids == self.config.image_token_index
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0) + 1  # Gemma3 positions are 1-indexed

        # Merge text and images
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)

            if input_ids is None:
                special_image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.image_token_index, dtype=torch.long, device=inputs_embeds.device)
                )
            else:
                special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
                special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

            # if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
            #     image_tokens_in_text = (special_image_mask).sum(dim=1).sum(dim=0)[0]
            #     raise ValueError(
            #         f"Number of images does not match number of special image tokens in the input text. "
            #         f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
            #         "tokens from image embeddings."
            #     )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
            return inputs_embeds




def test_siglip_multi_projector():

    model_id = "google/gemma-3-4b-it"
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    from PIL import Image

    img=Image.open('kane.jpg').resize((896,896))
    img=np.array(img,dtype=np.float32)[None,...]/255.0


    print(config.vision_config)

    model_jax=Gemma3ForConditionalGeneration(config)
    shape=(1,896,896,3)
    # img_jax=jax.random.normal(jax.random.PRNGKey(1),shape)
    img_jax=(img*2)-1
    print(img_jax)

    model = Gemma3ForConditionalGenerationTorch.from_pretrained(
        model_id, device_map="cpu"
    ).eval()
    print(model.dtype)

    params=convert_torch_to_flax_gemma3(model.state_dict())
    print(params.keys())
    params=jax.tree_util.tree_map(lambda x:np.array(x),params)

    with jax.default_device(jax.devices("cpu")[0]):
        out_jax=jax.jit(model_jax.apply)({'params':params},None, img_jax)
    del params

    img_torch=einops.rearrange(img_jax,'b h w c-> b c h w')
    img_torch=torch.from_numpy(np.array(img_torch))
    with torch.no_grad():
        torch_out=model(img_torch)#['last_hidden_state']
        print(torch_out.shape)

    out_jax=np.array(out_jax)
    torch_out=torch_out.numpy()
    print(out_jax-torch_out)
    print(np.abs(out_jax-torch_out).max())



def test_gemma():

    model_id = "google/gemma-3-4b-it"
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model_jax = Gemma3ForConditionalGeneration(config)
    processor = AutoProcessor.from_pretrained(model_id)

    messages = [
       [
           {
               "role": "system",
               "content": [{"type": "text", "text": "You are a helpful assistant."}]
           },
           {
               "role": "user",
               "content": [
                   {"type": "image", "image": "kane.jpg"},
                   {"type": "image", "image": "kane.jpg"},
                   {"type": "text", "text": "Is this two img same?"}
               ]
           }
       ]
    ]


    model = Gemma3ForConditionalGenerationTorch.from_pretrained(
        model_id, device_map="cpu"
    ).eval()

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=model.dtype)

    # 'input_ids', 'attention_mask', 'token_type_ids', 'pixel_values'
    print(inputs.keys())
    with torch.no_grad():
        torch_out=model(**inputs,)


    params = convert_torch_to_flax_gemma3(model.state_dict())
    print(params.keys())
    params = jax.tree_util.tree_map(lambda x: np.array(x), params)
    d=dict()
    with jax.default_device(jax.devices("cpu")[0]):

        for key in inputs:
            d[key]=np.array(inputs[key])

        d['pixel_values']=einops.rearrange(d['pixel_values'],'b c h w -> b h w c')
        out_jax = jax.jit(model_jax.apply)({'params': params}, **d)


    out_jax = np.array(out_jax)
    torch_out = torch_out.numpy()
    print(out_jax - torch_out)
    print(np.abs(out_jax - torch_out).max())





    # with torch.inference_mode():
    #     generation = model.generate(**inputs, max_new_tokens=500, do_sample=False)
    #     generation = generation[0][input_len:]
    #
    # decoded = processor.decode(generation, skip_special_tokens=True)
    # print(decoded)

    # **Overall Impression:** The image is a close-up shot of a vibrant garden scene,
    # focusing on a cluster of pink cosmos flowers and a busy bumblebee.
    # It has a slightly soft, natural feel, likely captured in daylight.


if __name__=="__main__":
    # jax.config.update('jax_platform_name', 'cpu')
    # test_siglip_multi_projector()
    test_gemma3()