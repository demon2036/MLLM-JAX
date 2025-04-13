import einops
import flax.traverse_util
import jax.random
import numpy as np
import torch
from transformers import AutoProcessor, AutoConfig
from transformers.models.gemma3 import Gemma3ForConditionalGeneration as Gemma3ForConditionalGenerationTorch
from transformers.models.gemma3.modeling_gemma3 import Gemma3MultiModalProjector

from MLLM_JAX.mutinomial.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration
from MLLM_JAX.vision.siglip.modeling_siglip import SiglipVisionModel, convert_torch_to_flax_siglip
import jax.numpy as jnp

def test_siglip():

    model_id = "google/gemma-3-4b-it"
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    from PIL import Image

    img=Image.open('kane.jpg').resize((896, 896))
    img=np.array(img,dtype=np.float32)[None,...]/255.0


    print(config.vision_config)

    siglip_jax=Gemma3ForConditionalGeneration(config)
    shape=(1,896,896,3)
    img_jax=jax.random.normal(jax.random.PRNGKey(1),shape)
    # img_jax=(img*2)-1
    print(img_jax)

    model = Gemma3ForConditionalGenerationTorch.from_pretrained(
        model_id, device_map="cpu"
    ).eval()
    print(model.dtype)


    siglip=model.vision_tower
    # del siglip.vision_model.encoder
    params=convert_torch_to_flax_siglip(siglip.state_dict())
    params=jax.tree_util.tree_map(lambda x:np.array(x),params)

    with jax.default_device(jax.devices("cpu")[0]):
        out_jax=jax.jit(siglip_jax.apply)({'params':params}, img_jax)


    del params

    img_torch=einops.rearrange(img_jax,'b h w c-> b c h w')
    img_torch=torch.from_numpy(np.array(img_torch))
    with torch.no_grad():
        torch_out=siglip(img_torch)['last_hidden_state']
        print(torch_out.shape)

    out_jax=np.array(out_jax)
    torch_out=torch_out.numpy()
    print(out_jax-torch_out)
    print(out_jax.max(),out_jax.min())



def test_gemma():

    model_id = "google/gemma-3-4b-it"



    processor = AutoProcessor.from_pretrained(model_id)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [

                {"type": "image","image": "kane.jpg"},
                {"type": "text", "text": "Describe this image in detail."}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to('cuda', dtype=torch.bfloat16)

    # 'input_ids', 'attention_mask', 'token_type_ids', 'pixel_values'
    print(inputs.keys())

    # for x in inputs.keys():
    #
    #
    #     if x=='input_ids':
    #         print(inputs[x][0])
    #         print(processor.tokenizer.decode(inputs[x][0] ))
    #     elif x=='pixel_values':
    #         print(inputs[x].max(),inputs[x].min())
    #
    #     print(x,inputs[x].shape)




    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto"
    ).eval()

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=500, do_sample=False)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)

    # **Overall Impression:** The image is a close-up shot of a vibrant garden scene,
    # focusing on a cluster of pink cosmos flowers and a busy bumblebee.
    # It has a slightly soft, natural feel, likely captured in daylight.


if __name__=="__main__":
    # test_siglip()
    test_gemma()