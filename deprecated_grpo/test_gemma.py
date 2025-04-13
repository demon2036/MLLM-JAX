# pip install accelerate
import flax.traverse_util
import jax.tree_util
import numpy as np
from transformers import AutoProcessor, AutoConfig, AutoModelForCausalLM
from transformers.models.gemma3 import Gemma3ForConditionalGeneration
import torch

from MLLM_JAX.language.gemma3.modeling_gemma3 import convert_torch_to_flax_gemma3, GemmaForCausalLM
from MLLM_JAX.utils import get_jax_mesh2, match_partition_rules, get_partition_rules_llama
from app.test_qwen import Sampler
import jax.numpy as jnp
import jax


def test_torch():

    model_id = "google/gemma-3-4b-it"

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto"
    ).eval()

    processor = AutoProcessor.from_pretrained(model_id)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
                {"type": "text", "text": "Describe this image in detail."}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)

# **Overall Impression:** The image is a close-up shot of a vibrant garden scene,
# focusing on a cluster of pink cosmos flowers and a busy bumblebee.
# It has a slightly soft, natural feel, likely captured in daylight.




def test_torch2():

    model_id = "google/gemma-3-4b-it"

    # from transformers import BitsAndBytesConfig
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    # )

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto",torch_dtype=torch.float32,#quantization_config=quantization_config
    ).eval()

    processor = AutoProcessor.from_pretrained(model_id)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Who are you."*500}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.float32)



    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]


    print(generation)
    decoded = processor.decode(generation, skip_special_tokens=False)
    print(decoded)

    #
    # decoded = processor.decode(torch.ones(1,dtype=torch.int64)*106, skip_special_tokens=False)
    # print(decoded)






def get_params(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
    )
    # state_dict = model.state_dict()

    state_dict=flax.traverse_util.unflatten_dict(model.state_dict(),sep='.')

    state_dict=flax.traverse_util.flatten_dict(state_dict,sep='.')

    params = convert_torch_to_flax_gemma3(state_dict)
    # params = jax.tree_util.tree_map(lambda x: jnp.asarray(np.array(x), dtype=dtype), params)
    params = jax.tree_util.tree_map(lambda x: np.array(x), params)

    return params


def get_model(mesh, max_cache_length=8192):
    model_path = "google/gemma-3-4b-it"

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)


    model = GemmaForCausalLM(config.text_config)
    params = get_params(model_path)

    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer=processor.tokenizer
    def init_fn(params):
        return params

    state_shapes = jax.eval_shape(init_fn, params, )

    train_state_partition = match_partition_rules(get_partition_rules_llama(), state_shapes)
    train_state_sharding = jax.tree_util.tree_map(lambda x: jax.sharding.NamedSharding(mesh, x), train_state_partition)

    params = jax.tree_util.tree_map(lambda x, d: jnp.asarray(x, dtype=jnp.bfloat16, device=d), params, train_state_sharding)

    params = jax.jit(init_fn,
                     # donate_argnums=(0,),
                     out_shardings=train_state_sharding)(params)

    return model, params, tokenizer




def test_jax():
    mesh = get_jax_mesh2("1,1,-1")

    model, params, tokenizer=get_model(mesh)

    sampler=Sampler(model, params, tokenizer, mesh=mesh)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Who are you."*500}
            ]
        }
    ]



    prompt = sampler.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(prompt)

    res_tokens = []
    for i, token in enumerate(
            sampler.generate_prefill_auto_regressive(prompt, max_length=8192, stream=True)):
        res_tokens.append(token)
        if (i + 1) % 100 == 0:
            current_text = sampler.tokenizer.decode(
                # np.array(next_token),
                np.array(res_tokens).reshape(-1),
                skip_special_tokens=True
            )
            # res_tokens = []
            # 检查是否生成结束 token
            # SSE 格式（每个消息前缀 "data:"），确保客户端能实时接收
            # yield f"{current_text}"
            # await asyncio.sleep(0.0001)  # 根据需要调节间隔

    current_text = sampler.tokenizer.decode(
        # np.array(next_token),
        np.array(res_tokens).reshape(-1),
        skip_special_tokens=True
    )
    print(current_text)







if __name__=="__main__":
    test_jax()
    # test_torch2()
    # test_torch()