import torch
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from language.qwen2.configuration_qwen2 import Qwen2Config, init_cache
from language.llama import convert_torch_to_flax_llama, test_vicuna, test_vicuna_torch, LlamaConfig, LlamaForCausalLM
from language.qwen2.modular_qwen2 import Qwen2ForCausalLM
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor, LlamaTokenizerFast, \
    AutoTokenizer, AutoModelForCausalLM

def test_vicuna():
    # prompt = "I'm a language model called"
    prompt = "I'm a language model called Vicuna"
    # prompt2 = "I'm a language model called Vicuna,"
    prompt2 = "I'm a language model called Vicuna,"






    # prompt2 = "I'm"
    # prompt2 = "I'm a language model called Vicuna, and I was "

    dtype = jnp.bfloat16
    max_cache_length=96
    tokenizer = AutoTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
    llama_config = LlamaConfig(max_cache_length=max_cache_length)


    print(tokenizer(prompt, return_tensors=None))
    print()
    print(tokenizer(prompt2, return_tensors=None))



    # while True:
    #     pass

    model = LlamaForCausalLM(llama_config)
    cache = llama_config.init_cache(1, dtype=dtype)
    cache2 = llama_config.init_cache(1, dtype=dtype)

    # Load the base model with adapters on top
    model_torch = AutoModelForCausalLM.from_pretrained(
        'lmsys/vicuna-7b-v1.5',
        torch_dtype=torch.float16,
    )
    state_dict = model_torch.state_dict()
    params = convert_torch_to_flax_llama(state_dict)
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(np.array(x), dtype=dtype), params)

    # prompt="I'm a language model called Vicuna, and I was trained by Large Model Systems Organization"
    # prompt="I'm a language model called Vicuna"
    prompt_input1 = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hello! ASSISTANT: Hi!</s>USER: Who are you? ASSISTANT: {prompt}"
    prompt_input2 = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hello! ASSISTANT: Hi!</s>USER: Who are you? ASSISTANT: {prompt2}"
    # prompt_input2 = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hello! ASSISTANT: Hi!</s>USER: Who are you? ASSISTANT:"
    jit_infer = jax.jit(model.apply)


    inputs_1 = tokenizer(prompt_input1, return_tensors="jax")
    input_ids_1 = inputs_1['input_ids']
    attention_mask_1 = inputs_1['attention_mask']
    pad_attention_1 = jnp.pad(attention_mask_1, ((0, 0), (0, max_cache_length - attention_mask_1.sum(axis=1)[0])))
    position_ids_1 = jnp.arange(0, input_ids_1.shape[1])[None, ...]

    logits, cache = jit_infer({'params': params}, input_ids=input_ids_1, position_ids=position_ids_1,
                              attention_mask=pad_attention_1, cache=cache)

    position_ids = position_ids_1[:, -1][..., None]
    max_decode_length = 250
    res = []

    exit_token_ids = tokenizer.eos_token_id
    print(f'{tokenizer.eos_token=} , {exit_token_ids=}')

    for i in tqdm(range(max_decode_length)):
        select_ids = jnp.argmax(logits[:, -1], axis=1)
        print(select_ids)
        res.append(select_ids)
        if select_ids[0] == exit_token_ids:
            break

        input_ids = select_ids[..., None]
        position_ids += 1
        pad_attention = pad_attention_1.at[:, attention_mask_1.sum(axis=1)[0] + i].set(1)



        if i==1:
            old_cache=cache
            inputs_2 = tokenizer(prompt_input2, return_tensors="jax")
            input_ids_2 = inputs_2['input_ids']
            attention_mask_2 = inputs_2['attention_mask']
            pad_attention_2 = jnp.pad(attention_mask_2,
                                      ((0, 0), (0, max_cache_length - attention_mask_2.sum(axis=1)[0])))
            position_ids_2 = jnp.arange(0, input_ids_2.shape[1])[None, ...]

            logits, cache = jit_infer({'params': params}, input_ids=input_ids_2, position_ids=position_ids_2,
                                      attention_mask=pad_attention_2, cache=cache2)

            # layer_name = f'layer_{0}'
            # #position_ids_2[0][-1]
            # layer_cache = cache[layer_name]['v'][0,0,:,0]
            # layer_cache2 = old_cache[layer_name]['v'][0,0,:,0]
            #
            # #(batch_size, num_heads, cache_size, head_dim)
            # print(layer_cache)
            # print('\n'*2)
            # print(layer_cache2)
            #
            #
            # print(layer_cache-layer_cache2)
            # print(jnp.allclose(layer_cache,layer_cache2),jnp.sum(jnp.abs(layer_cache-layer_cache2)))


            # while    True:
            #     params



        logits, cache = jit_infer({'params': params}, input_ids=input_ids, position_ids=position_ids,
                                  attention_mask=pad_attention, cache=cache)


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
    print(output)

    return cache


if __name__=="__main__":
    # test_qwen_torch()
    # test_qwen2()
    cache=test_vicuna()

    # cache2 = test_vicuna("I'm a language model called")
    #
    # layer_name = f'layer_{28}'
    # layer_cache = cache[layer_name]['v'][0,0,55:88,0]
    # layer_cache2 = cache2[layer_name]['v'][0,0,55:88,0]
    #
    # #(batch_size, num_heads, cache_size, head_dim)
    # print(layer_cache)
    # print('\n'*2)
    # print(layer_cache2)
    #
    # jnp.allclose(layer_cache,layer_cache2)
    # print(layer_cache-layer_cache2)
    # print(jnp.sum(jnp.abs(layer_cache-layer_cache2)))











    # test_vicuna_torch()