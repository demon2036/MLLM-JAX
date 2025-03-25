import torch
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from language.qwen2.configuration_qwen2 import Qwen2Config, init_cache
from language.llama import convert_torch_to_flax_llama, test_vicuna, test_vicuna_torch
from language.qwen2.modular_qwen2 import Qwen2ForCausalLM


def test_qwen2():
    dtype = jnp.bfloat16
    # dtype = jnp.float32
    max_cache_length=512

    model_path = 'Qwen/Qwen2-7B-Instruct'

    # jax.config.update('jax_platform_name', 'cpu')
    # variables=model.init(rng,x.astype(jnp.int32),x.astype(jnp.int32))
    # params=variables['params']

    from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor, LlamaTokenizerFast, \
        AutoTokenizer, AutoModelForCausalLM,AutoConfig,Qwen2Tokenizer


    llama_config=Qwen2Config()
    print(llama_config)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(config)
    # config.init_cache=llama_config.init_cache
    cache = init_cache(config,1,max_cache_length=max_cache_length, dtype=dtype)

    # while True:
    #     pass

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load the base model with adapters on top
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        # {"role": "user", "content": "Who are you?"},
        # {"role": "user", "content": "Give me a short introduction to large language model."},
        {"role": "user", "content": "Whose your daddy?"},
        #
        # {"role": "assistant", "content": "A large language model is a type of artificial intelligence (AI) model that"},
        # {"role": "assistant", "content": "A large language model is a type of"},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
                                           # add_generation_prompt=False,continue_final_message=True)
    print(prompt)

    inputs = tokenizer(prompt, return_tensors="jax")


    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    state_dict = model.state_dict()
    params = convert_torch_to_flax_llama(state_dict)

    params = jax.tree_util.tree_map(lambda x: jnp.asarray(np.array(x), dtype=dtype), params)

    pad_attention = jnp.pad(attention_mask, ((0, 0), (0, max_cache_length - attention_mask.sum(axis=1)[0])))

    input_ids = input_ids
    position_ids = jnp.arange(0, input_ids.shape[1])[None, ...]

    # llama_config = LlamaConfig(max_cache_length=1024)

    model = Qwen2ForCausalLM(config)

    b, n, d = shape = 1, 1, 768

    jit_infer = jax.jit(model.apply)

    logits, cache = jit_infer({'params': params}, input_ids=input_ids, position_ids=position_ids,
                              attention_mask=pad_attention, cache=cache)
    # select_ids = jnp.argmax(logits[:, -1], axis=1)
    # decoded_token = tokenizer.decode(select_ids)
    # print(decoded_token, end='', flush=True)

    # print(out)

    position_ids = position_ids[:, -1][..., None]
    max_decode_length = 250
    res = []
    # exit_token_ids=tokenizer.encode(tokenizer.st)[0]
    exit_token_ids = 151645
    print(f'{tokenizer.eos_token=} , {exit_token_ids=}')



    for i in tqdm(range(max_decode_length)):
        select_ids = jnp.argmax(logits[:, -1], axis=1)
        print(select_ids)
        res.append(select_ids)

        if select_ids[0] == exit_token_ids:
            break

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
        tokenizer.batch_decode(np.array(res).reshape(1, -1), skip_special_tokens=False,
                               clean_up_tokenization_spaces=False)[
            0]
    # output=tokenizer.decode(np.array(res), skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(output)



def test_qwen_torch():
    dtype = jnp.bfloat16
    # model_path = 'Qwen/Qwen2-7B-Instruct'
    model_path = '/autodl-tmp/qwen/Qwen2-7B-Instruct'
    # model_path = './autodl-tmp/qwen/Qwen2-7B-Instruct'

    from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor, LlamaTokenizerFast, \
        AutoTokenizer, AutoModelForCausalLM,AutoConfig,Qwen2Tokenizer


    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,)
    from transformers.generation import configuration_utils

    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load the base model with adapters on top
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        quantization_config=quantization_config,
    )


    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        # {"role": "user", "content": "Who are you?"},
        {"role": "user", "content": "Give me a short introduction to large language model."},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(prompt)


    device = 'cuda'
    # device = 'cpu'
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out=model.generate(**inputs,
                       do_sample=False,
                       # temperature=0.1,
                       # top_k=1000,
                       # top_p=1.0,
                       max_new_tokens=512)
    print(out)
    output=tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(output)
    #
    #
    # print('\n'*10)
    # while True:


if __name__=="__main__":
    # test_qwen_torch()
    test_qwen2()

    # cache=test_vicuna("I'm a")
    """
    cache2 = test_vicuna("I'm a language model called")

    layer_name = f'layer_{28}'
    layer_cache = cache[layer_name]['v'][0,0,55:88,0]
    layer_cache2 = cache2[layer_name]['v'][0,0,55:88,0]

    #(batch_size, num_heads, cache_size, head_dim)
    print(layer_cache)
    print('\n'*2)
    print(layer_cache2)

    jnp.allclose(layer_cache,layer_cache2)
    print(layer_cache-layer_cache2)
    print(jnp.sum(jnp.abs(layer_cache-layer_cache2)))

    """









    # test_vicuna_torch()