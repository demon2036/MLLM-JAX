import random
from typing import Any

import random
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from MLLM_JAX.language.llama.llama import convert_torch_to_flax_llama, LlamaJaxConfig
from MLLM_JAX.language.qwen2.configuration_qwen2 import init_cache, pad_cache, pad_cache_right
from MLLM_JAX.language.qwen2.modular_qwen2 import Qwen2ForCausalLM
from MLLM_JAX.utils import match_partition_rules, get_partition_rules_llama, get_jax_mesh2
from sanple_utils import _greedy_sampling, _temperature_sampling

content = """1+1=2 1+2=?
"""

dtype = jnp.bfloat16


def get_params(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
    )
    state_dict = model.state_dict()
    params = convert_torch_to_flax_llama(state_dict)
    # params = jax.tree_util.tree_map(lambda x: jnp.asarray(np.array(x), dtype=dtype), params)
    params = jax.tree_util.tree_map(lambda x: np.array(x), params)

    return params


def get_model(mesh,model_path='Qwen/Qwen2-3B-Instruct', only_model=False):
    # model_path = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
    # model_path = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B'
    # model_path = 'Qwen/Qwen2-0.5B-Instruct'
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(config)
    # Load the base model with adapters on top

    params = get_params(model_path)
    jax_config = LlamaJaxConfig(mesh=mesh)
    model = Qwen2ForCausalLM(config, jax_config)

    if only_model:
        return model


    def init_fn(params):
        return params

    state_shapes = jax.eval_shape(init_fn, params, )

    train_state_partition = match_partition_rules(get_partition_rules_llama(), state_shapes)
    train_state_sharding = jax.tree_util.tree_map(lambda x: jax.sharding.NamedSharding(mesh, x), train_state_partition)

    params = jax.tree_util.tree_map(lambda x, d: jnp.asarray(x, dtype=dtype, device=d), params, train_state_sharding)

    params = jax.jit(init_fn,
                     # donate_argnums=(0,),
                     out_shardings=train_state_sharding)(params)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, params, tokenizer


# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     # {"role": "system", "content": "You are a helpful and harmless assistant. You should think step-by-step."},
#     {"role": "user", "content": "Who are you?"},
#     # {"role": "user", "content": "Give me a short introduction to large language model."},
#     # {"role": "user", "content": "1+1=2 1+2=?"},
#     # {"role": "user", "content": "Mnist examples in jax?"},
#     # {"role": "user", "content": content, },
#     # {"role": "assistant", "content": "A large language model is a type of artificial intelligence (AI) model that"},
#     # {"role": "assistant", "content": "A large language model is a type of"},
# ]


messages = [
    #     [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     # {"role": "system", "content": "You are a helpful and harmless assistant. You should think step-by-step."},
    #     {"role": "user", "content": "Who are you?"},
    #     # {"role": "user", "content": "Give me a short introduction to large language model."},
    #     # {"role": "user", "content": "1+1=2 1+2=?"},
    #     # {"role": "user", "content": "Mnist examples in jax?"},
    #     # {"role": "user", "content": content, },
    #     # {"role": "assistant", "content": "A large language model is a type of artificial intelligence (AI) model that"},
    #     # {"role": "assistant", "content": "A large language model is a type of"},
    # ],

    [
        {"role": "system", "content": "You are a helpful assistant."},
        # {"role": "user", "content": "Who are you?"},
        {"role": "user", "content": " 1+1=?"},
    ],

    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who are you?"},
    ],

]


@chex.dataclass
class SampleState:
    decoding_step: jnp.int32
    num_input_tokens: jnp.ndarray
    token_buffer: jnp.ndarray
    positions: jnp.ndarray
    cache: Any
    attention_mask: jnp.ndarray
    next_token_buffer: jnp.ndarray
    key: jnp.ndarray
    dones: jnp.ndarray
    sample_steps: jnp.ndarray
    # logits_buffer:


def create_sample_state(input_ids_pad, position_ids, cache, pad_attention, true_length, decoding_step=0):
    sample_state = SampleState(decoding_step=decoding_step, num_input_tokens=true_length, token_buffer=input_ids_pad,
                               positions=position_ids, cache=cache, attention_mask=pad_attention,
                               next_token_buffer=jnp.zeros((pad_attention.shape[0])),
                               key=jax.random.PRNGKey(random.randint(0, 2036)),
                               dones=jnp.zeros((pad_attention.shape[0]), dtype=jnp.bool),
                               sample_steps=jnp.zeros((pad_attention.shape[0]), dtype=jnp.int32)
                               )
    return sample_state


class Sampler:
    def __init__(self, model, params, tokenizer, max_length=8192, prefill_length=128, cache=None):
        self.model = model
        self.tokenizer = tokenizer
        # self.params = params
        # self.max_length=max_length
        self.prefill_length = prefill_length
        self.cache = cache
        self.jit_infer_prefill = jax.jit(self.model.apply)
        self.jit_infer_step = jax.jit(self.infer)

        # self.sample_fn=_nucleus_sampling
        # self.sample_fn = _greedy_sampling
        self.sample_fn=_temperature_sampling

        self.prefill_bucket = [
             256, 512, 1024, 2048, 4096, 8192
        ]


    def infer(self, sample_state: SampleState, params):
        i = sample_state.decoding_step
        last_token = sample_state.token_buffer[:, i].reshape((sample_state.token_buffer.shape[0], 1))
        positions = sample_state.positions
        cache = sample_state.cache
        mask = sample_state.attention_mask

        logits, cache = self.model.apply({'params': params}, input_ids=last_token, position_ids=positions,
                                         attention_mask=mask, cache=cache)

        key, key2 = jax.random.split(sample_state.key)
        next_token_predict = self.sample_fn(key2, logits[:, -1])

        dones = sample_state.dones | (next_token_predict == self.tokenizer.eos_token_id)

        sample_state.dones=dones

        sample_state.sample_steps += 1 - sample_state.dones

        sample_state.key = key
        sample_state.attention_mask = sample_state.attention_mask.at[:, i + 1].set(1)
        sample_state.positions += 1
        sample_state.token_buffer = sample_state.token_buffer.at[:, i + 1].set(next_token_predict)
        sample_state.next_token_buffer = next_token_predict
        sample_state.decoding_step += 1
        sample_state.cache = cache
        return sample_state

    def preprocess_prompt(self, prompt, max_cache_length=8192):
        inputs = self.tokenizer(prompt, return_tensors="jax", )
        input_ids = inputs['input_ids']
        l = input_ids.shape[1]
        attention_mask = inputs['attention_mask']
        pad_attention = jnp.zeros_like(attention_mask).at[:, 0].set(1)

        input_ids_pad = jnp.pad(input_ids, ((0, 0), (0, max_cache_length - attention_mask.sum(axis=1)[0])),
                                constant_values=self.tokenizer.eos_token_id)
        pad_attention = jnp.pad(pad_attention, ((0, 0), (0, max_cache_length - attention_mask.sum(axis=1)[0])))

        position_ids = jnp.arange(0, 1)[None, ...]
        return input_ids_pad, pad_attention, position_ids, l

    def preprocess_prompt_prefill(self, prompt, prefill_length=128):
        inputs = self.tokenizer(prompt, return_tensors="jax", padding=True, padding_side="left")
        input_ids = inputs['input_ids']

        position_ids = inputs['attention_mask'].cumsum(-1) - 1
        position_ids = jnp.where(inputs['attention_mask'] == 0, 1, position_ids)

        prefill_length = self.find_ceil(input_ids.shape[1])
        # prefill_length = input_ids.shape[1]

        attention_mask = inputs['attention_mask']
        input_ids_pad = jnp.pad(input_ids, ((0, 0), (prefill_length - input_ids.shape[1], 0)),
                                constant_values=self.tokenizer.eos_token_id)

        pad_attention = jnp.pad(attention_mask, ((0, 0), (prefill_length - input_ids.shape[1], 0)))
        pad_position_ids = jnp.pad(position_ids, ((0, 0), (prefill_length - input_ids.shape[1], 0)))

        return input_ids_pad, pad_attention, pad_position_ids, prefill_length

    def prepare_from_prefill_to_decode(self, cache, input_ids_pad, pad_attention, position_ids, max_length=8192):

        b, prefill_length = input_ids_pad.shape

        cache = pad_cache_right(cache, prefill_length, max_length, )
        input_ids_pad = jnp.pad(input_ids_pad, ((0, 0), (0, max_length)),
                                constant_values=self.tokenizer.eos_token_id)

        pad_attention = jnp.pad(pad_attention, ((0, 0), (0, max_length)),
                                constant_values=0)
        pad_attention = pad_attention.at[:, prefill_length].set(1)
        position_ids = position_ids[:, -1:] + 1

        return cache, input_ids_pad, pad_attention, position_ids

    def prefill(self):
        pass

    def find_ceil(self, input):

        for num in self.prefill_bucket:  # 确保列表是有序的
            if num >= input:
                return num
        return None  # 如果 input 大于所有数字，返回 None

    def generate_prefill_auto_regressive(self, prompt, prefill_length=20, max_length=8192,params=None):

        input_ids_pad, pad_attention, position_ids, prefill_length = self.preprocess_prompt_prefill(prompt,
                                                                                                    prefill_length)

        print(f'{prefill_length=}')

        cache = init_cache(self.model.config, input_ids_pad.shape[0], max_cache_length=prefill_length, dtype=dtype)
        logits, cache = self.jit_infer_prefill({'params': params}, input_ids=input_ids_pad,
                                               position_ids=position_ids,
                                               attention_mask=pad_attention, cache=cache)

        cache, input_ids_pad, pad_attention, position_ids = self.prepare_from_prefill_to_decode(cache, input_ids_pad,
                                                                                                pad_attention,
                                                                                                position_ids,
                                                                                                max_length=max_length)

        next_token_predict = jnp.argmax(logits[:, - 1], axis=1)
        input_ids_pad = input_ids_pad.at[:, prefill_length].set(next_token_predict)
        sample_state = create_sample_state(input_ids_pad=input_ids_pad, position_ids=position_ids, cache=cache,
                                           pad_attention=pad_attention, true_length=prefill_length,
                                           decoding_step=prefill_length)

        exit_token_ids = self.tokenizer.eos_token_id
        # res = [next_token_predict[1:]]

        for i in tqdm(range(max_length)):
            sample_state = self.jit_infer_step(sample_state, params)
            # select_ids = sample_state.next_token_buffer
            # res.append(select_ids[1:])
            # ans_batch = ans_batch.at[:, i + 1].set(select_ids)
            if jnp.all(sample_state.dones):
                break
        texts=[]
        for i,step in enumerate(sample_state.sample_steps):
            output = \
                self.tokenizer.batch_decode(np.array(sample_state.token_buffer[i, prefill_length:prefill_length+step+1]).reshape(1, -1),
                                            skip_special_tokens=False,
                                            clean_up_tokenization_spaces=False)

            texts.extend(output)
            print(output)
        # print(texts)

        # while True:
        #     pass

        return texts


def test_qwen2_fast_jit_sample2():
    max_cache_length = 32
    mesh = get_jax_mesh2("-1,1,1")
    model, params, tokenizer, cache = get_model(mesh, max_cache_length=max_cache_length)
    exit_token_ids = tokenizer.eos_token_id
    print(f'{tokenizer.eos_token=} ,{tokenizer.eos_token_id=}, {exit_token_ids=}')
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)

    sampler = Sampler(model, params, tokenizer, max_length=max_cache_length)
    print('hi hi')
    sampler.generate_prefill_auto_regressive(prompt, max_length=max_cache_length)








if __name__ == "__main__":
    # test_qwen_torch()

    test_qwen2_fast_jit_sample2()
    # test_qwen2_fast_jit_sample2()
