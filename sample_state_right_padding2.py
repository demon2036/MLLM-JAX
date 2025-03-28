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

from jax.sharding import PartitionSpec as P

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
    model_path = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
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

    # [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": "Who are you?"},
    # ],

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
    def __init__(self, model, params, tokenizer, max_length=8192, prefill_length=128, cache=None,mesh=None):
        self.model = model
        self.tokenizer = tokenizer
        self.params = params
        # self.max_length=max_length
        self.prefill_length = prefill_length
        self.cache = cache
        self.jit_infer_prefill = jax.jit(self.model.apply)
        self.jit_infer_step = jax.jit(self.infer)


        self.sample_fn=_greedy_sampling


        self.prefill_bucket = [
            128, 256, 512, 1024, 2048, 4096,8192,16384,16384*2
        ]


        print(mesh)
        data_sharding = jax.sharding.NamedSharding(mesh, P('dp', 'tp'))

        def init_data(data):
            return data

        self.jit_init_data = jax.jit(init_data, out_shardings=data_sharding,)


    def infer(self, sample_state: SampleState, params):
        i = sample_state.decoding_step
        last_token = sample_state.token_buffer[:, i].reshape((sample_state.token_buffer.shape[0], 1))

        logits, cache = self.model.apply({'params': params}, input_ids=last_token, position_ids=sample_state.positions,
                                         attention_mask=sample_state.attention_mask, cache=sample_state.cache)



        key,key2=jax.random.split(sample_state.key)


        next_token_predict = jnp.where(i < sample_state.num_input_tokens - 1,
                                       sample_state.token_buffer[:, i + 1],
                                       # jnp.argmax(logits[:, -1], axis=1)
                                       self.sample_fn(key2,logits[:,-1])
                                       )



        sample_state.key=key
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
        inputs = self.tokenizer(prompt, return_tensors="jax", )
        input_ids = inputs['input_ids']
        true_length = input_ids.shape[1]

        # prefill_length = self.find_ceil(true_length)
        prefill_length=true_length
        # prefill_length = 32

        attention_mask = inputs['attention_mask']
        input_ids_pad = jnp.pad(input_ids, ((0, 0), (0, prefill_length - attention_mask.sum(axis=1)[0])),
                                constant_values=self.tokenizer.eos_token_id)

        pad_attention = jnp.pad(attention_mask, ((0, 0), (0, prefill_length - attention_mask.sum(axis=1)[0])))

        position_ids = jnp.arange(0, input_ids_pad.shape[1])[None, ...]
        pad_position_ids=position_ids
        # pad_position_ids = jnp.pad(position_ids, ((0, 0), (0, prefill_length - attention_mask.sum(axis=1)[0])))

        return input_ids_pad, pad_attention, pad_position_ids, true_length, prefill_length

    def prepare_from_prefill_to_decode(self, cache, input_ids_pad, pad_attention, true_length, max_length=8192):

        b, prefill_length = input_ids_pad.shape

        cache = pad_cache(cache, prefill_length, max_length, true_length=true_length)
        input_ids_pad = jnp.pad(input_ids_pad, ((0, 0), (0, max_length - prefill_length)),
                                constant_values=self.tokenizer.eos_token_id)

        pad_attention = jnp.pad(pad_attention, ((0, 0), (0, max_length - prefill_length)),
                                constant_values=0)

        pad_attention = pad_attention.at[:, true_length].set(1)
        position_ids = jnp.arange(true_length, true_length + 1)[None, ...]
        return cache, input_ids_pad, pad_attention, position_ids

    def prefill(self):
        pass

    def find_ceil(self, input ):

        for num in self.prefill_bucket:  # 确保列表是有序的
            if num >= input:
                return num
        return None  # 如果 input 大于所有数字，返回 None


    def generate_prefill_auto_regressive(self, prompt, prefill_length=20, max_length=8192, stream=False):

        input_ids_pad, pad_attention, position_ids, true_length, prefill_length = self.preprocess_prompt_prefill(prompt,
                                                                                                                 prefill_length)

        print(f'{prefill_length=}')

        # max_length=max(8192,prefill_length*2)
        # cache = init_cache(self.model.config, 1, max_cache_length=prefill_length, dtype=dtype)
        cache = init_cache(self.model.config, input_ids_pad.shape[0], max_cache_length=prefill_length, dtype=dtype,shard_method=None)
        # input_ids_pad, pad_attention, position_ids,cache=self.jit_init_data((input_ids_pad, pad_attention, position_ids,cache))

        logits, cache = self.jit_infer_prefill({'params': self.params}, input_ids=input_ids_pad,
                                               position_ids=position_ids,
                                               attention_mask=pad_attention, cache=cache)
        print(max_length,true_length)
        cache, input_ids_pad, pad_attention, position_ids = self.prepare_from_prefill_to_decode(cache, input_ids_pad,
                                                                                                pad_attention,
                                                                                                true_length,
                                                                                                max_length=max_length)

        next_token_predict = jnp.argmax(logits[:, true_length - 1], axis=1)

        # next_token_predict = jnp.argmax(logits[:, - 1], axis=1)

        input_ids_pad = input_ids_pad.at[:, true_length].set(next_token_predict)
        sample_state = create_sample_state(input_ids_pad=input_ids_pad, position_ids=position_ids, cache=cache,
                                           pad_attention=pad_attention, true_length=true_length,
                                           decoding_step=true_length)


        exit_token_ids = self.tokenizer.eos_token_id
        exit_token_ids=106



        res = [next_token_predict]
        for i in tqdm(range(max_length - true_length)):
            sample_state = self.jit_infer_step(sample_state, self.params)
            select_ids = sample_state.next_token_buffer
            res.append(select_ids)



            if select_ids[0] == exit_token_ids:
                output = \
                    self.tokenizer.batch_decode(np.array(res).reshape(1, -1), skip_special_tokens=False,
                                                clean_up_tokenization_spaces=False)[
                        0]


                break

        output = \
            self.tokenizer.batch_decode(np.array(res).reshape(1, -1), skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)[
                0]

        print(output)


def test_qwen2_fast_jit_sample2():
    max_cache_length = 1024
    mesh = get_jax_mesh2("1,1,-1")
    model, params, tokenizer = get_model(mesh, )
    exit_token_ids = tokenizer.eos_token_id
    print(f'{tokenizer.eos_token=} ,{tokenizer.eos_token_id=}, {exit_token_ids=}')
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)

    sampler = Sampler(model, params, tokenizer, max_length=max_cache_length,mesh=mesh)
    print('hi hi')
    sampler.generate_prefill_auto_regressive(prompt, max_length=max_cache_length,)








if __name__ == "__main__":
    test_qwen2_fast_jit_sample2()
