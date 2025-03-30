import random
from functools import partial
from typing import Any

import random
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.experimental.shard_map import shard_map
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from MLLM_JAX.language.llama.llama import convert_torch_to_flax_llama, LlamaJaxConfig
from MLLM_JAX.language.qwen2.configuration_qwen2 import init_cache, pad_cache, pad_cache_right
from MLLM_JAX.language.qwen2.modular_qwen2 import Qwen2ForCausalLM
from MLLM_JAX.utils import match_partition_rules, get_partition_rules_llama, get_jax_mesh2, _form_global_array, \
    collect_process_data
from sanple_utils import _greedy_sampling, _temperature_sampling, _nucleus_sampling,  \
    _top_k_sampling_batched
from jax.sharding import PartitionSpec as P
from jax.experimental.multihost_utils import process_allgather

content = """1+1=2 1+2=?
"""




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


def get_model(mesh,model_path = 'Qwen/Qwen2.5-14B', only_model=False):
    # model_path='Qwen/Qwen2.5-7B'
    # model_path = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
    # model_path = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B'
    # model_path = 'Qwen/Qwen2-0.5B-Instruct'
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if jax.process_index()==0:
        print(config)
    # Load the base model with adapters on top

    jax_config = LlamaJaxConfig(mesh=mesh)
    model = Qwen2ForCausalLM(config, jax_config)

    if only_model:
        return model


    params = get_params(model_path)


    def init_fn(params):
        return params

    state_shapes = jax.eval_shape(init_fn, params, )

    train_state_partition = match_partition_rules(get_partition_rules_llama(), state_shapes)
    train_state_sharding = jax.tree_util.tree_map(lambda x: jax.sharding.NamedSharding(mesh, x), train_state_partition)

    params = jax.tree_util.tree_map(lambda x, d: jnp.asarray(x, dtype=jnp.bfloat16, device=d), params, train_state_sharding)

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


    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who are you?"},
    ] for _ in range(12)

]
# messages.append(
#     [
#         {"role": "system", "content": "You are a helpful assistant."},
#         # {"role": "user", "content": "Who are you?"},
#         {"role": "user", "content": " 1+1=?"},
#     ])


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


def create_sample_state(input_ids_pad, position_ids, cache, pad_attention, true_length, decoding_step=0,key=None):


    if key is None:
        key=random.randint(0, 2036)
        key=2036
        key=jax.random.PRNGKey(key)
    print(f'{key=}')

    sample_state = SampleState(decoding_step=decoding_step, num_input_tokens=true_length, token_buffer=input_ids_pad,
                               positions=position_ids, cache=cache, attention_mask=pad_attention,
                               next_token_buffer=jnp.zeros((pad_attention.shape[0])),
                               key=key,
                               dones=jnp.zeros((pad_attention.shape[0]), dtype=jnp.bool),
                               sample_steps=jnp.zeros((pad_attention.shape[0]), dtype=jnp.int32)
                               )
    return sample_state


class Sampler:
    def __init__(self, model, tokenizer,mesh=None,*args,**kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.dtype = jnp.float32

        self.mesh=mesh

        # self.sample_fn = _greedy_sampling
        self.key=jax.random.PRNGKey(2036)

        data_sharding = jax.sharding.NamedSharding(mesh, P(['dp', 'fsdp']))

        # data_sharding.mesh.shape[0]

        def warp_sample_fn(rng,logits):
            rngs=jax.random.split(rng,jax.device_count())


            def sample_inner(rng,logits):
                return _top_k_sampling_batched(rng[0],logits)

            sample_fn=shard_map(sample_inner,mesh=mesh,in_specs=(P(['dp', 'fsdp']),P(['dp', 'fsdp'],'tp'))
                                ,out_specs=P(['dp', 'fsdp']),check_rep=False)

            return sample_fn(rngs,logits)

        self.sample_fn=jax.jit(warp_sample_fn)

        # self.sample_fn=shard_map(_top_k_sampling_batched,mesh=mesh,in_specs=(None,P(['dp', 'fsdp'],'tp'))
        #                     ,out_specs=P(['dp', 'fsdp']),check_rep=False)

        # self.sample_fn=jax.jit(_top_k_sampling_batched)

        # self.sample_fn=jax.jit(_top_k_sampling_batched)

        self.jit_infer_prefill = jax.jit(self.model.apply)
        self.jit_infer_step = jax.jit(self.infer,donate_argnums=(0,))
        self.prefill_bucket = [
             128, 256,512, 1024, 2048, 4096, 8192
        ]

        def init_data(data):
            return data

        self.jit_init_data = jax.jit(init_data, out_shardings=data_sharding,)
        self.global_collect_method=partial(_form_global_array, global_mesh=self.mesh)



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


        next_token_predict=jnp.where(sample_state.dones,self.tokenizer.eos_token_id,next_token_predict)


        slice_tokens=jax.lax.dynamic_slice(sample_state.token_buffer,(0,i-5),(sample_state.token_buffer.shape[0],5))

        dones = sample_state.dones | (next_token_predict == self.tokenizer.eos_token_id) | (jnp.sum(slice_tokens==sample_state.token_buffer[:,i][:,None],axis=1)==5)

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



    def preprocess_prompt_prefill(self, prompt, prefill_length=128):


        inputs = self.tokenizer(prompt, return_tensors="jax", padding=True, padding_side="right")
        input_ids = inputs['input_ids']

        # position_ids = inputs['attention_mask'].cumsum(-1) - 1
        # position_ids = jnp.where(inputs['attention_mask'] == 0, 1, position_ids)


        global_length=jnp.max(process_allgather(input_ids.shape[1]))
        # global_length=512
        prefill_length = self.find_ceil(global_length)
        # prefill_length = input_ids.shape[1]

        attention_mask = inputs['attention_mask']
        input_ids_pad = jnp.pad(input_ids, ((0, 0), (0,prefill_length - input_ids.shape[1])),
                                constant_values=self.tokenizer.eos_token_id)

        pad_attention = jnp.pad(attention_mask, ((0, 0), (0,prefill_length - input_ids.shape[1])))
        # pad_position_ids = jnp.pad(position_ids, ((0, 0), (0,prefill_length - input_ids.shape[1])))
        pad_position_ids=jnp.arange(0,prefill_length)[None,...]

        return input_ids_pad, pad_attention, pad_position_ids, prefill_length

    def prepare_from_prefill_to_decode(self, cache, input_ids_pad, pad_attention, position_ids, max_length=8192):

        b, prefill_length = input_ids_pad.shape
        cache,input_ids_pad,pad_attention,position_ids=jax.tree_util.tree_map(collect_process_data,(cache,input_ids_pad,pad_attention,position_ids))

        position_ids = jnp.max(position_ids*pad_attention, axis=1).reshape((-1, 1)) + 1

        cache = pad_cache_right(cache, prefill_length, max_length, )

        input_ids_pad = jnp.pad(input_ids_pad, ((0, 0), (0, max_length)),
                                constant_values=self.tokenizer.eos_token_id)

        pad_attention = jnp.pad(pad_attention, ((0, 0), (0, max_length)),
                                constant_values=0)

        pad_attention = pad_attention.at[:, prefill_length].set(1)



        return jax.tree_util.tree_map_with_path(self.global_collect_method,(cache, input_ids_pad, pad_attention, position_ids))

        # return cache, input_ids_pad, pad_attention, position_ids


    def find_ceil(self, input):

        for num in self.prefill_bucket:  # 确保列表是有序的
            if num >= input:
                return num
        return None  # 如果 input 大于所有数字，返回 None



    def generate(self,input_ids_pad, pad_attention, position_ids, prefill_length, max_length=8192,params=None):
        # input_ids_pad, pad_attention, position_ids, prefill_length = self.preprocess_prompt_prefill(prompt,
        #                                                                                             prefill_length)

        if jax.process_index() == 0:
            print(f'{prefill_length=}')
        cache = init_cache(self.model.config, input_ids_pad.shape[0], max_cache_length=prefill_length, dtype=self.dtype,
                           shard_method=self.global_collect_method)

        input_ids_pad, pad_attention, position_ids = jax.tree_util.tree_map_with_path(self.global_collect_method,
                                                                                      (input_ids_pad, pad_attention,
                                                                                       position_ids))

        logits, cache = self.jit_infer_prefill({'params': params}, input_ids=input_ids_pad,
                                               position_ids=position_ids,
                                               attention_mask=pad_attention, cache=cache)
        cache, input_ids_pad, pad_attention, position_ids = self.prepare_from_prefill_to_decode(cache, input_ids_pad,
                                                                                                pad_attention,
                                                                                                position_ids,
                                                                                                max_length=max_length)

        next_token_logits = jnp.take_along_axis(logits, position_ids[..., None] - 1, axis=1)[:, -1]
        # next_token_predict = jnp.argmax(, axis=-1)[:,0]
        next_token_predict = self.sample_fn(self.key, next_token_logits)

        # next_token_predict = jnp.argmax(logits[:, position_ids-1], axis=1)
        input_ids_pad = input_ids_pad.at[:, prefill_length].set(next_token_predict)
        sample_state = create_sample_state(input_ids_pad=input_ids_pad, position_ids=position_ids, cache=cache,
                                           pad_attention=pad_attention, true_length=prefill_length,
                                           decoding_step=prefill_length, key=self.key)

        for i in tqdm(range(max_length)):
            sample_state = self.jit_infer_step(sample_state, params)
            if jnp.all(sample_state.dones):
                break

        local_sample_step = collect_process_data(sample_state.sample_steps)
        local_token_buffer = collect_process_data(sample_state.token_buffer)

        self.key = sample_state.key
        return local_token_buffer,local_sample_step





    def generate_prefill_auto_regressive(self, prompt, prefill_length=20, max_length=8192,params=None):

        input_ids_pad, pad_attention, position_ids, prefill_length = self.preprocess_prompt_prefill(prompt,
                                                                                                    prefill_length)

        if jax.process_index()==0:
            print(f'{prefill_length=}')
        cache = init_cache(self.model.config, input_ids_pad.shape[0], max_cache_length=prefill_length, dtype=self.dtype,shard_method=self.global_collect_method)

        # input_ids_pad, pad_attention, position_ids,cache=self.jit_init_data((input_ids_pad, pad_attention, position_ids,cache))
        input_ids_pad, pad_attention, position_ids = jax.tree_util.tree_map_with_path(self.global_collect_method,
                                                                                  (input_ids_pad, pad_attention, position_ids))

        logits, cache = self.jit_infer_prefill({'params': params}, input_ids=input_ids_pad,
                                               position_ids=position_ids,
                                               attention_mask=pad_attention, cache=cache)
        cache, input_ids_pad, pad_attention, position_ids = self.prepare_from_prefill_to_decode(cache, input_ids_pad,
                                                                                                pad_attention,
                                                                                                position_ids,
                                                                                                max_length=max_length)


        next_token_logits=jnp.take_along_axis(logits,position_ids[...,None]-1,axis=1)[:,-1]
        # next_token_predict = jnp.argmax(, axis=-1)[:,0]
        next_token_predict=self.sample_fn(self.key,next_token_logits)

        # next_token_predict = jnp.argmax(logits[:, position_ids-1], axis=1)
        input_ids_pad = input_ids_pad.at[:, prefill_length].set(next_token_predict)
        sample_state = create_sample_state(input_ids_pad=input_ids_pad, position_ids=position_ids, cache=cache,
                                           pad_attention=pad_attention, true_length=prefill_length,
                                           decoding_step=prefill_length,key=self.key)

        for i in tqdm(range(max_length)):
            sample_state = self.jit_infer_step(sample_state, params)
            # if jnp.all(sample_state.dones):
            #     break



        local_sample_step=collect_process_data(sample_state.sample_steps)
        local_token_buffer=collect_process_data(sample_state.token_buffer)


        texts=[]
        for i,step in enumerate(local_sample_step):
            output = \
                self.tokenizer.batch_decode(local_token_buffer[i, prefill_length:prefill_length + step + 1].reshape(1, -1),
                                        skip_special_tokens=False,
                                        )

            texts.extend(output)

        # texts2= self.tokenizer.batch_decode(local_token_buffer[:,prefill_length:],
        #                                 skip_special_tokens=True,
        #                                 )

        self.key=sample_state.key
        return texts


def test_qwen2_fast_jit_sample2():
    max_cache_length = 32
    mesh = get_jax_mesh2("1,1,-1")
    model, params, tokenizer = get_model(mesh, )
    exit_token_ids = tokenizer.eos_token_id
    print(f'{tokenizer.eos_token=} ,{tokenizer.eos_token_id=}, {exit_token_ids=}')
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)

    sampler = Sampler(model, tokenizer,mesh=mesh,)
    print('hi hi')
    sampler.generate_prefill_auto_regressive(prompt, max_length=max_cache_length,params=params)





if __name__ == "__main__":
    # test_qwen_torch()

    test_qwen2_fast_jit_sample2()
    # test_qwen2_fast_jit_sample2()
