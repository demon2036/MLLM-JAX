import asyncio
import functools
import random
from typing import Any

import random
from typing import Any

import chex
import flax.linen
import jax
import jax.numpy as jnp
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from MLLM_JAX.language.llama.llama import convert_torch_to_flax_llama, LlamaJaxConfig
from MLLM_JAX.language.qwen2.configuration_qwen2 import Qwen2Config, init_cache, pad_cache
from MLLM_JAX.language.qwen2.modular_qwen2 import Qwen2ForCausalLM
from MLLM_JAX.language.qwen3.modular_qwen3 import Qwen3ForCausalLM
from MLLM_JAX.sample.sanple_utils import _temperature_sampling
from MLLM_JAX.utils import match_partition_rules, get_partition_rules_llama, get_jax_mesh2
import os
from jax.sharding import PartitionSpec as P

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
# content="""已经知道有model，请你用JAX写一个top-p默认=0.95的采样"""

# content="""这里的问题在于无法jit,如何修正
#     def apply_top_p(probs, top_p):
#         # 按概率降序排列
#         sorted_probs = jnp.sort(probs)[::-1]
#         cumulative_probs = jnp.cumsum(sorted_probs)
#         # 找到累积概率超过top_p的最小阈值
#         cutoff_mask = cumulative_probs < top_p
#         # 保留至少一个token（避免全mask）
#         cutoff = jnp.max(jnp.where(cutoff_mask, jnp.arange(len(probs)), -1)) + 1
#         cutoff = jnp.maximum(cutoff, 1)  # 保证至少选1个
#         # 获取保留的probs对应的原始索引
#         sorted_indices = jnp.argsort(probs)[::-1][:cutoff]
#         kept_probs = probs[sorted_indices]
#         # 重新归一化概率
#         kept_probs = kept_probs / jnp.sum(kept_probs)
#         return sorted_indices, kept_probs
#
#     def sample_fn(logits_batch, top_p=0.95):
#     probs = jax.nn.softmax(logits_batch, axis=-1)
#     sorted_indices, kept_probs = apply_top_p(probs, top_p)
#     # 从保留的token中采样
#     return jax.random.choice(
#         key2, sorted_indices, p=kept_probs
#     )"""

# content="""写出一个可以jax jit的top p 采样,注意要定长，static，这是关键，认真思考"""

# content="""写出一个可以jax jit的top p 采样,注意要定长，static，这是关键，认真思考.
#     def static_top_p_sampling(logits, key, top_p=0.95):
#         # 确保所有操作保持静态形状
#         sorted_indices = jnp.argsort(-logits)  # 降序排列
#         sorted_logits = logits[sorted_indices]
#
#         # 计算排序后的概率分布
#         sorted_probs = jax.nn.softmax(sorted_logits)
#
#         # 计算累积概率（使用双精度提升数值稳定性）
#         cum_probs = jnp.cumsum(sorted_probs)  # .astype(sorted_probs.dtype)
# """


# content="""who are you
# """


#
# content = """1+1=2 1+2=?
# """

# content="""python如何找到一个数字离他最近的2的n次方，向上取整，比如7,就是8,9就是16这样子"""

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


def get_model(mesh, max_cache_length=8192):
    # model_path = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
    # model_path = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'
    model_path = 'Qwen/Qwen3-32B'
    # model_path = 'Qwen/Qwen2.5-14B-Instruct'
    # model_path = 'Qwen/QwQ-32B'
    # model_path = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'


    snapshot_download(model_path,max_workers=32)

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(config)
    # Load the base model with adapters on top

    params = get_params(model_path)
    jax_config = LlamaJaxConfig(mesh=mesh)
    model = Qwen3ForCausalLM(config, jax_config)

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
    cache = init_cache(config, 1, max_cache_length=max_cache_length, dtype=dtype)

    return model, params, tokenizer, cache,


messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    # {"role": "system", "content": "You are a helpful and harmless assistant. You should think step-by-step."},
    {"role": "user", "content": "Who are you?"},
    # {"role": "user", "content": "Give me a short introduction to large language model."},
    # {"role": "user", "content": "1+1=2 1+2=?"},
    # {"role": "user", "content": "Mnist examples in jax?"},
    # {"role": "user", "content": content, },
    # {"role": "assistant", "content": "A large language model is a type of artificial intelligence (AI) model that"},
    # {"role": "assistant", "content": "A large language model is a type of"},
]

def _greedy_sampling(rng, logits, ):
  del rng
  return jnp.argmax(logits, axis=-1)

def _nucleus_sampling(rng,logits ,p: float=0.95, t: float = 0.6, ):
  logits = logits / t
  neg_inf = np.array(-1.0e7)  # Effective negative infinity.
  logits_sorted = jnp.sort(logits, axis=-1, descending=True)
  sorted_cum_probs = jnp.cumsum(
      jax.nn.softmax(logits_sorted, axis=-1), axis=-1)
  cutoff_index = jnp.sum(sorted_cum_probs < p, axis=-1, keepdims=True)
  cutoff_logit = jnp.take_along_axis(logits_sorted, cutoff_index, axis=-1)
  logits = jnp.where(logits < cutoff_logit,
                     jnp.full_like(logits, neg_inf), logits)
  return jax.random.categorical(rng, logits)


@chex.dataclass
class SampleState:
    decoding_step: jnp.int32
    num_input_tokens: jnp.ndarray
    token_buffer: jnp.ndarray
    positions: jnp.ndarray
    cache: Any
    attention_mask: jnp.ndarray
    next_token_buffer: jnp.ndarray
    key:jnp.ndarray
    # logits_buffer:


def create_sample_state(input_ids_pad, position_ids, cache, pad_attention, true_length, decoding_step=0):
    sample_state = SampleState(decoding_step=decoding_step, num_input_tokens=true_length, token_buffer=input_ids_pad,
                               positions=position_ids, cache=cache, attention_mask=pad_attention,
                               next_token_buffer=jnp.zeros((pad_attention.shape[0])),key=jax.random.PRNGKey(random.randint(0,2036))
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
        self.sample_fn=functools.partial(_temperature_sampling,t=0.6)
        self.prefill_bucket = [
            128, 256, 512, 1024, 2048, 4096,8192,16384,int(16384*1.5),16384*2-1024,16384*2
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

        prefill_length = self.find_ceil(true_length)
        # prefill_length=true_length
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


    async def generate_prefill_auto_regressive(self, prompt, prefill_length=20, max_length=8192, stream=False):

        print(self.jit_init_data,len(prompt))



        input_ids_pad, pad_attention, position_ids, true_length, prefill_length = self.preprocess_prompt_prefill(prompt,
                                                                                                                 prefill_length)


        cache = init_cache(self.model.config, input_ids_pad.shape[0], max_cache_length=prefill_length, dtype=dtype,
                           shard_method=self.jit_init_data
                           )

        print(f'{prefill_length=}')

        # max_length=max(8192,prefill_length*2)
        # cache = init_cache(self.model.config, 1, max_cache_length=prefill_length, dtype=dtype)

        # input_ids_pad, pad_attention, position_ids,cache=self.jit_init_data((input_ids_pad, pad_attention, position_ids,cache))

        logits, cache = self.jit_infer_prefill({'params': self.params}, input_ids=input_ids_pad,
                                               position_ids=position_ids,
                                               attention_mask=pad_attention, cache=cache,true_length=true_length-1)


        next_token_predict = jnp.argmax(logits, axis=1)
        print(max_length,true_length)
        cache, input_ids_pad, pad_attention, position_ids = self.prepare_from_prefill_to_decode(cache, input_ids_pad,
                                                                                                pad_attention,
                                                                                                true_length,
                                                                                                max_length=max_length)
        # next_token_predict = jnp.argmax(logits[:, true_length - 1], axis=1)
        # next_token_predict = jnp.argmax(logits[:, - 1], axis=1)
        input_ids_pad = input_ids_pad.at[:, true_length].set(next_token_predict)
        sample_state = create_sample_state(input_ids_pad=input_ids_pad, position_ids=position_ids, cache=cache,
                                           pad_attention=pad_attention, true_length=true_length,
                                           decoding_step=true_length)


        exit_token_ids = self.tokenizer.eos_token_id

        if stream:
            yield next_token_predict

        res = [next_token_predict]
        for i in tqdm(range(max_length - true_length)  ,):
            sample_state = self.jit_infer_step(sample_state, self.params)
            select_ids = sample_state.next_token_buffer
            res.append(select_ids)

            output = \
                    self.tokenizer.batch_decode(np.array(res).reshape(1, -1), skip_special_tokens=False,
                                                clean_up_tokenization_spaces=False)[
                        0]
            print(output)

            if select_ids[0] == exit_token_ids:
                output = \
                    self.tokenizer.batch_decode(np.array(res).reshape(1, -1), skip_special_tokens=False,
                                                clean_up_tokenization_spaces=False)[
                        0]
                print(output)
                break
            if stream:
                yield select_ids

        # return None


async def test_qwen2_fast_jit_sample2():
    max_cache_length = 1024
    mesh = get_jax_mesh2("1,1,-1")
    model, params, tokenizer, cache = get_model(mesh, max_cache_length=max_cache_length)
    exit_token_ids = tokenizer.eos_token_id
    print(f'{tokenizer.eos_token=} ,{tokenizer.eos_token_id=}, {exit_token_ids=}')

    # prompt = "Who are you?"
    # messages = [
    #     {"role": "user", "content": prompt}
    # ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True  # Switches between thinking and non-thinking modes. Default is True.
    )

    sampler = Sampler(model, params, tokenizer,mesh=mesh)
    print(prompt)
    async for _ in sampler.generate_prefill_auto_regressive(prompt,max_length=max_cache_length):
        pass





if __name__ == "__main__":
    # test_qwen_torch()

    # state2 = test_qwen2_fast_jit_sampler()


    asyncio.run(    test_qwen2_fast_jit_sample2())


