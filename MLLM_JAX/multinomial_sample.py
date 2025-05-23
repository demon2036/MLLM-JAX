import gc
import random
import time
from typing import Any

import random
from typing import Any

import chex
import einops
import flax.linen
import jax
import jax.numpy as jnp
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from transformers import Gemma3ForConditionalGeneration as Gemma3ForConditionalGenerationTorch

from MLLM_JAX.language.llama.llama import convert_torch_to_flax_llama, LlamaJaxConfig
from MLLM_JAX.language.qwen2.configuration_qwen2 import Qwen2Config, init_cache, pad_cache
from MLLM_JAX.language.qwen2.modular_qwen2 import Qwen2ForCausalLM
from MLLM_JAX.mutinomial.gemma3.modeling_gemma3 import convert_torch_to_flax_gemma3, Gemma3ForConditionalGeneration, \
    get_partition_rules_gemma3
from MLLM_JAX.utils import match_partition_rules, get_partition_rules_llama, get_jax_mesh2, tree_path_to_string
import os
from jax.sharding import PartitionSpec as P

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


dtype = jnp.bfloat16


def get_params(model_path):
    model = Gemma3ForConditionalGenerationTorch.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
    )
    state_dict = model.state_dict()
    params = convert_torch_to_flax_gemma3(state_dict)
    # params = jax.tree_util.tree_map(lambda x: jnp.asarray(np.array(x), dtype=dtype), params)
    params = jax.tree_util.tree_map(lambda x: np.array(x), params)

    return params


def get_model(mesh, max_cache_length=8192):
    # model_path = "google/gemma-3-27b-it"
    model_path = "google/gemma-3-12b-it"
    # model_path = "google/gemma-3-4b-it"

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # Load the base model with adapters on top

    params = get_params(model_path)
    jax_config = LlamaJaxConfig(mesh=mesh)
    model = Gemma3ForConditionalGeneration(config,jax_config=jax_config )#jax_config

    def init_fn(params):
        return params

    state_shapes = jax.eval_shape(init_fn, params, )
    train_state_partition = match_partition_rules(get_partition_rules_gemma3(), state_shapes)
    train_state_sharding = jax.tree_util.tree_map(lambda x: jax.sharding.NamedSharding(mesh, x), train_state_partition)



    params = jax.tree_util.tree_map(lambda x, d: jnp.asarray(x, dtype=dtype, device=d), params, train_state_sharding)

    params = jax.jit(init_fn,
                     donate_argnums=(0,),
                     out_shardings=train_state_sharding)(params)
    processor = AutoProcessor.from_pretrained(model_path)

    return model, params, processor




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
    def __init__(self, model, params, processor, max_length=8192, prefill_length=128, cache=None,mesh=None):
        self.model = model
        self.processor = processor
        self.tokenizer=self.processor.tokenizer
        self.params = params
        # self.max_length=max_length
        self.prefill_length = prefill_length
        self.cache = cache
        self.jit_infer_prefill = jax.jit(self.model.apply,donate_argnames=('cache',))
        self.jit_infer_step = jax.jit(self.infer)


        self.sample_fn=_nucleus_sampling

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



    def preprocess_prompt_prefill(self, messages,):

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="jax"
        ).data

        input_ids = inputs['input_ids']
        if 'pixel_values' in inputs:
            pixel_values=einops.rearrange(inputs['pixel_values'],'b c h w -> b h w c')
        else:
            pixel_values=None

        attention_mask = inputs['attention_mask']


        true_length = int(attention_mask.sum(axis=1)[0])
        prefill_length = self.find_ceil(true_length)
        print(input_ids.shape)

        input_ids_pad = jnp.pad(input_ids, ((0, 0), (0, prefill_length - attention_mask.sum(axis=1)[0])),
                                constant_values=self.tokenizer.eos_token_id)

        pad_attention = jnp.pad(attention_mask, ((0, 0), (0, prefill_length - attention_mask.sum(axis=1)[0])))

        position_ids = jnp.arange(0, input_ids_pad.shape[1])[None, ...]
        pad_position_ids=position_ids
        # pad_position_ids = jnp.pad(position_ids, ((0, 0), (0, prefill_length - attention_mask.sum(axis=1)[0])))

        return input_ids_pad, pad_attention, pad_position_ids, pixel_values,true_length, prefill_length

    def prepare_from_prefill_to_decode(self, cache, input_ids_pad, pad_attention, true_length, max_length=8192):

        b, prefill_length = input_ids_pad.shape

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


    def generate_prefill_auto_regressive(self, messages,  max_length=8192, stream=False):

        input_ids_pad, pad_attention, position_ids,pixel_values, true_length, prefill_length = self.preprocess_prompt_prefill(messages,
                                                                                                                 )

        print(f'{prefill_length=}')
        if prefill_length>=8192:
            max_length=min(prefill_length*2,16384)



        cache = init_cache(self.model.config, input_ids_pad.shape[0], max_cache_length=prefill_length, dtype=dtype,shard_method=self.jit_init_data)

        print(max_length,true_length)
        logits, cache = self.jit_infer_prefill({'params': self.params}, input_ids=input_ids_pad,pixel_values=pixel_values,
                                               position_ids=position_ids,
                                               attention_mask=pad_attention, cache=cache,true_length=true_length-1)
        next_token_predict = jnp.argmax(logits, axis=1)
        # del logits

        print('begin prepare')
        cache = pad_cache(cache, prefill_length, max_length, true_length=true_length)
        print('after prepare')

        print(max_length,true_length)
        cache, input_ids_pad, pad_attention, position_ids = self.prepare_from_prefill_to_decode(cache, input_ids_pad,
                                                                                                pad_attention,
                                                                                                true_length,
                                                                                                max_length=max_length)



        input_ids_pad = input_ids_pad.at[:, true_length].set(next_token_predict)
        sample_state = create_sample_state(input_ids_pad=input_ids_pad, position_ids=position_ids, cache=cache,
                                           pad_attention=pad_attention, true_length=true_length,
                                           decoding_step=true_length)

        print('after create_sample_state')
        exit_token_ids = self.tokenizer.eos_token_id
        exit_token_ids=106

        if stream:
            yield next_token_predict

        res = [next_token_predict]
        print(max_length - true_length)
        for i in tqdm(range(max_length - true_length)):
            sample_state = self.jit_infer_step(sample_state, self.params)
            select_ids = sample_state.next_token_buffer
            res.append(select_ids)

            if stream:
                yield select_ids

            if select_ids[0] == exit_token_ids:
                output = \
                    self.tokenizer.batch_decode(np.array(res).reshape(1, -1), skip_special_tokens=False,
                                                clean_up_tokenization_spaces=False)[
                        0]
                print(output)

                break

        return None


def test_gemma3():
    max_cache_length = 8192
    mesh = get_jax_mesh2("1,1,-1")
    model, params, processor = get_model(mesh, max_cache_length=max_cache_length)
    # exit_token_ids = tokenizer.eos_token_id
    # print(f'{tokenizer.eos_token=} ,{tokenizer.eos_token_id=}, {exit_token_ids=}')
    # prompt = tokenizer.apply_chat_template(messages, tokenize=False,
    #                                        add_generation_prompt=True)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image","image": "kaf.png"},
                {"type": "image", "image": "kane.jpg"},
                {"type": "text", "text": "Is this two img same?"}
            ]
        }
    ]

    sampler = Sampler(model, params, processor,mesh=mesh)
    print('hi hi')
    for _ in sampler.generate_prefill_auto_regressive(messages,max_length=8192):
        pass






if __name__ == "__main__":
    # test_qwen_torch()
    # state2 = test_qwen2_fast_jit_sampler()
    test_gemma3()
