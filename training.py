import copy
import os
from typing import Any

import flax
import tqdm
from jax.experimental.multihost_utils import process_allgather
from latex2sympy2_extended import NormalizationConfig

os.environ['JAX_TRACEBACK_FILTERING']='off'


import random
import re
from functools import partial

import jax.tree_util
import numpy as np
import optax
import torch
from chex import ArrayTree
from datasets import load_dataset
from flax.training import train_state
from jax import NamedSharding
from jax._src.mesh import Mesh
from jax._src.partition_spec import PartitionSpec
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from MLLM_JAX.language.llama.llama import convert_torch_to_flax_llama
from MLLM_JAX.language.qwen2.modular_qwen2 import Qwen2ForCausalLM
from MLLM_JAX.train_modules import TrainSFTModule, TrainGRPOModule
from MLLM_JAX.utils import get_jax_mesh2, match_partition_rules, get_partition_rules_llama, _form_global_array
from sample_state_right_padding import get_model, Sampler
# from sample_state_left_padding import get_model, Sampler
import jax.numpy as jnp
from math_verify import parse, verify, ExprExtractionConfig, LatexExtractionConfig


def slice_data(x,accumulate_steps,i):
    b,*_=x.shape
    assert b%accumulate_steps==0
    micro_batch_size=b//accumulate_steps
    data=x[i*micro_batch_size:(i+1)*micro_batch_size]
    # print(data.shape)
    return data






class TrainState(train_state.TrainState):
    micro_step: int = 0
    micro_in_mini: int = 1
    grad_accum: ArrayTree | None = None
    ref_params:Any=None


def get_state(mesh,training_steps=100,grad_accum_steps=1,model_path='Qwen/Qwen2.5-3B',num_pre_q=16,max_lengths=None):
    model, params, tokenizer = get_model(mesh,model_path=model_path, )
    model_ref = get_model(mesh, model_path=model_path, only_model=True)

    beta=0.0

    train_module = flax.linen.remat(TrainGRPOModule,policy=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims)(model=model,
                                   pad_token_id=tokenizer.pad_token_id,
                                   ref_model=model_ref,
                                   num_pre_Q=num_pre_q,
                                   beta=beta,
                                   max_lengths=max_lengths,
                                   )



    def init_fn(params):

        learning_rate = optax.warmup_cosine_decay_schedule(
            init_value=1e-7,
            peak_value=1e-6,
            warmup_steps=int(training_steps*0.05),
            decay_steps=training_steps,
            end_value=1e-7,
        )
        # tx = optax.adamw(learning_rate)
        tx = optax.lion(learning_rate)
        # tx = optax.sgd(learning_rate)
        if grad_accum_steps > 1:
            print(f'{grad_accum_steps=}')
            grad_accum = jax.tree_map(jnp.zeros_like, params)


        return TrainState.create(apply_fn=train_module.apply,params=params,tx=tx,
                                 ref_params=copy.deepcopy(params) if beta==0 else None,
                                 micro_step=0,
                                 micro_in_mini=grad_accum_steps,
                                 grad_accum=grad_accum if grad_accum_steps > 1 else None,
                                 )


    state_shapes = jax.eval_shape(init_fn, params, )
    train_state_partition = match_partition_rules(get_partition_rules_llama(), state_shapes)
    train_state_sharding = jax.tree_util.tree_map(lambda x: jax.sharding.NamedSharding(mesh, x), train_state_partition)
    state = jax.jit(init_fn, donate_argnums=(0,),out_shardings=train_state_sharding)(params)

    sampler = Sampler(model, tokenizer,mesh=mesh,)


    return state,sampler,train_state_sharding




def training_step(state: TrainState, inputs: ArrayTree) -> tuple[TrainState, ArrayTree]:

    def loss_fn(params: ArrayTree) -> ArrayTree:
        metrics=state.apply_fn({'params': {'model':params,'ref_model':state.ref_params }, },inputs)
        metrics = jax.tree_map(jnp.mean, metrics)
        return metrics["loss"], metrics

    def update_fn(state: TrainState) -> TrainState:

        # Collect a global gradient from the accumulated gradients and apply actual
        # parameter update with resetting the accumulations to zero.
        grads = jax.tree_map(lambda g: g / state.micro_in_mini, state.grad_accum)
        state = state.apply_gradients(
            grads=grads,
            grad_accum=jax.tree_map(jnp.zeros_like, state.grad_accum),
            micro_step=state.micro_step % state.micro_in_mini,
        )
        return state


    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # Update parameters with the gradients. If the gradient accumulation is enabled,
    # then the parameters will be updated at the end of each mini-batch step. In every
    # micro steps, the gradients will be accumulated.
    if state.grad_accum is None:
        state = state.apply_gradients(grads=grads)
    else:
        state = state.replace(
            grad_accum=jax.tree_map(lambda ga, g: ga + g, state.grad_accum, grads),
            micro_step=state.micro_step + 1,
        )
        state = jax.lax.cond(
            state.micro_step == state.micro_in_mini, update_fn, lambda x: x, state
        )
    return state#, metrics




















def repeat(lst:list,repeats:int):
    return [x    for x in lst  for _ in range(repeats) ]



def reward_correct(item, answer):
    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    nums = re.findall(pattern, answer) # 使用正则表达式在answer中查找所有数字
    if len(nums) == 0: return 0.0
    lastnum = nums[-1] # 用answer中最后一个数字和ground_truth做比较
    ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
    ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
    # print(item["A"],ans)

    return 1 if verify(ans, ground_truth) else 0.0







def reward_format(item, answer):
    # pattern = r"^<think>(?:(?!</?think>)[\s\S]*?)</think>\s*<answer>(?:(?!</?answer>)[\s\S]*?)</answer><\|im_end\|>$"
    # pattern = r"^<think>.*?</think>.*?<answer>.*?</answer>$"
    pattern = r"^.*?</think>.*?<answer>.*?</answer>$"
    # return 0.75 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) else -0.5
    # return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) else -1
    return 1 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) else 0





def get_advantages(rewards,groups,advantage_estimator='john_grpo',alpha=0.1,mean_global=None,std_global=None):

    if advantage_estimator=='grpo':
        mean_grouped_rewards = rewards.reshape(-1, groups).mean(axis=1)
        std_grouped_rewards = rewards.reshape(-1, groups).std(axis=1)
        mean_grouped_rewards = jnp.repeat(mean_grouped_rewards, groups, axis=0)
        std_grouped_rewards = jnp.repeat(std_grouped_rewards, groups, axis=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
    elif advantage_estimator=='dr_grpo':
        mean_grouped_rewards = rewards.reshape(-1, groups).mean(axis=1)
        mean_grouped_rewards = jnp.repeat(mean_grouped_rewards, groups, axis=0)
        advantages = (rewards - mean_grouped_rewards)

    elif advantage_estimator == 'john_grpo':
        mean_grouped_rewards = rewards.reshape(-1, groups).mean(axis=1)
        std_grouped_rewards = rewards.reshape(-1, groups).std(axis=1)
        mean_grouped_rewards = jnp.repeat(mean_grouped_rewards, groups, axis=0)
        std_grouped_rewards = jnp.repeat(std_grouped_rewards, groups, axis=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4) +alpha*(rewards - mean_global) / (std_global + 1e-4)

    else:
        mean_grouped_rewards = rewards.reshape(-1, groups).mean(axis=1)
        mean_grouped_rewards = jnp.repeat(mean_grouped_rewards, groups, axis=0)
        advantages = (rewards - mean_grouped_rewards) +alpha*(rewards-mean_global)

    return advantages
    # return mean_grouped_rewards,std_grouped_rewards,advantages