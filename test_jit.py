import copy
import os
from typing import Any

import tqdm

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

max_prompt_length=400
num_pre_Q=8
MAX_LENGTH_SAMPLE=512
MAX_LENGTH=MAX_LENGTH_SAMPLE+512 #-128
BATCH=1
grad_accum_steps = 1

model_path = 'Qwen/Qwen2.5-3B'
tokenizer = AutoTokenizer.from_pretrained(model_path)
system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""


def repeat(lst:list,repeats:int):
    return [x    for x in lst  for _ in range(repeats) ]



from math_verify import parse, verify, ExprExtractionConfig
def reward_correct(item, answer):
    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    nums = re.findall(pattern, answer) # 使用正则表达式在answer中查找所有数字
    if len(nums) == 0: return -1.0
    lastnum = nums[-1] # 用answer中最后一个数字和ground_truth做比较
    ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
    ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
    # print(item["A"],ans)

    return 1 if verify(ans, ground_truth) else -1
def reward_format(item, answer):
    # pattern = r"^<think>(?:(?!</?think>)[\s\S]*?)</think>\s*<answer>(?:(?!</?answer>)[\s\S]*?)</answer><\|im_end\|>$"
    pattern = r"^<think>.*?</think>.*?<answer>.*?</answer>$"
    # return 0.75 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) else -0.5
    return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) else -1



def gen_answers_jax(prompts,sampler,params):
    tip_text = []
    for x in prompts:
        tip_text.append(tokenizer.apply_chat_template([
             {"role": "system", "content": system_prompt},
             {"role": "user", "content": x}
            # {"role": "system", "content": "You are a helpful assistant."},
            # {"role": "user", "content": "Who are you?"}
        ],

            tokenize=False, add_generation_prompt=True))
    # tip_inputs = tokenizer(tip_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    # prompt_length = tip_inputs["input_ids"].shape[-1]

    # if prompt_length > max_prompt_length: return []
    answers=sampler.generate_prefill_auto_regressive(tip_text, max_length=MAX_LENGTH_SAMPLE,params=params)

    for ans in answers:
        print(ans,len(ans))
        print('\n'*2,flush=True)

    # print(answers[-1])

    # while True:
    #     pass
    return tip_text,answers


def process_func_padding(prompt,answer, tokenizer):
    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    # instruction_text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    # instruction_text += f"<|im_start|>user\n{prompt}<|im_end|>\n"

    instruction_text=prompt
    response_text = f"{answer}<|im_end|>\n"


    instruction = tokenizer(instruction_text, add_special_tokens=False, )  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(response_text, add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"]   + [tokenizer.pad_token_id]*(MAX_LENGTH-len(instruction["input_ids"] + response["input_ids"]))
    attention_mask = instruction["attention_mask"] + response["attention_mask"]  + [1] *(MAX_LENGTH-len(instruction["input_ids"] + response["input_ids"])) # 因为eos token咱们也是要关注的所以 补充为1
    # labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]

    labels = [tokenizer.pad_token_id] * len(instruction["input_ids"]) + response[
        "input_ids"]   + [tokenizer.pad_token_id]*(MAX_LENGTH-len(instruction["input_ids"] + response["input_ids"]))

    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def batch_process(tip_texts,answers,rewards,tokenizer):
    batch_results = [process_func_padding(prompt, answer, tokenizer)
                     for prompt, answer in zip(tip_texts, answers)]

    input_ids = jnp.array([item["input_ids"] for item in batch_results])
    attention_mask = jnp.array([item["attention_mask"] for item in batch_results])
    labels = jnp.array([item["labels"] for item in batch_results])
    rewards=jnp.array([item for item in rewards])
    return {'input_ids':input_ids,'attention_mask':attention_mask,'labels':labels,'rewards':rewards}




def gen_samples(inputs,sampler,params):
    prompts = [x["Q"] for x in inputs]

    tip_text,answers = gen_answers_jax(prompts,sampler,params)

    if len(answers) == 0: return None, None, None, None
    rewards = []

    # for i, inp in enumerate(inputs):
    #     for a in answers[i*num_pre_Q:(i+1)*num_pre_Q]:
    #         rewards.append(reward_correct(inp, a) + reward_format(inp, a))

    for i, (inp,a) in enumerate(zip(inputs,answers)):
        rewards.append(reward_correct(inp, a) + reward_format(inp, a))

    print(rewards,np.mean(rewards))
    datas=batch_process(tip_text,answers,rewards,sampler.tokenizer)
    return datas

class TrainState(train_state.TrainState):
    micro_step: int = 0
    micro_in_mini: int = 1
    grad_accum: ArrayTree | None = None
    ref_params:Any=None


def get_state(mesh,training_steps=100):
    model, params, tokenizer = get_model(mesh,model_path=model_path, )
    model_ref = get_model(mesh, model_path=model_path, only_model=True)

    beta=0.0

    train_module = TrainGRPOModule(model=model, pad_token_id=tokenizer.pad_token_id,ref_model=model_ref,num_pre_Q=num_pre_Q,beta=beta)
    def init_fn(params):

        learning_rate = optax.warmup_cosine_decay_schedule(
            init_value=1e-7,
            peak_value=1e-6,
            warmup_steps=int(training_steps*0.05),
            decay_steps=training_steps,
            end_value=1e-7,
        )
        tx = optax.adamw(learning_rate)
        # tx = optax.lion(learning_rate)
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




def training_step(state: TrainState, inputs: ArrayTree,) -> tuple[TrainState, ArrayTree]:

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
    return state, metrics


def slice_data(x,accumulate_steps,i):
    b,*_=x.shape
    assert b%accumulate_steps==0
    micro_batch_size=b//accumulate_steps
    data=x[i*micro_batch_size:(i+1)*micro_batch_size]
    # print(data.shape)
    return data


if __name__=="__main__":
    jax.distributed.initialize()
    jax.config.update("jax_compilation_cache_dir", "gs://arm-central-2b/jax-cache")


    dataset = load_dataset("openai/gsm8k", "main", split="train")
    QAs = [{'Q': x, 'A': y.split('####')[-1].strip()} for x, y in zip(dataset['question'], dataset['answer'])]


    max_cache_length = MAX_LENGTH_SAMPLE
    mesh = get_jax_mesh2("1,-1,1")

    training_steps=100

    state,sampler,train_state_sharding=get_state(mesh,training_steps)


    test_fn=jax.jit(training_step,donate_argnums=(0,),)

    for i in range(training_steps):
        inputs = random.sample(QAs, BATCH)
        datas=gen_samples(repeat(inputs,num_pre_Q),sampler,state.params)
        # batch = jax.tree_util.tree_map_with_path(partial(_form_global_array, global_mesh=mesh), datas)

        for j in range(grad_accum_steps):
            local_data=jax.tree_util.tree_map(lambda x:slice_data(x,grad_accum_steps,j)      ,datas,     )
            batch=jax.tree_util.tree_map_with_path(partial(_form_global_array, global_mesh=mesh), local_data)
            state, metrics=test_fn(state,batch)
            print(f"{j=} {metrics=} {test_fn._cache_size()=} ")


        # print(metrics)










