import os

from jax.experimental.multihost_utils import process_allgather

from training import reward_correct, reward_format, get_state, training_step, repeat, slice_data

os.environ['JAX_TRACEBACK_FILTERING']='off'


import random
from functools import partial

import jax.tree_util
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from MLLM_JAX.utils import get_jax_mesh2, _form_global_array, collect_process_data
# from sample_state_left_padding import get_model, Sampler
import jax.numpy as jnp

max_prompt_length=400
num_pre_Q=16
MAX_LENGTH_SAMPLE=1024
MAX_LENGTH=MAX_LENGTH_SAMPLE+512 #-128
BATCH=8
grad_accum_steps = 8

model_path = 'Qwen/Qwen2.5-3B'
tokenizer = AutoTokenizer.from_pretrained(model_path)
system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""



    # input_ids_global=jax.tree_util.tree_map_with_path(sampler.global_collect_method,input_ids)
    #
    # local_ids=collect_process_data(input_ids_global)
    #
    #
    # oris=sampler.tokenizer.batch_decode(input_ids,
    #                                          skip_special_tokens=True, )
    #
    # local_answers = sampler.tokenizer.batch_decode(local_ids,
    #                                          skip_special_tokens=True, )
    #
    # for ori,loc in zip(oris,local_answers):
    #     if jax.process_index()==0:
    #         print(ori==loc)
    # while True:
    #     pass



def gen_answers_jax(prompts,sampler,params):
    prompt = []
    for x in prompts:
        prompt.append(tokenizer.apply_chat_template([
             {"role": "system", "content": system_prompt},
             {"role": "user", "content": x}
            # {"role": "system", "content": "You are a helpful assistant."},
            # {"role": "user", "content": "Who are you?"}
        ],

            tokenize=False, add_generation_prompt=True))




    inputs = sampler.tokenizer(prompt, return_tensors="jax", padding=True, padding_side="right")
    input_ids = inputs['input_ids']

    position_ids = inputs['attention_mask'].cumsum(-1) - 1
    position_ids = jnp.where(inputs['attention_mask'] == 0, 1, position_ids)

    # global_length = jnp.max(process_allgather(input_ids.shape[1]))
    global_length=512
    prefill_length = sampler.find_ceil(global_length)

    attention_mask = inputs['attention_mask']
    input_ids_pad = jnp.pad(input_ids, ((0, 0), (0, prefill_length - input_ids.shape[1])),
                            constant_values=sampler.tokenizer.eos_token_id)

    pad_attention = jnp.pad(attention_mask, ((0, 0), (0, prefill_length - input_ids.shape[1])))
    pad_position_ids = jnp.pad(position_ids, ((0, 0), (0, prefill_length - input_ids.shape[1])))

    completion_ids,local_sample_step=sampler.generate(input_ids_pad, pad_attention, pad_position_ids, prefill_length, max_length=MAX_LENGTH_SAMPLE,params=params)

    # answers=sampler.tokenizer.batch_decode(completion_ids[:,prefill_length:],
    #                                     skip_special_tokens=True,)

    answers = []
    for i, step in enumerate(local_sample_step):
        output = \
            sampler.tokenizer.batch_decode(completion_ids[i, prefill_length:prefill_length + step + 1].reshape(1, -1),
                                        skip_special_tokens=True,
                                        )

        answers.extend(output)

    print(len(prompt),len(answers))


    if jax.process_index()==0:
        print(answers[-2:])
        print('\n' * 2, flush=True)
    return prompt,answers


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



def get_advantages(rewards,groups):
    mean_grouped_rewards = rewards.reshape(-1, groups).mean(axis=1)
    std_grouped_rewards = rewards.reshape(-1, groups).std(axis=1)
    mean_grouped_rewards = jnp.repeat(mean_grouped_rewards, groups, axis=0)
    std_grouped_rewards = jnp.repeat(std_grouped_rewards, groups, axis=0)
    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

    return mean_grouped_rewards,std_grouped_rewards,advantages



def gen_samples(inputs,sampler,params):
    prompts = [x["Q"] for x in inputs]

    tip_text,answers = gen_answers_jax(prompts,sampler,params)

    if len(answers) == 0: return None, None, None, None
    rewards = []

    # for i, inp in enumerate(inputs):
    #     for a in answers[i*num_pre_Q:(i+1)*num_pre_Q]:
    #         rewards.append(reward_correct(inp, a) + reward_format(inp, a))

    for i, (inp,a) in enumerate(zip(inputs,answers)):
        try:
            rewards.append(reward_correct(inp, a) + reward_format(inp, a))
        except Exception as e:
            print(e,a)
            rewards.append(-10)

    print(rewards,np.mean(rewards))
    datas=batch_process(tip_text,answers,rewards,sampler.tokenizer)
    return datas



def main():
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    QAs = [{'Q': x, 'A': y.split('####')[-1].strip()} for x, y in zip(dataset['question'], dataset['answer'])]

    max_cache_length = MAX_LENGTH_SAMPLE
    mesh = get_jax_mesh2("-1,8,1")
    training_steps = 100
    state, sampler, train_state_sharding = get_state(mesh, training_steps)
    test_fn = jax.jit(training_step, donate_argnums=(0,), )

    for i in range(training_steps):
        inputs = random.sample(QAs, BATCH)
        # datas = gen_samples(repeat(inputs, num_pre_Q), sampler, state.params)
        repeated_inputs=repeat(inputs, num_pre_Q)
        prompts = [x["Q"] for x in repeated_inputs]
        tip_text, answers = gen_answers_jax(prompts, sampler, state.params)
        rewards = []
        for _, (inp, a) in enumerate(zip(repeated_inputs, answers)):
            try:
                rewards.append(reward_correct(inp, a) + reward_format(inp, a))
            except Exception as e:
                print(e, a)
                rewards.append(-10)

        print(rewards, np.mean(rewards))
        datas = batch_process(tip_text, answers, rewards, sampler.tokenizer)



        for j in range(grad_accum_steps):
            local_data = jax.tree_util.tree_map(lambda x: slice_data(x, grad_accum_steps, j), datas, )
            batch = jax.tree_util.tree_map_with_path(partial(_form_global_array, global_mesh=mesh), local_data)
            state, metrics = test_fn(state, batch)



        print(f"{i=} ")


if __name__=="__main__":
    jax.distributed.initialize()
    jax.config.update("jax_compilation_cache_dir", "gs://arm-central-2b/jax-cache")
    main()


        # print(metrics)










