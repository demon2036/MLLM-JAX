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


def gen_answers_jax(prompts,sampler,params):
    prompt = []
    for x in prompts:
        prompt.append(tokenizer.apply_chat_template([
             {"role": "system", "content": system_prompt},
             {"role": "user", "content": x}
        ],
            tokenize=False, add_generation_prompt=True))

    inputs = sampler.tokenizer(prompt, return_tensors="jax", padding=True, padding_side="right")



    true_lengths = inputs['attention_mask'].sum(axis=1)[0]

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



    answers = []
    for i, step in enumerate(local_sample_step):
        output = \
            sampler.tokenizer.batch_decode(completion_ids[i, prefill_length:prefill_length + step + 1].reshape(1, -1),
                                        skip_special_tokens=True,
                                        )

        answers.extend(output)


    if jax.process_index()==0:
        print(answers[-2:])
        print('\n' * 2, flush=True)
    return prompt,answers







def batch_process(tip_texts,answers,rewards,tokenizer):



    total_texts=[tip_text+answer+tokenizer.eos_token for tip_text,answer in zip(tip_texts,answers)]


    tip_text=tip_texts[0]
    total_text=total_texts[0]

    tip_text_encoded=tokenizer([tip_text], return_tensors="jax", padding=True, padding_side="right")
    total_text_encoded=tokenizer([total_text], return_tensors="jax", padding=True, padding_side="right")

    diff=total_text_encoded[0,tip_text_encoded.shape[1]:]

    out= tokenizer.batch_decode(diff.reshape(1, -1),
                                        skip_special_tokens=True,
                                        )


    print(out)
    print('\n'*2)
    print(answers+tokenizer.eos_token)
    print(out==(answers+tokenizer.eos_token))




    while True:
        pass



















def get_advantages(rewards,groups):
    mean_grouped_rewards = rewards.reshape(-1, groups).mean(axis=1)
    std_grouped_rewards = rewards.reshape(-1, groups).std(axis=1)
    mean_grouped_rewards = jnp.repeat(mean_grouped_rewards, groups, axis=0)
    std_grouped_rewards = jnp.repeat(std_grouped_rewards, groups, axis=0)
    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

    return mean_grouped_rewards,std_grouped_rewards,advantages



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










