import os
import jax

os.environ['JAX_TRACEBACK_FILTERING']='off'


jax.distributed.initialize()
# jax.config.update("jax_compilation_cache_dir", "gs://arm-central-2b/jax-cache")

from jax.sharding import PartitionSpec as P
from jax import NamedSharding

import numpy
import wandb
from jax.experimental import multihost_utils
from jax.experimental.multihost_utils import process_allgather

from training import reward_correct, reward_format, get_state, training_step, repeat, slice_data, get_advantages

import random
from functools import partial

import jax.tree_util
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from MLLM_JAX.utils import get_jax_mesh2, _form_global_array, collect_process_data, match_partition_rules, \
    get_partition_rules_llama
# from sample_state_left_padding import get_model, Sampler
import jax.numpy as jnp


max_prompt_length=400
num_pre_Q=16
MAX_LENGTH_SAMPLE=2048
MAX_LENGTH=MAX_LENGTH_SAMPLE+512 #-128
BATCH=4
grad_accum_steps = 4

model_path = 'Qwen/Qwen2.5-7B'
tokenizer = AutoTokenizer.from_pretrained(model_path)
system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""


def apply_r1_template(question: str):
    return (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: "
        + question
        + "\nAssistant: <think>"
    )



def gen_answers_jax(prompts,sampler,params):
    prompt = []
    for x in prompts:
        # prompt.append(tokenizer.apply_chat_template([
        #      {"role": "system", "content": system_prompt},
        #      {"role": "user", "content": x}
        # ],
        #     tokenize=False, add_generation_prompt=True))

        prompt.append(apply_r1_template(x))

    inputs = sampler.tokenizer(prompt, return_tensors="jax", padding=True, padding_side="right")
    input_ids = inputs['input_ids']
    position_ids = inputs['attention_mask'].cumsum(-1) - 1
    position_ids = jnp.where(inputs['attention_mask'] == 0, 1, position_ids)

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

    print(answers[-2:])
    print('\n' * 2, flush=True)
    return prompt,answers



def batch_process(tip_texts,answers,rewards,tokenizer,max_length):
    total_texts=[tip_text+answer+tokenizer.eos_token for tip_text,answer in zip(tip_texts,answers)]
    tip_text_inputs=tokenizer(tip_texts, return_tensors="np", padding=True, padding_side="right")
    total_text_inputs=tokenizer(total_texts, return_tensors="np", padding=True, padding_side="right")

    true_lengths_prompts = tip_text_inputs['attention_mask'].sum(axis=1)
    attention_mask=total_text_inputs['attention_mask']
    labels=[]
    for true_length,mask in zip(true_lengths_prompts,attention_mask):
        temp=numpy.copy(mask)
        temp[:true_length]=0
        labels.append(temp)

    labels=np.array(labels,dtype=np.int32)
    input_ids=total_text_inputs['input_ids']

    input_ids_pad=np.full((input_ids.shape[0],MAX_LENGTH),fill_value=tokenizer.eos_token_id)
    input_ids_pad[:,:input_ids.shape[1]]=input_ids

    pad_attention=np.full((attention_mask.shape[0],MAX_LENGTH),fill_value=0)
    attention_mask[:,:attention_mask.shape[1]]=attention_mask

    pad_labels=np.full((labels.shape[0],MAX_LENGTH),fill_value=0)
    pad_labels[:,:labels.shape[1]]=labels

    # input_ids_pad = jnp.pad(input_ids, ((0, 0), (0, max_length - input_ids.shape[1])),
    #                         constant_values=tokenizer.eos_token_id)
    #
    # pad_attention = jnp.pad(attention_mask, ((0, 0), (0, max_length - input_ids.shape[1])))
    # pad_labels = jnp.pad(labels, ((0, 0), (0, max_length - input_ids.shape[1])))


    return {
        "input_ids": input_ids_pad,
        "attention_mask": pad_attention,
        "labels": pad_labels,
        'rewards': rewards
    }

    # rewards=jnp.array([item for item in rewards])

# rewards = []
# for _, (inp, a) in enumerate(zip(repeated_inputs, answers)):
#     try:
#         rewards.append(reward_correct(inp, a) + reward_format(inp, a))
#     except Exception as e:
#         print(e, a)
#         rewards.append(-10)


def mean(x):
    return x.mean()

def init_fn(x):
    return x


def main():
    reward_funcs=[reward_correct,reward_format]

    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset = dataset.shard(num_shards=jax.process_count(), index=jax.process_index())

    print(jnp.min(process_allgather(np.array(len(dataset)))))
    QAs = [{'Q': x, 'A': y.split('####')[-1].strip()} for x, y in zip(dataset['question'], dataset['answer'])]


    mesh_dp = get_jax_mesh2("-1,1,1")
    mesh_fsdp = get_jax_mesh2("1,-1,1")




    training_steps = 100
    state, sampler, train_state_sharding = get_state(mesh_fsdp, training_steps,model_path=model_path,
                                                     grad_accum_steps=grad_accum_steps,num_pre_q=num_pre_Q,max_lengths=MAX_LENGTH)
    test_fn = jax.jit(training_step, donate_argnums=(0,), )

    get_advantages_jit=jax.jit(get_advantages,static_argnums=(1,))


    mean_jit=jax.jit(mean,in_shardings=NamedSharding(mesh_dp,P(['dp','fsdp'])))
    params_shapes = jax.eval_shape(init_fn, state.params, )
    params_partition = match_partition_rules(get_partition_rules_llama(), params_shapes)
    params_sharding_dp = jax.tree_util.tree_map(lambda x: jax.sharding.NamedSharding(mesh_dp, x), params_partition)
    params_sharding_fsdp = jax.tree_util.tree_map(lambda x: jax.sharding.NamedSharding(mesh_fsdp, x), params_partition)
    params_to_dp = jax.jit(init_fn,out_shardings=params_sharding_dp)
    params_to_fsdp = jax.jit(init_fn,out_shardings=params_sharding_fsdp)




    if jax.process_index() == 0:
        # wandb.init(name=configs['name'], project=configs['project'], config=configs)
        wandb.init(name='test', project='grop-gsm8k',)


    for step in range(training_steps):
        inputs = random.sample(QAs, BATCH)
        # datas = gen_samples(repeat(inputs, num_pre_Q), sampler, state.params)
        repeated_inputs=repeat(inputs, num_pre_Q)
        prompts = [x["Q"] for x in repeated_inputs]


        tip_text, answers = gen_answers_jax(prompts, sampler, params_to_dp(state.params))

        print(f"{step=} syn for generate start")
        multihost_utils.sync_global_devices('syn for metric')
        print(f"{step=} syn for generate end")


        rewards_per_func=np.zeros( ( len(reward_funcs), len(answers),  ))
        for i, reward_func in enumerate(reward_funcs):
            for j, (inp, a) in enumerate(zip(repeated_inputs, answers)):
                try:
                    rewards_per_func[i,j]=reward_func(inp,a)
                except Exception as e:
                    print(e)
                    rewards_per_func[i, j] = -10

        rewards=rewards_per_func.sum(axis=0)

        print(rewards, np.mean(rewards))
        print(f"{step=} syn for batch_process start")
        datas = batch_process(tip_text, answers, rewards, sampler.tokenizer,max_length=MAX_LENGTH)
        multihost_utils.sync_global_devices('syn for batch_process')
        print(f"{step=} syn for batch_process end")
        # advantages = get_advantages_jit(datas['rewards'], num_pre_Q)
        # datas['advantages'] = advantages

        rewards_per_func=jnp.array(rewards_per_func)

        print(f"{step=} syn for metric start")
        multihost_utils.sync_global_devices('syn for metric')
        print(f"{step=} syn for metric end")

        metrics=dict()
        for i, reward_func in enumerate(reward_funcs):
            reward_funcs_name=reward_func.__name__
            reward_datas_local=rewards_per_func[i]
            reward_datas_mean= mean_jit(jax.tree_util.tree_map_with_path(partial(_form_global_array, global_mesh=mesh_dp), reward_datas_local))
            metrics[f"{reward_funcs_name}"]=reward_datas_mean

        print(f"{step=} syn for data start")
        multihost_utils.sync_global_devices('syn for data')
        datas = jax.tree_util.tree_map_with_path(partial(_form_global_array, global_mesh=mesh_dp), datas)
        # metrics['advantages']=datas['advantages'].mean()
        print(f"{step=} syn for data end")

        for j in range(grad_accum_steps):
            local_data = jax.tree_util.tree_map(lambda x: slice_data(x, grad_accum_steps, j), datas, )
            # batch = jax.tree_util.tree_map_with_path(partial(_form_global_array, global_mesh=mesh), local_data)
            state, _ = test_fn(state, local_data)


        if jax.process_index()==0:
            wandb.log(metrics,step)


if __name__=="__main__":

    main()


        # print(metrics)










