import functools
from typing import Any

import numpy as np
import optax
import torch
import os
import pandas as pd
import tqdm
from chex import ArrayTree
from datasets import Dataset
from flax.training import train_state
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor, LlamaTokenizerFast, \
    AutoTokenizer, AutoModelForCausalLM, AutoConfig, Qwen2Tokenizer, DataCollatorForSeq2Seq
import jax
import jax.numpy as jnp

from MLLM_JAX.language.llama import convert_torch_to_flax_llama
from MLLM_JAX.language.qwen2.configuration_qwen2 import init_cache
from MLLM_JAX.language.qwen2.modular_qwen2 import Qwen2ForCausalLM
from MLLM_JAX.utils import get_jax_mesh2, match_partition_rules, get_partition_rules_llama
from jax.sharding import PartitionSpec as P

from MLLM_JAX.train_modules import TrainSFTModule

MAX_LENGTH = 384
epochs = 3





def test_qwen2(state,tokenizer,prompt="你好",model=None):
    dtype = jnp.bfloat16
    # dtype = jnp.float32
    max_cache_length=512

    model_path = 'Qwen/Qwen2-7B-Instruct'
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(config)
    cache = init_cache(config,1,max_cache_length=max_cache_length, dtype=dtype)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        # {"role": "user", "content": "Who are you?"},
        # {"role": "user", "content": "Give me a short introduction to large language model."},
        {"role": "user", "content": prompt},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
                                           # add_generation_prompt=False,continue_final_message=True)
    print(prompt)

    inputs = tokenizer(prompt, return_tensors="jax")


    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']


    pad_attention = jnp.pad(attention_mask, ((0, 0), (0, max_cache_length - attention_mask.sum(axis=1)[0])))

    input_ids = input_ids
    position_ids = jnp.arange(0, input_ids.shape[1])[None, ...]

    jit_infer = jax.jit(model.apply)



    logits, cache = jit_infer({'params': state.params['model']}, input_ids=input_ids, position_ids=position_ids,
                              attention_mask=pad_attention, cache=cache)

    position_ids = position_ids[:, -1][..., None]
    max_decode_length = 250
    res = []
    # exit_token_ids=tokenizer.encode(tokenizer.st)[0]
    exit_token_ids = 151645
    print(f'{tokenizer.eos_token=} , {exit_token_ids=}')



    for i in tqdm.tqdm(range(max_decode_length)):
        select_ids = jnp.argmax(logits[:, -1], axis=1)
        # print(select_ids)
        res.append(select_ids)

        if select_ids[0] == exit_token_ids:
            break

        input_ids = select_ids[..., None]
        position_ids += 1
        pad_attention = pad_attention.at[:, attention_mask.sum(axis=1)[0] + i].set(1)
        logits, cache = jit_infer({'params': state.params['model']}, input_ids=input_ids, position_ids=position_ids,
                                  attention_mask=pad_attention, cache=cache)

    ans = []
    for t in res:
        decoded_token = tokenizer.decode(t)
        print(decoded_token, end='', flush=True)
        ans.append(decoded_token)
    print('\n' * 10)

    print(np.array(res))
    output = \
        tokenizer.batch_decode(np.array(res).reshape(1, -1), skip_special_tokens=False,
                               clean_up_tokenization_spaces=False)[
            0]
    # output=tokenizer.decode(np.array(res), skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(output)














def process_func(example,tokenizer):
      # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction_text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    instruction_text += f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
    response_text = f"<|im_start|>assistant\n{example['output']}<|im_end|>\n"

    instruction = tokenizer(instruction_text, add_special_tokens=False,)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(response_text, add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"]  #+ [tokenizer.eos_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] #+ [1]  # 因为eos token咱们也是要关注的所以 补充为1
    # labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]

    labels = [tokenizer.pad_token_id] * len(instruction["input_ids"]) + response["input_ids"] #+ [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def process_func_padding(example, tokenizer):
    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction_text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    instruction_text += f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
    response_text = f"<|im_start|>assistant\n{example['output']}<|im_end|>\n"

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


def get_ds():

    df = pd.read_json('../deprecated_huan/huanhuan.json')
    ds = Dataset.from_pandas(df)

    model_path = 'Qwen/Qwen2-7B-Instruct'
    # model_path = './autodl-tmp/qwen/Qwen2-7B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, )
    tokenized_id = ds.map(functools.partial(process_func_padding,tokenizer=tokenizer), remove_columns=ds.column_names)

    # for data in tokenized_id:
    #     print(data)
    #     print(data['input_ids'])
    #     print(''.join(tokenizer.batch_decode(data['input_ids'])))
    #     print()
    dataloader = torch.utils.data.DataLoader(tokenized_id,batch_size=32,
                                             shuffle=True,drop_last=True,
                                             collate_fn=functools.partial(DataCollatorForSeq2Seq(tokenizer=tokenizer,label_pad_token_id=tokenizer.pad_token_id),return_tensors='np'))

    # for data in dataloader:
    #     print(data.keys())
    #
    #
    #     print(data['labels']!=tokenizer.pad_token_id)
    #     break
    return dataloader,tokenizer


def get_model(mesh,tokenizer):
    dtype = jnp.bfloat16
    model_path = 'Qwen/Qwen2-7B-Instruct'
    model_path = 'Qwen/Qwen2-7B'
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(config)
    # Load the base model with adapters on top
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
    )
    state_dict = model.state_dict()
    params = convert_torch_to_flax_llama(state_dict)
    # params = jax.tree_util.tree_map(lambda x: jnp.asarray(np.array(x), dtype=dtype), params)
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(np.array(x), dtype=dtype), params)
    params = {'model': params}
    model = Qwen2ForCausalLM(config)
    train_module = TrainSFTModule(model=model, pad_token_id=tokenizer.pad_token_id)


    training_steps=len(dataloader)*epochs

    def init_fn(params):
        learning_rate = optax.warmup_cosine_decay_schedule(
            init_value=1e-7,
            peak_value=5e-5,
            warmup_steps=int(training_steps*0.05),
            decay_steps=training_steps,
            end_value=1e-7,
        )

        # tx=optax.lion(learning_rate)
        tx = optax.lion(learning_rate)
        return TrainState.create(params=params,tx=tx,apply_fn=train_module.apply)


    state_shapes = jax.eval_shape(init_fn, params, )

    train_state_partition = match_partition_rules(get_partition_rules_llama(), state_shapes)
    train_state_sharding = jax.tree_util.tree_map(lambda x: jax.sharding.NamedSharding(mesh, x), train_state_partition)

    state = jax.jit(init_fn, donate_argnums=(0,),out_shardings=train_state_sharding)(params)

    return state,model



class TrainState(train_state.TrainState):
    micro_step: int = 0
    micro_in_mini: int = 1
    grad_accum: ArrayTree | None = None

    ema_params: ArrayTree|None = None
    ema_decay: float = 0.9998




if __name__=="__main__":

    dataloader,tokenizer=get_ds()
    mesh=get_jax_mesh2("1,-1,1")

    state,model=get_model(mesh,tokenizer)

    def test_fn(state, inputs):
        def loss_fn(params):
            metrics=state.apply_fn({'params': params, },inputs)
            return metrics['loss'],metrics

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(
            grads=grads,
        )
        return state, metrics


    test_fn=jax.jit(test_fn,donate_argnums=(0,))
    data_sharding=None

    data_sharding = jax.sharding.NamedSharding(mesh, P(['dp', 'fsdp']))

    def init_data(data):
        return data

    init_data=jax.jit(init_data,out_shardings=data_sharding)


    for i in range(epochs):
        pbar=tqdm.tqdm(dataloader)

        for data in pbar:
            data=data.data
            data=jax.tree_util.tree_map(init_data,data)
            state,loss = test_fn(state, data)
            # print(loss)
            pbar.set_description(f"{loss=}")


        test_qwen2(state,tokenizer,model=model)#prompt='Hi!'




    """"""