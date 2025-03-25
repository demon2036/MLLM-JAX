import random
import re

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from MLLM_JAX.utils import get_jax_mesh2
from sample_state_left_padding import get_model, Sampler

max_prompt_length=400
num_pre_Q=2



dataset = load_dataset("openai/gsm8k", "main", split="train")
QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} for x,y in zip(dataset['question'], dataset['answer'])]
# print(QAs)




model_path = 'Qwen/Qwen2-0.5B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_path)
system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""


def repeat(lst:list,repeats:int):
    return [x    for x in lst  for _ in range(repeats) ]




# inputs = random.sample(QAs, 1)
inputs = QAs[:1]
# gen_samples()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()


tip_text = []
for x in inputs:
    tip_text.append(tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": x['Q']}], tokenize=False, add_generation_prompt=True))

model_inputs = tokenizer(tip_text, return_tensors="pt").to('cuda')
generation_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=True,
            temperature=0.9,
            # num_return_sequences=num_pre_Q,
            pad_token_id=tokenizer.pad_token_id,
        )


generated_ids = model.generate(model_inputs.input_ids,generation_config=generation_config)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print('皇上：', prompt)
print('嬛嬛：',response)
