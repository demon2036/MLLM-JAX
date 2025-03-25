from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaForCausalLM
import torch

mode_path = 'Qwen/Qwen2-0.5B-Instruct'

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)
# 加载模型
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()


prompt = "Who are you?"
# prompt = "你是谁? "

messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
]

input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# print(input_ids)

model_inputs = tokenizer([input_ids], return_tensors="pt").to('cuda')

from transformers import GenerationConfig
generation_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=False,
    # temperature=0.9,
            # num_return_sequences=num_pre_Q,
            pad_token_id=tokenizer.pad_token_id,
        )


generated_ids = model.generate(model_inputs.input_ids,generation_config=generation_config)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print('皇上：', prompt)
print('嬛嬛：',response)