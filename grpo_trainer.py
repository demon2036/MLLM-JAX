from transformers import AutoTokenizer



model_path = 'Qwen/Qwen2.5-3B'
tokenizer=AutoTokenizer.from_pretrained(model_path)
system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""



prompt=tokenizer.apply_chat_template([
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": '1+1=2 1+2=?'},
],
    tokenize=False, add_generation_prompt=False)

answer='3'
print(prompt+answer)


print(tokenizer.pad_token_id)

print(tokenizer.decode(tokenizer.pad_token_id))








