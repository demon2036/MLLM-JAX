import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, PreTrainedTokenizerFast


def process_func(example):
    MAX_LENGTH = 384    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n现在你要扮演皇帝身边的女人--甄嬛<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id] *max(0,MAX_LENGTH-len(instruction["input_ids"])+len(response["input_ids"]))
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1] *max(0,MAX_LENGTH-len(instruction["input_ids"])+len(response["input_ids"])) # 因为eos token咱们也是要关注的所以 补充为1
    labels = [tokenizer.pad_token_id] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id] *max(0,MAX_LENGTH-len(instruction["input_ids"])+len(response["input_ids"]))
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

if __name__ == "__main__":


    tokenizer = AutoTokenizer.from_pretrained('../MLLM-JAX/autodl-tmp/LLM-Research/Meta-Llama-3.1-8B-Instruct', use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token



    # messages = [
    #     {"role": "system", "content": "假设你是皇帝身边的女人--甄嬛。"},
    #     {"role": "user", "content": 'hi'}
    # ]
    #
    # input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # print(input_ids)
    #
    # while True:
    #     pass

    # 将JSON文件转换为CSV文件
    df = pd.read_json('huanhuan.json')
    ds = Dataset.from_pandas(df)
    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

    train_dataset = tokenized_id,
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)


    dataloader=torch.utils.data.DataLoader(train_dataset[0],collate_fn=data_collator,batch_size=2)
    for data in dataloader:
        print(data['input_ids'].shape,tokenizer.pad_token_id,data.keys())
        while True:
            pass
