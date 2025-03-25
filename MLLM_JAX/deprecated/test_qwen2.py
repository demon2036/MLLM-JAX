import torch

import numpy as np
from tqdm import tqdm

def test_qwen_torch():
    # model_path = 'Qwen/Qwen2-7B-Instruct'
    model_path = '/autodl-tmp/qwen/Qwen2-7B-Instruct'
    model_path = './autodl-tmp/qwen/Qwen2-7B-Instruct'

    from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor, LlamaTokenizerFast, \
        AutoTokenizer, AutoModelForCausalLM,AutoConfig,Qwen2Tokenizer


    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,)
    from transformers.generation import configuration_utils

    # from transformers import BitsAndBytesConfig
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    # )

    # Load the base model with adapters on top
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,device_map='auto'
        # quantization_config=quantization_config,
    )


    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        # {"role": "user", "content": "Who are you?"},
        # {"role": "user", "content": "Give me a short introduction to large language model."},
        {"role": "user", "content": "Whose your daddy?"},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(prompt)


    device = 'cuda'
    # device = 'cpu'
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out=model.generate(**inputs,
                       do_sample=False,
                       # temperature=0.1,
                       # top_k=1000,
                       # top_p=1.0,
                       max_new_tokens=512)
    print(out)
    output=tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(output)
    #
    #
    # print('\n'*10)
    # while True:


if __name__=="__main__":
    test_qwen_torch()
