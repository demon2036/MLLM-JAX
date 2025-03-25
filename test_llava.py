import jax.tree_util
import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor, BitsAndBytesConfig,LlavaNextProcessor,LlavaConfig

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16,
                                                      device_map="auto",
                                                      quantization_config=BitsAndBytesConfig(
                                                          load_in_4bit=True,
                                                          bnb_4bit_use_double_quant=True,
                                                          bnb_4bit_quant_type="nf4",
                                                          bnb_4bit_compute_dtype=torch.bfloat16,
                                                      ),

                                                      # device_map="cpu",

                                                      )

# def pp(p,x):
#     print(p[0].key)
# keys=model.state_dict()
# jax.tree_util.tree_map_with_path(pp,keys)

config=LlavaConfig.from_pretrained("llava-hf/llava-1.5-7b-hf")
print(config)

# print(keys)


# while True:
#     pass


processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# Get two different images
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image_stop = Image.open(requests.get(url, stream=True).raw)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image_cats = Image.open(requests.get(url, stream=True).raw)

# Prepare a batch of two prompts
conversation_1 = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
    {
        "role": "user",
        "content": [
            # {"type": "image"},
            # {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]

conversation_2 = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]

prompt_1 = processor.apply_chat_template(conversation_1, add_generation_prompt=True)
prompt_2 = processor.apply_chat_template(conversation_2, add_generation_prompt=True)
prompts = [prompt_1, prompt_2]

# We can simply feed images in the order they have to be used in the text prompt
inputs = processor(images=[image_stop, image_cats ], text=prompts, padding=True, return_tensors="pt").to(model.device, torch.float16)


print(inputs['pixel_values'].shape)

# Generate
# generate_ids = model.generate(**inputs, max_new_tokens=30)
# output=processor.batch_decode(generate_ids, skip_special_tokens=True)

# print(output)

print(processor.batch_decode(inputs['input_ids'], skip_special_tokens=True))