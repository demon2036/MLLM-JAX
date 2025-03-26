import requests
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, LlavaNextProcessor, \
    LlavaNextForConditionalGeneration, BitsAndBytesConfig

# Load the model in half-precision
model = AutoModelForImageTextToText.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16,
                                                    quantization_config=BitsAndBytesConfig(
                                                        load_in_4bit=True,
                                                        bnb_4bit_use_double_quant=True,
                                                        bnb_4bit_quant_type="nf4",
                                                        bnb_4bit_compute_dtype=torch.bfloat16,
                                                    ),

                                                    device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

# Get three different images
url = "kane.jpg"
# image_stop = Image.open(requests.get(url, stream=True).raw)
image_stop = Image.open(url)

url = "kane.jpg"
# image_cats = Image.open(requests.get(url, stream=True).raw)
image_cats = Image.open(url)
url = "kane.jpg"
image_snowman = Image.open(url)
# image_snowman = Image.open(requests.get(url, stream=True).raw)

# Prepare a batch of two prompts, where the first one is a multi-turn conversation and the second is not
conversation_1 = [
    {
        "role": "user",
        "content": [

            {"type": "text", "text": "What is shown in this image?"},
            {"type": "image"},
            ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "There is a red stop sign in the image."},
            ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What about this image? "},
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
# Each "<image>" token uses one image leaving the next for the subsequent "<image>" tokens
inputs = processor(images=[image_stop, image_cats, image_snowman], text=prompts, padding=True, return_tensors="pt").to(model.device)

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=30)
print(processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))

# print(inputs['pixel_values'].shape)
# print(inputs.keys())
# print(processor.batch_decode(inputs['input_ids'], skip_special_tokens=False, clean_up_tokenization_spaces=False))