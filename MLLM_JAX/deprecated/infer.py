import einops
import flax
import flax.linen as nn
from PIL import Image
from transformers import PaliGemmaProcessor

from language.gemma.transformer import Transformer as Gemma, TransformerConfig, Cache
from load_weight import get_pretrain_pali_params
from vision.siglip.vit import Model as ViT
import jax
import jax.numpy as jnp





model_id = "google/paligemma-3b-mix-224"
image = Image.open('../kane.jpg')

processor = PaliGemmaProcessor.from_pretrained(model_id)
# Instruct the model to create a caption in Spanish
prompt="What is the color of a girl? "
model_inputs = processor(text=prompt, images=image, return_tensors="jax")
print(prompt)
# print(model_inputs.keys())
input_ids=model_inputs['input_ids']
attention_mask=model_inputs['attention_mask']
pixel_values=model_inputs['pixel_values']

# print(attention_mask.shape,pixel_values.min(),pixel_values.max())



pad_attention=jnp.pad(attention_mask,((0, 0),(0, 1024 - attention_mask.sum(axis=1)[0])))
# attention_mask=attention_mask[:,:256]
# pad_attention=jnp.pad(attention_mask,((0, 0),(0, 1024 - 256)))

# print(pad_attention.shape)




vit_model=ViT(num_classes=2048,variant="So400m/14",pool_type='none')
gemma_2b_config = TransformerConfig.gemma_2b_pali_mix(1024)

cache=gemma_2b_config.init_cache(1,jnp.float32)

transformer = Gemma(gemma_2b_config)
pali_gemma=PaliGemma(vit_model,transformer)
params=get_pretrain_pali_params()

pixel_values=einops.rearrange(pixel_values,'b c h w -> b h w c')
input_ids=input_ids[input_ids!=257152]

prefill_l=len(input_ids)
input_ids=input_ids[None,...]
# print(input_ids)
img_features=pali_gemma.apply(params,pixel_values,method=pali_gemma.embed_image)
text_features=pali_gemma.apply(params,input_ids,method=pali_gemma.embed_text)


text_features=text_features

# print(img_features.shape,text_features.shape)

prefix_embed=jnp.concatenate([img_features,text_features],axis=1)

position_ids=jnp.arange(0,prefix_embed.shape[1])[None,...]
# print(position_ids.shape)




logits, cache=pali_gemma.apply(params,prefix_embed,position_ids,cache,pad_attention,method=pali_gemma.decode_embeding)

position_ids=position_ids[:,-1]


"""
@chex.dataclass
class _SamplingState:
  # Decoding step.
  decoding_step: jnp.int32

  positions: jnp.ndarray  # [B, L]

  # Model state for conditioning the model on autoregressively.
  cache: dict[str, modules.LayerCache]



"""

res=[]
for i in range(4):
    select_ids = jnp.argmax(logits[:, -1], axis=1)
    res.append(select_ids)
    input_ids=select_ids
    position_ids += 1
    pad_attention = pad_attention.at[:, 260+i].set(1)
    next_embed = pali_gemma.apply(params, input_ids, method=pali_gemma.embed_text)[None,...]
    logits, cache = pali_gemma.apply(params, next_embed, position_ids, cache, pad_attention,
                                     method=pali_gemma.decode_embeding)

select_ids = jnp.argmax(logits[:, -1], axis=1)
res.append(select_ids)
for t in res:
    print(processor.tokenizer.decode(t),end='')


"""

logits, cache=pali_gemma.apply(params,prefix_embed,position_ids,cache,pad_attention,method=pali_gemma.decode_embeding)
logits=logits[:,-1]
print(logits.shape)
select_ids=jnp.argmax(logits,axis=1)
print(select_ids.shape)
print(processor.tokenizer.decode(select_ids))

pad_attention=pad_attention.at[:,263].set(1)



next_embed=pali_gemma.apply(params,select_ids,method=pali_gemma.embed_text)[None,...]
position_ids+=1
position_ids=position_ids[:,-1:]


print(next_embed.shape)

logits, cache=pali_gemma.apply(params,next_embed,position_ids,cache,pad_attention,method=pali_gemma.decode_embeding)
logits=logits[:,-1]
print(logits.shape)
select_ids=jnp.argmax(logits,axis=1)
print(select_ids.shape)
print(processor.tokenizer.decode(select_ids))
""""""
# print(img)
"""