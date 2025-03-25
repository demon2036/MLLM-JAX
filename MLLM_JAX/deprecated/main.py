from linecache import cache

import jax.random

from language.gemma.transformer import Transformer,TransformerConfig
import jax.numpy as jnp

from load_weight import get_pretrain_params

jax.config.update('jax_platform_name', 'cpu')
gemma_2b_config=TransformerConfig.gemma_2b_pali(1024)
model=Transformer(gemma_2b_config)

last_tokens=[[1,]]
last_tokens=jnp.array(last_tokens )
position_ids=jnp.array(last_tokens )

rng=jax.random.PRNGKey(1)
# variables=model.init(rng,last_tokens,position_ids,cache=None,attention_mask=None)
# params=variables['params']
# print(params['layer_0']['attn'].keys())


params=get_pretrain_params()
print(params.keys())

print(params['params'].keys())

# input_mask = sampler_state.token_buffer != self.vocab.pad_id()
input_mask = last_tokens != jnp.array([0])
print(input_mask)
out,cache=model.apply(params,last_tokens,position_ids,cache=None,attention_mask=input_mask)
print(out)



"""
dict_keys(['q_einsum', 'kv_einsum', 'attn_vec_einsum'])

dict_keys(['embedder', 'layer_0', 'layer_1', 'layer_2', 'layer_3', 'layer_4', 'layer_5', 'layer_6', 'layer_7', 'layer_8', 'layer_9', 'layer_10', 'layer_11', 'layer_12', 'layer_13', 'layer_14', 'layer_15', 'layer_16', 'layer_17', 'final_norm'])
"""