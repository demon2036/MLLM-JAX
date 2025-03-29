import flax
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np


conv=flax.linen.Conv(features=1,kernel_size=(3,),kernel_init=flax.linen.initializers.ones_init(),use_bias=True,padding='SAME',bias_init=flax.linen.initializers.zeros_init())
key=jax.random.PRNGKey(1)

x=np.arange(0,10).reshape(1,-1,1)
params=conv.init(key,x)
temp1=conv.apply(params,conv.apply(params,x))
# print(temp1)

x=np.arange(0,15).reshape(1,-1,1)
x[:,10:,:]=0

temp2=conv.apply(params,conv.apply(params,x).at[:,10:,:].set(0))
print(temp1-temp2[:,:10])