import jax
import jax.numpy as jnp
import flax.linen as nn




def quick_gelu(x):
    return x * jax.nn.sigmoid(1.702 * x)

ACT2FN = {
    'gelu': nn.gelu,
    'relu': nn.relu,
    'tanh': nn.tanh,
    'quick_gelu' :quick_gelu,
    # Add more activations as needed
}
