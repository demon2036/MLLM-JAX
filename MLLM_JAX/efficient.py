import functools
import jax
import math

from jax import numpy as jnp
from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention
import flax.linen as nn
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu





def attention_math(query_states,key_states,value_states,attn_mask):
    attn_weights = (query_states @ key_states.swapaxes(2, 3))
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask
    attn_weights = nn.softmax(attn_weights.astype(jnp.float32), axis=-1, ).astype(attn_weights.dtype)
    attn_output = attn_weights @ value_states
    return attn_output


def matmul_kernel(q_ref, k_ref, v_ref,mask_ref,
                m_scratch_ref,
                l_scratch_ref,
                o_ref, block_kv=128,block=128, ):

    i,j=pl.program_id(0),pl.program_id(1)

    @pl.when(j==0)
    def init():
        m_scratch_ref[...] = jnp.full_like(m_scratch_ref,-1e37)
        l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
        o_ref[...]=jnp.zeros_like(o_ref)


    def body(kv_index,carry):
        # slice_q = pl.ds(i * block, block)
        slice_k=pl.ds(kv_index*block_kv,block_kv)

        k=pl.load(k_ref,(slice_k,slice(None)))
        q=q_ref[...]
        # qk=q@k.T

        dims = ((1,), (1,)), ((), ())

        qk = jax.lax.dot_general(
            q, k, dims, preferred_element_type=jnp.float32,
        )

        m_prev = m_scratch_ref[...]
        l_prev = l_scratch_ref[...]


        mask=pl.load(mask_ref,(slice(None),slice_k))
        qk=qk+jnp.where(mask,0,-1e37)

        m_curr = qk.max(axis=-1)[:, None]  # Row max, shape [block_q, 1].
        m_next = jnp.maximum(m_prev, m_curr)  # Shape [block_q, 128].
        s_curr = jnp.exp(qk - m_next)

        l_curr=s_curr.sum(-1)[:,None]
        alpha = jnp.exp(m_prev - m_next)

        l_next=l_curr+alpha*l_prev
        m_scratch_ref[...]=m_next[...].astype(m_scratch_ref.dtype)
        l_scratch_ref[...]=l_next[...].astype(l_scratch_ref.dtype)

        v = pl.load(v_ref, (slice_k, slice(None)))
        o_curr=s_curr@v

        alpha_o = jnp.exp(m_prev - m_next)
        o_ref[...] = (alpha_o*o_ref[:]+o_curr).astype(o_ref.dtype)

    num_iters = (
            k_ref.shape[0 ] // block_kv
    )
    jax.lax.fori_loop(0,num_iters,body,None,)

    o_ref[...]=o_ref[...]/l_scratch_ref[...]



def matmul(q: jax.Array, k: jax.Array,v: jax.Array,mask,block_q:int=128):
    n,d=q.shape
    grids=(q.shape[0]//block_q,1)

    out_shape=[
        jax.ShapeDtypeStruct((q.shape[0], 1), q.dtype),
        jax.ShapeDtypeStruct((q.shape[0], d), q.dtype),
        jax.ShapeDtypeStruct((q.shape[0], v.shape[1]), q.dtype),
    ]


    _,_,output= pl.pallas_call(
        kernel=matmul_kernel,
        grid=grids,
        out_shape=out_shape,
        in_specs=[
                pl.BlockSpec(block_shape=(block_q,q.shape[1]) , index_map=lambda i,j: (i, 0),  ),
                pl.BlockSpec(block_shape=(k.shape[0] , k.shape[1]), index_map=lambda i,j: (0, 0), ),
                pl.BlockSpec(block_shape=(v.shape[0] , v.shape[1]), index_map=lambda i,j: (0, 0), ),
                pl.BlockSpec(block_shape=(block_q, v.shape[0]), index_map=lambda i, j: (i, 0), ),

            ],

        out_specs=[
            pl.BlockSpec(block_shape=(q.shape[0]//grids[0], 1), index_map=lambda i, j: (i, 0), ),
            pl.BlockSpec(block_shape=(q.shape[0] // grids[0], d), index_map=lambda i, j: (i, 0), ),
            pl.BlockSpec(block_shape=(q.shape[0] // grids[0], v.shape[1]), index_map=lambda i, j: (i, 0), ),
        ],
              interpret=True,

        compiler_params=pltpu.TPUCompilerParams(
            dimension_semantics=("parallel",  "arbitrary"),
        ),

    )(q,k,v,mask)

    return output



def flash_attention(q,k,v,mask):
    return jax.vmap(matmul,in_axes=(0,0,0,None))(q,k,v,mask[0])
