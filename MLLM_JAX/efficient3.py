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


def matmul_kernel(q_ref, k_ref, v_ref,
                mask_ref,
                m_scratch_ref,
                l_scratch_ref,
                o_ref, kv_seq_len,):

    block_k_major = k_ref.shape[0]
    block_q = q_ref.shape[0]
    head_dim = q_ref.shape[-1]

    kv_seq_idx = pl.program_id(2)

    @pl.when(kv_seq_idx==0)
    def init():
        m_scratch_ref[...] = jnp.full_like(m_scratch_ref,-1e37)
        l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
        o_ref[...]=jnp.zeros_like(o_ref)

    k = k_ref[...]
    q = q_ref[...]
    # qk=q@k.T

    dims = ((1,), (1,)), ((), ())

    qk = jax.lax.dot_general(
        q, k, dims, preferred_element_type=jnp.float32,
    )

    m_prev = m_scratch_ref[...]
    l_prev = l_scratch_ref[...]

    mask = mask_ref[...]
    qk=qk+jnp.where(mask,0,-1e37)

    m_curr = qk.max(axis=-1)[:, None]  # Row max, shape [block_q, 1].
    m_next = jnp.maximum(m_prev, m_curr)  # Shape [block_q, 128].
    s_curr = jnp.exp(qk - m_next)

    l_curr=s_curr.sum(-1)[:,None]
    alpha = jnp.exp(m_prev - m_next)

    l_next=l_curr+alpha*l_prev
    m_scratch_ref[...]=m_next[...].astype(m_scratch_ref.dtype)
    l_scratch_ref[...]=l_next[...].astype(l_scratch_ref.dtype)

    v = v_ref[...]
    o_curr=s_curr@v

    alpha_o = jnp.exp(m_prev - m_next)
    o_ref[...] = (alpha_o*o_ref[:]+o_curr).astype(o_ref.dtype)

    @pl.when(kv_seq_idx == (kv_seq_len // block_k_major) - 1)
    def out():
        o_ref[...] = o_ref[...] / l_scratch_ref[...]




def matmul(q: jax.Array, k: jax.Array,v: jax.Array,mask,block_q:int=128,block_kv=128):

    h,n,d=q.shape

    grids=(h,n//block_q,n//block_kv)

    matmul_kernels=functools.partial(
        matmul_kernel,kv_seq_len=n
    )

    print(grids,n // block_q)

    def q_index_map(head_index, q_seq_index, _):
        return head_index,q_seq_index,0

    def kv_index_map(head_index, q_seq_index, kv_seq_index):
        next_kv_index = kv_seq_index
        return head_index,next_kv_index,0


    def mask_index_map(head_index, q_seq_index, kv_seq_index):
        return None,q_seq_index,kv_seq_index


    def out_index_map(head_index, q_seq_index, _):
        return head_index,q_seq_index,0



    out_shape=[
        jax.ShapeDtypeStruct((n, 1), q.dtype),
        jax.ShapeDtypeStruct((n, d), q.dtype),
        jax.ShapeDtypeStruct((  h,n, d), q.dtype)
    ]

    in_specs = [
        pl.BlockSpec(block_shape=(None,block_q, d), index_map=q_index_map, ),
        pl.BlockSpec(block_shape=(None,block_kv, d), index_map=kv_index_map, ),
        pl.BlockSpec(block_shape=(None,block_kv, d), index_map=kv_index_map, ),
        pl.BlockSpec(block_shape=(block_q, block_kv), index_map=mask_index_map, ),

    ]

    out_specs = [
        pl.BlockSpec(block_shape=(block_q, 1), index_map=lambda h,i, j: (i, 0), ),
        pl.BlockSpec(block_shape=( block_q, d), index_map=lambda h,i, j: (i, 0), ),
        pl.BlockSpec(block_shape=(None, block_q, d), index_map=lambda h,i, j: (h,i, 0), ),
    ]



    _,_,output= jax.jit(pl.pallas_call(
        kernel=matmul_kernels,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grids,
            in_specs=in_specs,
            out_specs=out_specs,
            # scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shape,

        compiler_params=pltpu.TPUCompilerParams(
            dimension_semantics=("parallel","parallel", "arbitrary"),
        ),
              interpret=False,
    ))(
        q,k,v,mask,

       )
    return output



def flash_attention(q,k,v,mask):
    return matmul(q,k,v,mask[0])

