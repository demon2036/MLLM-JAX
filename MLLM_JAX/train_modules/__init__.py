from __future__ import annotations

from typing import Callable, Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import torch
from chex import Array, ArrayTree
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

CRITERION_COLLECTION = {
    "ce": optax.softmax_cross_entropy,
    "bce": lambda x, y: optax.sigmoid_binary_cross_entropy(x, y > 0).mean(-1),
}


class TrainSFTModule(nn.Module):
    model: Any
    pad_token_id:float
    label_smoothing: float = 0.0
    criterion: Callable[[Array, Array], Array] = CRITERION_COLLECTION["ce"]

    def __call__(self, inputs) -> ArrayTree:
        input_ids=inputs['input_ids']
        attention_mask=inputs['attention_mask']
        labels=inputs['labels']

        logits, cache = self.model( input_ids=input_ids[:, :-1],
                                       attention_mask=attention_mask[:, :-1])

        targets=jax.nn.one_hot(labels[:,1:],logits.shape[-1])
        loss=optax.softmax_cross_entropy(logits,targets)
        mask_loss=labels[:,1:]!=self.pad_token_id
        loss=loss*mask_loss
        loss_s=loss.sum()/(mask_loss.astype(jnp.float32).sum())


        targets = targets == targets.max(-1, keepdims=True)

        # Instead of directly comparing the maximum classes of predicted logits with the
        # given one-hot labels, we will check if the predicted classes are within the
        # label set. This approach is equivalent to traditional methods in single-label
        # classification and also supports multi-label tasks.
        preds = jax.lax.top_k(logits, k=5)[1]
        accs = jnp.take_along_axis(targets, preds, axis=-1)
        return {"loss": loss_s, }


# def selective_log_softmax_jax(logits: jnp.ndarray, index: jnp.ndarray) -> jnp.ndarray:
#     """
#     A memory-efficient JAX implementation for `log(softmax(logits))[index]`.
#
#     Equivalent to: `jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), index[..., None], axis=-1).squeeze(axis=-1)`
#     but avoids materializing the full log_softmax result.
#
#     Args:
#         logits: Logits tensor of shape `(..., num_classes)`.
#         index: Index tensor of shape `(...)`, specifying the positions to gather
#                from the log-softmax output along the last axis.
#
#     Returns:
#         Gathered log probabilities with the same shape as `index`.
#     """
#     # Ensure index has the same number of dimensions as logits for broadcasting subtraction later
#     # Or ensure index can be broadcasted correctly against logits dimensions before the last one.
#     # If index is (...) and logits is (..., V), ensure (...) matches the prefix shape.
#
#     # Gather the logits corresponding to the target indices: shape (...)
#     # Add a dimension to index for take_along_axis, then remove it
#     selected_logits = jnp.take_along_axis(logits, index[..., None], axis=-1).squeeze(axis=-1)
#
#     # Calculate the logsumexp normalization factor across the last dimension: shape (...)
#     logsumexp_logits = jax.nn.logsumexp(logits, axis=-1)
#
#     # Apply the identity: log_softmax(x)_i = x_i - logsumexp(x)
#     per_token_logps = selected_logits - logsumexp_logits
#
#     return per_token_logps



def selective_log_softmax_jax_inner(logits,index):
    return jnp.take_along_axis(  # [B, S]
        jax.nn.log_softmax(logits, axis=-1), index[..., None], axis=-1
    )[..., 0]


def selective_log_softmax_jax(logits: jnp.ndarray, index: jnp.ndarray,mesh) -> jnp.ndarray:
    """
    A memory-efficient JAX implementation for `log(softmax(logits))[index]`.

    Equivalent to: `jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), index[..., None], axis=-1).squeeze(axis=-1)`
    but avoids materializing the full log_softmax result.

    Args:
        logits: Logits tensor of shape `(..., num_classes)`.
        index: Index tensor of shape `(...)`, specifying the positions to gather
               from the log-softmax output along the last axis.

    Returns:
        Gathered log probabilities with the same shape as `index`.

    """


    return shard_map(jax.vmap(selective_log_softmax_jax_inner),mesh=mesh,
                     in_specs=P(['dp','fsdp']),out_specs=P(['dp','fsdp'])
                     )(logits,index)





class TrainGRPOModule(nn.Module):
    model: Any
    pad_token_id:float
    num_pre_Q:int =8
    ref_model: Any =None
    beta:float =0.04
    temperature:float =0.9
    max_lengths:float=2048
    mesh:Any=None



    def __call__(self, inputs,) -> ArrayTree:
        input_ids=inputs['input_ids']
        attention_mask=inputs['attention_mask']
        labels=inputs['labels']
        rewards = inputs['rewards']
        advantages=inputs.get( "advantages", None)


        logits, cache = self.model( input_ids=input_ids,
                                       attention_mask=attention_mask)

        logits=logits[..., :-1, :]

        print(logits.dtype)



        chosen_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # mask_loss = labels[:, 1:] != self.pad_token_id
        mask_loss = labels[:, 1:]

        if self.beta!=0:
            ref_logits, cache = self.ref_model( input_ids=input_ids,
                                           attention_mask=attention_mask)

            ref_logits=jax.lax.stop_gradient(ref_logits)

            ref_per_token_logps = jnp.take_along_axis(  # [B, S]
                jax.nn.log_softmax(ref_logits[..., :-1, :], axis=-1), chosen_ids[..., None], axis=-1
            )[..., 0]



        # per_token_logps = jnp.take_along_axis(  # [B, S]
        #     jax.nn.log_softmax(logits, axis=-1), chosen_ids[..., None], axis=-1
        # )[..., 0]/self.temperature

        per_token_logps = selective_log_softmax_jax(logits,chosen_ids,self.mesh)/self.temperature



        if advantages is None:
            mean_grouped_rewards = rewards.reshape(-1, self.num_pre_Q).mean(axis=1)
            std_grouped_rewards = rewards.reshape(-1, self.num_pre_Q).std(axis=1)
            mean_grouped_rewards = jnp.repeat(mean_grouped_rewards, self.num_pre_Q, axis=0)
            std_grouped_rewards = jnp.repeat(std_grouped_rewards, self.num_pre_Q, axis=0)
            # advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
            advantages = (rewards - mean_grouped_rewards)

        print(advantages.shape)

        per_token_loss = jnp.exp(per_token_logps - jax.lax.stop_gradient(per_token_logps)) * advantages[..., None]

        if self.beta!=0:
            per_token_kl = jnp.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

            per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        else:
            per_token_loss = -per_token_loss


        # loss = ((per_token_loss * mask_loss).sum(axis=1) / mask_loss.sum(axis=1)).mean()

        loss = ((per_token_loss * mask_loss).sum(axis=1) / self.max_lengths ).mean()

        return {"loss": loss, }
