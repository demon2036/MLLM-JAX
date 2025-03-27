from __future__ import annotations

from typing import Callable, Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from chex import Array, ArrayTree



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









class TrainGRPOModule(nn.Module):
    model: Any
    pad_token_id:float
    num_pre_Q:int =8
    ref_model: Any =None
    beta:float =0.04



    def __call__(self, inputs,) -> ArrayTree:
        input_ids=inputs['input_ids']
        attention_mask=inputs['attention_mask']
        labels=inputs['labels']
        rewards = inputs['rewards']
        advantages=getattr(inputs, "advantages", None)
        print(advantages.shape)


        logits, cache = self.model( input_ids=input_ids,
                                       attention_mask=attention_mask)


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



        per_token_logps = jnp.take_along_axis(  # [B, S]
            jax.nn.log_softmax(logits[..., :-1, :], axis=-1), chosen_ids[..., None], axis=-1
        )[..., 0]


        if advantages is None:
            mean_grouped_rewards = rewards.reshape(-1, self.num_pre_Q).mean(axis=1)
            std_grouped_rewards = rewards.reshape(-1, self.num_pre_Q).std(axis=1)
            mean_grouped_rewards = jnp.repeat(mean_grouped_rewards, self.num_pre_Q, axis=0)
            std_grouped_rewards = jnp.repeat(std_grouped_rewards, self.num_pre_Q, axis=0)
            advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        print(advantages.shape)

        per_token_loss = jnp.exp(per_token_logps - jax.lax.stop_gradient(per_token_logps)) * advantages[..., None]

        if self.beta!=0:
            per_token_kl = jnp.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

            per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        else:
            per_token_loss = -per_token_loss


        loss = ((per_token_loss * mask_loss).sum(axis=1) / mask_loss.sum(axis=1)).mean()
        return {"loss": loss, }
