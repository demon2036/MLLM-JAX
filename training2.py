import copy
import os
from typing import Any

import flax

os.environ['JAX_TRACEBACK_FILTERING']='off'

import re

import jax.tree_util
import optax
from chex import ArrayTree
from flax.training import train_state

from MLLM_JAX.train_modules import TrainGRPOModule
from MLLM_JAX.utils import match_partition_rules, get_partition_rules_llama
from MLLM_JAX.sample.sample_state_right_padding2 import get_model, Sampler
# from sample_state_left_padding import get_model, Sampler
import jax.numpy as jnp
from math_verify import parse, verify, ExprExtractionConfig


def slice_data(x,accumulate_steps,i):
    b,*_=x.shape
    assert b%accumulate_steps==0
    micro_batch_size=b//accumulate_steps
    data=x[i*micro_batch_size:(i+1)*micro_batch_size]
    # print(data.shape)
    return data






class TrainState(train_state.TrainState):
    micro_step: int = 0
    micro_in_mini: int = 1
    grad_accum: ArrayTree | None = None
    ref_params:Any=None

    ema_params: ArrayTree|None = None
    ema_decay: float = 0.9998


def get_state(mesh,training_steps=100,grad_accum_steps=1,model_path='Qwen/Qwen2.5-3B',num_pre_q=16,max_lengths=None):
    model, params, tokenizer = get_model(mesh,model_path=model_path, )
    model_ref = get_model(mesh, model_path=model_path, only_model=True)

    beta=0.04

    train_module = flax.linen.remat(TrainGRPOModule,policy=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims)(model=model,
                                   pad_token_id=tokenizer.pad_token_id,
                                   ref_model=model_ref,
                                   num_pre_Q=num_pre_q,
                                   beta=beta,
                                   max_lengths=max_lengths,
                                   )

    # train_module = TrainGRPOModule(
    #     model=model,
    #     pad_token_id=tokenizer.pad_token_id,
    #     ref_model=model_ref,
    #     num_pre_Q=num_pre_q,
    #     beta=beta,
    #     max_lengths=max_lengths,
    #     )



    def init_fn(params):

        learning_rate = optax.warmup_cosine_decay_schedule(
            init_value=0,
            peak_value=1e-6,
            warmup_steps=int(training_steps*0.05),
            decay_steps=training_steps,
            end_value=0,
        )
        # tx = optax.adamw(learning_rate)
        tx = optax.lion(learning_rate,weight_decay=1e-8)
        # tx = optax.sgd(learning_rate)
        tx = optax.chain(optax.clip_by_global_norm(1.0), tx)
        if grad_accum_steps > 1:
            print(f'{grad_accum_steps=}')
            grad_accum = jax.tree_util.tree_map(jnp.zeros_like, params)


        return TrainState.create(apply_fn=train_module.apply,params=params,tx=tx,
                                 ref_params=copy.deepcopy(params) if beta!=0 else None,
                                 micro_step=0,
                                 micro_in_mini=grad_accum_steps,
                                 grad_accum=grad_accum if grad_accum_steps > 1 else None,
                                 # ema_decay=0.99,
                                 # ema_params=copy.deepcopy(params),
                                 )


    state_shapes = jax.eval_shape(init_fn, params, )
    train_state_partition = match_partition_rules(get_partition_rules_llama(), state_shapes)
    train_state_sharding = jax.tree_util.tree_map(lambda x: jax.sharding.NamedSharding(mesh, x), train_state_partition)
    state = jax.jit(init_fn, donate_argnums=(0,),out_shardings=train_state_sharding)(params)

    sampler = Sampler(model, tokenizer,mesh=mesh,)


    return state,sampler,train_state_sharding




def training_step(state: TrainState, inputs: ArrayTree) -> tuple[TrainState, ArrayTree]:

    def loss_fn(params: ArrayTree) -> ArrayTree:
        metrics=state.apply_fn({'params': {'model':params,'ref_model':state.ref_params }, },inputs)
        per_token_logps=metrics.pop('per_token_logps',None)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return metrics["loss"], metrics |{'per_token_logps':per_token_logps}

    def update_fn(state: TrainState) -> TrainState:

        # Collect a global gradient from the accumulated gradients and apply actual
        # parameter update with resetting the accumulations to zero.
        # grads = jax.tree_map(lambda g: g / state.micro_in_mini, state.grad_accum)
        state = state.apply_gradients(
            grads=grads,
            grad_accum=jax.tree_util.tree_map(jnp.zeros_like, state.grad_accum),
            micro_step=state.micro_step % state.micro_in_mini,
        )

        # new_ema_params = jax.tree_util.tree_map(
        #     lambda ema, normal: ema * state.ema_decay + (1 - state.ema_decay) * normal,
        #     state.ema_params, state.params)
        # state = state.replace(ema_params=new_ema_params)

        return state


    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # Update parameters with the gradients. If the gradient accumulation is enabled,
    # then the parameters will be updated at the end of each mini-batch step. In every
    # micro steps, the gradients will be accumulated.
    if state.grad_accum is None:
        state = state.apply_gradients(grads=grads)
    else:
        state = state.replace(
            grad_accum=jax.tree_util.tree_map(lambda ga, g: ga + g, state.grad_accum, grads),
            micro_step=state.micro_step + 1,
        )
        state = jax.lax.cond(
            state.micro_step == state.micro_in_mini, update_fn, lambda x: x, state
        )
    return state, metrics



def repeat(lst:list,repeats:int):
    return [x    for x in lst  for _ in range(repeats) ]



def reward_correct(item, answer):
    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    nums = re.findall(pattern, answer) # 使用正则表达式在answer中查找所有数字
    if len(nums) == 0: return 0.0
    lastnum = nums[-1] # 用answer中最后一个数字和ground_truth做比较
    ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
    ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
    # print(item["A"],ans)

    return 1 if verify(ans, ground_truth) else 0.0



def reward_format(item, answer, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    # pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    pattern = r"^<think>.*?</think>\n<answer>.*?</answer>$"
    match = re.match(pattern, answer, re.DOTALL | re.MULTILINE)
    return 1.0 if match else 0.0



def tag_count_reward(item, answer, **kwargs) -> float:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>") == 1:
            count += 0.25
        if text.count("</think>") == 1:
            count += 0.25
        if text.count("<answer>") == 1:
            count += 0.25
        if text.count("</answer>") == 1:
            count += 0.25
        return count

    return count_tags(answer)





def get_advantages(rewards,groups,advantage_estimator='grpo',alpha=0.02,mean_global=None,std_global=None,avg_entropy_per_sample=None):

    if advantage_estimator=='grpo':
        mean_grouped_rewards = rewards.reshape(-1, groups).mean(axis=1)
        std_grouped_rewards = rewards.reshape(-1, groups).std(axis=1)
        mean_grouped_rewards = jnp.repeat(mean_grouped_rewards, groups, axis=0)
        std_grouped_rewards = jnp.repeat(std_grouped_rewards, groups, axis=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
    elif advantage_estimator=='dr_grpo':
        mean_grouped_rewards = rewards.reshape(-1, groups).mean(axis=1)
        mean_grouped_rewards = jnp.repeat(mean_grouped_rewards, groups, axis=0)
        advantages = (rewards - mean_grouped_rewards)

    elif advantage_estimator=='grpo_clip':
        mean_grouped_rewards = rewards.reshape(-1, groups).mean(axis=1)
        std_grouped_rewards = rewards.reshape(-1, groups).std(axis=1)
        mean_grouped_rewards = jnp.repeat(mean_grouped_rewards, groups, axis=0)
        std_grouped_rewards = jnp.repeat(std_grouped_rewards, groups, axis=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        max_grouped_advantages=advantages.reshape(-1, groups).max(axis=1)
        max_grouped_advantages = jnp.repeat(max_grouped_advantages, groups, axis=0)
        advantages=jnp.clip(advantages,-4*max_grouped_advantages,None)

    elif advantage_estimator=='grpo_clip2':
        mean_grouped_rewards = rewards.reshape(-1, groups).mean(axis=1)
        std_grouped_rewards = rewards.reshape(-1, groups).std(axis=1)
        mean_grouped_rewards = jnp.repeat(mean_grouped_rewards, groups, axis=0)
        std_grouped_rewards = jnp.repeat(std_grouped_rewards, groups, axis=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        advantages=advantages+alpha*(rewards - mean_global) / (std_global + 1e-4)
        max_grouped_advantages=advantages.reshape(-1, groups).max(axis=1)
        max_grouped_advantages = jnp.repeat(max_grouped_advantages, groups, axis=0)

        group_max = advantages.reshape(-1, groups).max(axis=1)
        scale_factors = 1.5 / (group_max + 1e-6)
        scale_factors = jnp.repeat(scale_factors, groups, axis=0)
        advantages = jnp.where(advantages > 0, advantages * scale_factors, advantages)

        # group_min = advantages.reshape(-1, groups).min(axis=1)
        # scale_factors_neg = -0.5 / (group_min + 1e-6)
        # scale_factors_neg = jnp.repeat(scale_factors_neg, groups, axis=0)
        # advantages = jnp.where(advantages < 0, advantages * scale_factors_neg, advantages)

        advantages = jnp.clip(advantages, -4 * max_grouped_advantages, None)
    elif advantage_estimator == 'john_grpo_replace':
        mean_grouped_rewards = rewards.reshape(-1, groups).mean(axis=1)
        std_grouped_rewards = rewards.reshape(-1, groups).std(axis=1)
        mean_grouped_rewards = jnp.repeat(mean_grouped_rewards, groups, axis=0)
        std_grouped_rewards = jnp.repeat(std_grouped_rewards, groups, axis=0)

        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        advantages_global = (rewards - mean_global) / (std_global + 1e-4)
        advantages=jnp.where(std_grouped_rewards<std_global,advantages_global,advantages)

    elif advantage_estimator == 'john_grpo':
        mean_grouped_rewards = rewards.reshape(-1, groups).mean(axis=1)
        std_grouped_rewards = rewards.reshape(-1, groups).std(axis=1)
        mean_grouped_rewards = jnp.repeat(mean_grouped_rewards, groups, axis=0)
        std_grouped_rewards = jnp.repeat(std_grouped_rewards, groups, axis=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4) +alpha*(rewards - mean_global) / (std_global + 1e-4)
    elif advantage_estimator =='grpo_entropy':
        avg_entropy_grouped = avg_entropy_per_sample.reshape(-1, groups)
        # Ranks within each group (0=lowest entropy, groups-1=highest)
        ranks_grouped = jnp.argsort(jnp.argsort(avg_entropy_grouped, axis=1), axis=1)

        denom_grp = jnp.maximum(groups - 1.0, 0)  # Avoid division by zero if groups=1
        entropy_scores_grouped = -1.0 + 2.0 * ranks_grouped.astype(jnp.float32) / denom_grp

        entropy_scores = entropy_scores_grouped.reshape(-1)  # Shape [B]

        # 2. Combine original rewards with weighted entropy score
        modified_rewards = rewards + alpha * entropy_scores  # Shape [B]
        # 3. Apply standard GRPO normalization to the *modified* rewards
        mean_grouped_mod_rewards = modified_rewards.reshape(-1, groups).mean(axis=1)
        std_grouped_mod_rewards = modified_rewards.reshape(-1, groups).std(axis=1)
        mean_grouped_mod_rewards = jnp.repeat(mean_grouped_mod_rewards, groups, axis=0)
        std_grouped_mod_rewards = jnp.repeat(std_grouped_mod_rewards, groups, axis=0)
        advantages = (modified_rewards - mean_grouped_mod_rewards) / (std_grouped_mod_rewards + 1e-4)
        return advantages
    elif advantage_estimator=='reinforce':
        return (rewards-mean_global)/std_global


    return advantages





def mean(x):
    return x.mean()


def init_fn(x):
    return x
