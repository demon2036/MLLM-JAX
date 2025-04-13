# %% Imports
import random
import logging
import warnings
warnings.filterwarnings("ignore")

from functools import partial
from typing import Dict, List, Tuple, Any, Callable # No longer need Deque
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
import wandb

# JAX specific imports
# os.environ['JAX_TRACEBACK_FILTERING']='off'
#
# jax.distributed.initialize()
from jax.sharding import NamedSharding
from jax.experimental.multihost_utils import process_allgather

# Local imports
from prompts.prompts import system_prompt
from training2 import (reward_correct, reward_format, get_state, training_step,
                       repeat, slice_data, get_advantages, tag_count_reward, init_fn)
from MLLM_JAX.utils import (get_jax_mesh2, _form_global_array, match_partition_rules, get_partition_rules_llama)
# %% Configuration & Data Structures

@dataclass
class ReplayBufferEntry:
    """Represents a single entry in the replay buffer."""
    original_input: Dict[str, str]
    prompt_used: str
    generated_answer: str
    total_reward: float
    rewards_per_func: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class TrainingConfig:
    """Configuration settings for the training script."""
    # Model and Tokenizer
    model_path: str = 'Qwen/Qwen2.5-1.5B-Instruct'
    max_length_sample: int = 1024
    max_length_total: int = max_length_sample + 512

    # Dataset
    dataset_name: str = "openai/gsm8k"
    dataset_split: str = "train"

    # Training Hyperparameters
    num_pre_q: int = 16
    batch_size: int = 1
    training_steps: int = 400
    grad_accum_steps: int = 1
    ppo_epochs: int = 2

    # JAX/Distributed Settings
    mesh_shape_dp: str = "-1,1,1"
    mesh_shape_fsdp: str = "1,-1,1"

    # Replay Buffer Settings (Infinite Size)
    # replay_buffer_max_size: int = 10000 # REMOVED - Buffer size is now infinite
    sample_from_buffer_prob: float = 0.25
    initial_buffer_fill_steps: int = 50

    # Reward Functions and Weights (Set in reward_setup)
    reward_funcs_weights: Dict[str, float] = field(default_factory=dict)

    # Logging
    wandb_project: str = 'grop-gsm8k'
    wandb_run_name: str = 'refactored_v3_infinite_replay' # Updated run name
    log_level: int = logging.INFO

# Setup Logging
logging.basicConfig(level=TrainingConfig.log_level, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# %% Reward Functions Setup
def reward_setup() -> Tuple[List[Callable], List[float]]:
    """Defines the reward functions and their corresponding weights."""
    reward_functions = [reward_correct, reward_format, tag_count_reward]
    reward_weights = [1.0, 0.5, 0.5]
    assert len(reward_functions) == len(reward_weights), "Mismatch between reward functions and weights."
    return reward_functions, reward_weights

# %% Helper Functions (apply_chat_template, load_data, setup_jax - unchanged)
def apply_chat_template(tokenizer: PreTrainedTokenizerBase, question: str) -> str:
    """Applies the chat template to a given question."""
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        tokenize=False, add_generation_prompt=True
    )

def load_data(config: TrainingConfig) -> List[Dict[str, str]]:
    """Loads and prepares the dataset."""
    logger.info(f"Loading dataset: {config.dataset_name}, split: {config.dataset_split}")
    dataset = load_dataset(config.dataset_name, "main", split=config.dataset_split)
    dataset = dataset.shard(num_shards=jax.process_count(), index=jax.process_index())
    qas = [{'Q': x, 'A': y.split('####')[-1].strip()} for x, y in zip(dataset['question'], dataset['answer'])]
    logger.info(f"Loaded {len(qas)} Q/A pairs for process {jax.process_index()}.")
    return qas

def setup_jax(config: TrainingConfig) -> Dict[str, Any]:
    """Initializes JAX mesh, state, sampler, and JIT functions."""
    logger.info("Setting up JAX environment...")
    mesh_dp = get_jax_mesh2(config.mesh_shape_dp)
    mesh_fsdp = get_jax_mesh2(config.mesh_shape_fsdp)

    state, sampler, _ = get_state(
        mesh_fsdp,
        config.training_steps,
        model_path=config.model_path,
        grad_accum_steps=config.grad_accum_steps,
        num_pre_q=config.num_pre_q,
        max_lengths=config.max_length_total
    )

    params_shapes = jax.eval_shape(init_fn, state.params)
    params_partition = match_partition_rules(get_partition_rules_llama(), params_shapes)
    params_sharding_dp = jax.tree_util.tree_map(lambda x: NamedSharding(mesh_dp, x), params_partition)
    params_sharding_fsdp = jax.tree_util.tree_map(lambda x: NamedSharding(mesh_fsdp, x), params_partition)

    params_to_dp = jax.jit(init_fn, out_shardings=params_sharding_dp)
    params_to_fsdp = jax.jit(init_fn, out_shardings=params_sharding_fsdp)
    get_advantages_jit = jax.jit(get_advantages, static_argnums=(1,))
    train_fn_jit = jax.jit(training_step, donate_argnums=(0,))

    logger.info("JAX setup complete.")
    return {
        "state": state,
        "sampler": sampler,
        "mesh_dp": mesh_dp,
        "params_to_dp": params_to_dp,
        "get_advantages_jit": get_advantages_jit,
        "train_fn_jit": train_fn_jit,
        "tokenizer": sampler.tokenizer
    }

# %% Core Logic Functions (run_generation_step, calculate_rewards - unchanged)
def run_generation_step(
    prompts: List[str],
    jax_setup: Dict[str, Any],
    config: TrainingConfig
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Generates answers using the model. Returns generated answers and data dict."""
    sampler = jax_setup["sampler"]
    state = jax_setup["state"]
    params_to_dp = jax_setup["params_to_dp"]
    tokenizer = jax_setup["tokenizer"]

    params_dp_bf16 = params_to_dp(jax.tree_util.tree_map(lambda x: jnp.astype(x, jnp.bfloat16), state.params))

    inputs = tokenizer(prompts, return_tensors="jax", padding=True, padding_side="right")
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    position_ids = attention_mask.cumsum(-1) - 1
    position_ids = jnp.where(attention_mask == 0, 1, position_ids)

    prefill_length = sampler.find_ceil(config.max_length_total)
    pad_width = max(0, prefill_length - input_ids.shape[1])

    if pad_width > 0:
         input_ids = jnp.pad(input_ids, ((0, 0), (0, pad_width)), constant_values=tokenizer.eos_token_id)
         attention_mask = jnp.pad(attention_mask, ((0, 0), (0, pad_width)), constant_values=0)
         position_ids = jnp.pad(position_ids, ((0, 0), (0, pad_width)), constant_values=1)
    elif input_ids.shape[1] > prefill_length:
         logger.warning(f"Input length {input_ids.shape[1]} exceeds prefill length {prefill_length}. Truncating.")
         input_ids = input_ids[:, :prefill_length]
         attention_mask = attention_mask[:, :prefill_length]
         position_ids = position_ids[:, :prefill_length]

    true_length_prompts = (inputs['attention_mask']).sum(axis=1)

    logger.info(f"Generating completions for {len(prompts)} prompts...")


    outputs = sampler.generate(
        input_ids_pad=input_ids,
        pad_attention=attention_mask,
        position_ids=position_ids,
        prefill_length=prefill_length,
        max_length=config.max_length_sample,
        params=params_dp_bf16
    )
    logger.info("Generation complete.")

    buffer_len = outputs['local_token_buffer'].shape[1]
    train_input_ids = np.full((len(prompts), buffer_len), fill_value=tokenizer.pad_token_id, dtype=np.int32)
    train_attention_mask = np.zeros_like(train_input_ids, dtype=np.int32)
    train_completions_mask = np.zeros_like(train_input_ids, dtype=np.int32)
    generated_answers = []

    for i, (true_len, gen_step) in enumerate(zip(true_length_prompts, outputs['local_sample_step'])):
        start_idx = prefill_length
        end_idx = start_idx + gen_step + 1
        start_idx = min(start_idx, buffer_len)
        end_idx = min(end_idx, buffer_len)

        generated_tokens = outputs['local_token_buffer'][i, start_idx:end_idx]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        generated_answers.append(answer)

        prompt_tokens = inputs['input_ids'][i, :true_len]
        train_input_ids[i, :true_len] = prompt_tokens
        train_attention_mask[i, :true_len] = 1

        gen_len = end_idx - start_idx
        prompt_end_idx = true_len + gen_len
        prompt_end_idx = min(prompt_end_idx, buffer_len)
        actual_gen_len = prompt_end_idx - true_len

        if actual_gen_len > 0:
            train_input_ids[i, true_len:prompt_end_idx] = generated_tokens[:actual_gen_len]
            train_attention_mask[i, true_len:prompt_end_idx] = 1
            train_completions_mask[i, true_len:prompt_end_idx] = 1

    logger.info(f"Sample generated answers (last 2): {generated_answers[-2:]}")

    max_len = config.max_length_total
    data = {
        'input_ids': train_input_ids[:, :max_len],
        'attention_mask': train_attention_mask[:, :max_len],
        'labels': train_completions_mask[:, :max_len],
    }
    return generated_answers, data

def calculate_rewards(
    repeated_inputs: List[Dict[str, str]],
    generated_answers: List[str],
    reward_functions: List[Callable],
    reward_weights: List[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates rewards for each generated answer based on multiple criteria."""
    num_answers = len(generated_answers)
    num_funcs = len(reward_functions)
    rewards_per_func = np.zeros((num_funcs, num_answers))

    for i, reward_func in enumerate(reward_functions):
        func_weight = reward_weights[i]
        func_name = reward_func.__name__
        for j, (inp, ans) in enumerate(zip(repeated_inputs, generated_answers)):
            try:
                reward_val = reward_func(inp, ans)
                rewards_per_func[i, j] = func_weight * reward_val
            except Exception as e:
                logger.error(f"Error calculating reward '{func_name}' for sample {j}: {e}", exc_info=False)
                rewards_per_func[i, j] = -1.0

    total_rewards = rewards_per_func.sum(axis=0)
    return total_rewards, rewards_per_func

# %% Replay Buffer Update Function (MODIFIED LOGIC)
def update_replay_buffer(
    replay_buffer: List[ReplayBufferEntry], # Use List now
    repeated_inputs: List[Dict[str, str]],
    prompts_used: List[str],
    generated_answers: List[str],
    total_rewards: np.ndarray,
    rewards_per_func_values: np.ndarray,
    reward_functions: List[Callable],
    step: int,
    config: TrainingConfig # Config still passed for potential future use
) -> None:
    """Adds new experiences to the replay buffer (infinite size)."""
    # NOTE: This buffer grows indefinitely and may consume large amounts of memory.
    num_new_entries = len(generated_answers)
    added_count = 0

    reward_func_names = [func.__name__ for func in reward_functions]

    for i in range(num_new_entries):
        rewards_dict = {name: rewards_per_func_values[k, i] for k, name in enumerate(reward_func_names)}

        entry = ReplayBufferEntry(
            original_input=repeated_inputs[i],
            prompt_used=prompts_used[i],
            generated_answer=generated_answers[i],
            total_reward=total_rewards[i],
            rewards_per_func=rewards_dict,
            metadata={'step_added': step}
        )
        replay_buffer.append(entry) # Simply append to the list
        added_count += 1

    if added_count > 0:
        logger.debug(f"Added {added_count} entries to replay buffer.")
    # No eviction logic needed
    logger.info(f"Replay buffer size: {len(replay_buffer)}")


# %% PPO Update & Logging Functions (perform_ppo_update, collect_and_log_metrics - unchanged except buffer size logging)
def perform_ppo_update(
    jax_setup: Dict[str, Any],
    datas: Dict[str, jnp.ndarray],
    config: TrainingConfig
) -> Tuple[Any, Dict[str, Any]]:
    """Performs PPO optimization steps for one batch of data."""
    state = jax_setup["state"]
    train_fn_jit = jax_setup["train_fn_jit"]
    mesh = jax_setup["mesh_dp"]

    datas = jax.tree_util.tree_map_with_path(
        partial(_form_global_array, global_mesh=mesh), datas
    )

    total_valid_token_count = datas['labels'][:, 1:].sum()

    per_token_logps_list = []
    final_meta_data = {}

    logger.info(f"Starting PPO updates ({config.ppo_epochs} epochs)...")
    for ppo_epoch in range(config.ppo_epochs):
        for accum_step in range(config.grad_accum_steps):
            local_data = jax.tree_util.tree_map(
                lambda x: slice_data(x, config.grad_accum_steps, accum_step),
                datas
            )
            local_data['total_valid_token_count'] = total_valid_token_count

            state, meta_data = train_fn_jit(state, local_data)
            final_meta_data = meta_data

            if ppo_epoch == 0 and 'per_token_logps' in meta_data:
                 per_token_logps_list.append(meta_data['per_token_logps'])

        if ppo_epoch == 0 and per_token_logps_list:
            datas['old_per_token_logps'] = jnp.concatenate(per_token_logps_list)
            logger.info("Stored old log probabilities.")

    logger.info("PPO updates finished.")
    jax_setup["state"] = state
    return final_meta_data


def collect_and_log_metrics(
    step: int,
    total_rewards_local: np.ndarray,
    rewards_per_func_local: np.ndarray,
    reward_functions: List[Callable],
    advantages_local: np.ndarray,
    completion_ids_local: np.ndarray,
    final_ppo_metadata: Dict[str, Any],
    config: TrainingConfig,
    buffer_size: int # Still log the current size
) -> None:
    """Gathers data across hosts, calculates metrics, and logs them."""
    metrics = {}

    rewards_global = process_allgather(total_rewards_local)
    advantages_global = process_allgather(advantages_local)
    completion_ids_global = process_allgather(completion_ids_local)

    mean_global = rewards_global.mean()
    std_global = rewards_global.std()
    metrics['reward/global_mean'] = mean_global
    metrics['reward/global_std'] = std_global
    metrics['replay_buffer/size'] = buffer_size # Log current size

    reward_func_names = [func.__name__ for func in reward_functions]
    for i, name in enumerate(reward_func_names):
        rewards_func_global = process_allgather(rewards_per_func_local[i])
        metrics[f'reward/{name}_mean'] = rewards_func_global.mean()

    metrics['ppo/advantages_max'] = advantages_global.max()
    metrics['ppo/advantages_min'] = advantages_global.min()
    metrics['ppo/advantages_mean'] = advantages_global.mean()

    try:
        correct_idx = reward_func_names.index('reward_correct')
        # Ensure reward_funcs_weights exists and has the key before accessing
        if 'reward_correct' in config.reward_funcs_weights and config.reward_funcs_weights['reward_correct'] != 0:
             reward_corrects_local = rewards_per_func_local[correct_idx] / config.reward_funcs_weights['reward_correct']
        else:
             # Handle case where weight is 0 or missing (treat as raw value)
             reward_corrects_local = rewards_per_func_local[correct_idx]
        correct_mask_global = process_allgather(reward_corrects_local) == 1.0
    except (ValueError, KeyError, IndexError): # Catch potential errors
        logger.warning("Could not find 'reward_correct' or its weight for length stats.")
        correct_mask_global = np.zeros(rewards_global.shape, dtype=bool)

    completion_lengths = completion_ids_global.sum(axis=-1)
    print(completion_lengths.shape,completion_ids_global.shape,completion_ids_local.shape)
    if correct_mask_global.any():
         metrics['completion/correct_length_mean'] = completion_lengths[correct_mask_global].mean()
         metrics['completion/correct_length_max'] = completion_lengths[correct_mask_global].max()
    else:
         metrics['completion/correct_length_mean'] = 0
         metrics['completion/correct_length_max'] = 0

    if (~correct_mask_global).any():
         metrics['completion/incorrect_length_mean'] = completion_lengths[~correct_mask_global].mean()
         metrics['completion/incorrect_length_max'] = completion_lengths[~correct_mask_global].max()
    else:
         metrics['completion/incorrect_length_mean'] = 0
         metrics['completion/incorrect_length_max'] = 0

    if 'entropy' in final_ppo_metadata:
        metrics['ppo/entropy'] = final_ppo_metadata['entropy']

    if jax.process_index() == 0:
        logger.info(f"Step {step}: Logging metrics: { {k: f'{v:.4f}' if isinstance(v, float) else v for k, v in metrics.items()} }")
        try:
             wandb.log(metrics, step=step)
        except Exception as e:
             logger.error(f"WandB logging failed: {e}")


# %% Main Training Loop (MODIFIED INITIALIZATION)
def main():
    """Main function to run the PPO training loop."""
    try:
        jax.distributed.initialize()
        logger.info(f"JAX Distributed initialized. Process count: {jax.process_count()}, Index: {jax.process_index()}")
    except Exception as e:
        logger.warning(f"Could not initialize JAX distributed: {e}. Running in single-process mode.")

    config = TrainingConfig()
    reward_functions, reward_weights = reward_setup()
    config.reward_funcs_weights = {func.__name__: weight for func, weight in zip(reward_functions, reward_weights)}

    jax_setup = setup_jax(config)
    qas_data = load_data(config)
    tokenizer = jax_setup["tokenizer"]

    if jax.process_index() == 0:
        try:
            wandb.init(name=config.wandb_run_name, project=config.wandb_project, config=vars(config))
            logger.info("WandB initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}")
            wandb.init(mode="disabled")

    # Initialize replay buffer as a standard list
    replay_buffer: List[ReplayBufferEntry] = []
    logger.warning("Replay buffer size is infinite. Monitor memory usage.") # Add warning

    logger.info("Starting training loop...")
    for step in range(config.training_steps):
        logger.info(f"--- Step {step}/{config.training_steps} ---")

        # --- Data Selection ---
        use_buffer = (
            step >= config.initial_buffer_fill_steps and
            len(replay_buffer) >= config.batch_size and # Check if buffer has enough samples
            random.random() < config.sample_from_buffer_prob
        )

        if use_buffer:
            sampled_entries = random.sample(replay_buffer, config.batch_size) # Sample from list
            batch_inputs = [entry.original_input for entry in sampled_entries]
            logger.info(f"Sampling {config.batch_size} inputs from replay buffer (current size {len(replay_buffer)}).")
        else:
            batch_inputs = random.sample(qas_data, config.batch_size)
            logger.info(f"Sampling {config.batch_size} inputs from dataset.")

        # Repeat the selected base inputs and apply template
        repeated_inputs = repeat(batch_inputs, config.num_pre_q)
        prompts = [apply_chat_template(tokenizer, item['Q']) for item in repeated_inputs]

        # --- Generation ---
        generated_answers, datas = run_generation_step(prompts, jax_setup, config)

        # --- Reward Calculation ---
        total_rewards_local, rewards_per_func_local = calculate_rewards(
            repeated_inputs, generated_answers, reward_functions, reward_weights
        )
        datas['rewards'] = total_rewards_local

        # --- Replay Buffer Update ---
        update_replay_buffer(
            replay_buffer,
            repeated_inputs,
            prompts,
            generated_answers,
            total_rewards_local,
            rewards_per_func_local,
            reward_functions,
            step,
            config
        )

        # --- Advantage Calculation ---
        rewards_global = process_allgather(total_rewards_local)
        mean_global = rewards_global.mean()
        std_global = max(rewards_global.std(), 1e-6)

        logger.info(f"Step {step}: Local Rewards Mean: {total_rewards_local.mean():.4f}, Global Rewards Mean: {mean_global:.4f}, Std: {std_global:.4f}  {rewards_global.shape=}")

        advantages_local = jax_setup["get_advantages_jit"](
            datas['rewards'], config.num_pre_q, mean_global=mean_global, std_global=std_global
        )
        datas['advantages'] = advantages_local

        # --- PPO Update ---
        datas_jax = jax.tree_util.tree_map(jnp.asarray, datas)
        final_ppo_metadata = perform_ppo_update(jax_setup, datas_jax, config)

        # --- Logging ---
        collect_and_log_metrics(
            step,
            total_rewards_local,
            rewards_per_func_local,
            reward_functions,
            np.asarray(advantages_local),
            datas['labels'],
            final_ppo_metadata,
            config,
            len(replay_buffer) # Pass current buffer size
        )

    logger.info("Training finished.")
    if jax.process_index() == 0 and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    jax.config.update("jax_compilation_cache_dir", "gs://luck-central-2b/jax-cache")
    main()
