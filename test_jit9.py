# main.py
import functools
import os
import random
import logging
from typing import List # Only need List here
import numpy as np
import jax
import jax.numpy as jnp # Keep jax numpy import if needed elsewhere
import wandb
import warnings # Added warnings import
warnings.filterwarnings("ignore") # Added to ignore warnings

# Import necessary components from (assumed) local modules or files
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Tuple, Any, Callable
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
from jax.sharding import NamedSharding
from jax.experimental.multihost_utils import process_allgather

# Assuming these are correctly importable from your project structure
from prompts.prompts import system_prompt # Make sure this path is correct
# Ensure get_advantages is imported from the correct location if training2 is split
from training2 import (reward_correct, reward_format, get_state, training_step,
                       repeat, slice_data as util_slice_data, get_advantages, tag_count_reward, init_fn)
from MLLM_JAX.utils import (get_jax_mesh2, _form_global_array, match_partition_rules,
                            get_partition_rules_llama, ) # Renamed slice_data if conflicting

# %% Configuration & Data Structures (Copied back for single file)
@dataclass
class ReplayBufferEntry:
    """Represents a single entry in the replay buffer."""
    original_input: Dict[str, str]
    prompt_used: str
    generated_answer: str
    total_reward: float
    rewards_per_func: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrainingConfig:
    """Configuration settings for the training script."""
    model_path: str = 'Qwen/Qwen2.5-3B'#'Qwen/Qwen2.5-1.5B-Instruct'
    max_length_sample: int = 1024
    max_length_total: int = max_length_sample + 512
    dataset_name: str = "openai/gsm8k"
    dataset_split: str = "train"
    num_pre_q: int = 16
    batch_size: int = 1
    training_steps: int = 400
    grad_accum_steps: int = 1
    ppo_epochs: int = 2
    mesh_shape_dp: str = "-1,1,1"
    mesh_shape_fsdp: str = "1,-1,1"
    sample_from_buffer_prob: float = 0.0
    initial_buffer_fill_steps: int = 20
    # Advantage calculation alpha (for grpo_clip2)
    advantage_alpha: float = 0.02 # Added alpha for grpo_clip2
    reward_funcs_weights: Dict[str, float] = field(default_factory=dict)
    wandb_project: str = 'grop-gsm8k-2'
    wandb_run_name: str = 'refactored_v5_dynamic_adv' # Updated run name
    log_level: int = logging.INFO

# Setup Logging
logging.basicConfig(level=TrainingConfig.log_level, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# %% Reward Functions Setup (Copied back for single file)
def reward_setup() -> Tuple[List[Callable], List[float]]:
    """Defines the reward functions and their corresponding weights."""
    reward_functions = [reward_correct, reward_format, tag_count_reward]
    reward_weights = [1.0, 0.5, 0.5]
    assert len(reward_functions) == len(reward_weights), "Mismatch between reward functions and weights."
    return reward_functions, reward_weights

# %% Helper Functions (Copied back for single file)
def apply_chat_template(tokenizer: PreTrainedTokenizerBase, conversation_history: List[Dict[str, str]]) -> str:
    """Applies the chat template to a given conversation history."""
    # Ensure the generation prompt is added correctly for the final turn
    return tokenizer.apply_chat_template(
        conversation_history,
        tokenize=False,
        add_generation_prompt=True # This typically adds the prompt for the assistant's next turn
    )

def load_data(config: TrainingConfig) -> List[Dict[str, str]]:
    """Loads and prepares the dataset."""
    try:
        process_count = jax.process_count()
        process_index = jax.process_index()
    except RuntimeError:
        process_count = 1
        process_index = 0

    logger.info(f"Loading dataset: {config.dataset_name}, split: {config.dataset_split}")
    dataset = load_dataset(config.dataset_name, "main", split=config.dataset_split)
    if process_count > 1:
        dataset = dataset.shard(num_shards=process_count, index=process_index)
    qas = [{'Q': x, 'A': y.split('####')[-1].strip()} for x, y in zip(dataset['question'], dataset['answer'])]
    logger.info(f"Loaded {len(qas)} Q/A pairs for process {process_index}.")
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
    # --- MODIFIED JIT for get_advantages ---
    # Store the JITted functions in a dictionary
    get_advantages_jitted_funcs = {
        'grpo_clip2': jax.jit(functools.partial(get_advantages,advantage_estimator='grpo_clip2'), static_argnames=('groups',)),
        'reinforce': jax.jit(functools.partial(get_advantages,advantage_estimator='reinforce'), static_argnames=('groups',)),
    }
    # ---------------------------------------

    train_fn_jit = jax.jit(training_step, donate_argnums=(0,))

    logger.info("JAX setup complete.")
    return {
        "state": state,
        "sampler": sampler,
        "mesh_dp": mesh_dp,
        "params_to_dp": params_to_dp,
        # "get_advantages_jit": get_advantages_jit, # Removed old single JIT function
        "get_advantages_jitted_funcs": get_advantages_jitted_funcs, # Store the dict of JITted functions
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

    prefill_length=process_allgather(input_ids.shape[1]).max()
    prefill_length = sampler.find_ceil(prefill_length)
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

    data = {
        'input_ids': train_input_ids,
        'attention_mask': train_attention_mask,
        'labels': train_completions_mask,
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

def update_replay_buffer(
    replay_buffer: List[ReplayBufferEntry],
    repeated_inputs: List[Dict[str, str]],
    prompts_used: List[str],
    generated_answers: List[str],
    total_rewards: np.ndarray,
    rewards_per_func_values: np.ndarray,
    reward_functions: List[Callable],
    step: int,
    config: TrainingConfig
) -> None:
    """Adds new experiences to the replay buffer (infinite size)."""
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
        replay_buffer.append(entry)
        added_count += 1

    if added_count > 0:
        logger.debug(f"Added {added_count} entries to replay buffer.")
    logger.info(f"Replay buffer size: {len(replay_buffer)}")

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
                lambda x: util_slice_data(x, config.grad_accum_steps, accum_step),
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
    buffer_size: int,
    advantage_estimator_used: str # Add parameter to log which estimator was used
) -> None:
    """Gathers data across hosts, calculates metrics, and logs them."""
    metrics = {}
    advantages_local_np = np.asarray(advantages_local)

    try:
        rewards_global = process_allgather(total_rewards_local)
        advantages_global = process_allgather(advantages_local_np)
        completion_ids_global = process_allgather(completion_ids_local)
        rewards_per_func_gathered = [process_allgather(rewards_per_func_local[i]) for i in range(rewards_per_func_local.shape[0])]
    except Exception as e:
        logger.error(f"Error during process_allgather: {e}. Using local data.", exc_info=True)
        rewards_global = total_rewards_local
        advantages_global = advantages_local_np
        completion_ids_global = completion_ids_local
        rewards_per_func_gathered = [rewards_per_func_local[i] for i in range(rewards_per_func_local.shape[0])]

    mean_global = rewards_global.mean()
    std_global = rewards_global.std()
    metrics['reward/global_mean'] = mean_global
    metrics['reward/global_std'] = std_global
    metrics['replay_buffer/size'] = buffer_size
    metrics['ppo/advantage_estimator'] = advantage_estimator_used # Log the estimator used this step

    reward_func_names = [func.__name__ for func in reward_functions]
    for i, name in enumerate(reward_func_names):
        metrics[f'reward/{name}_mean'] = rewards_per_func_gathered[i].mean()

    metrics['ppo/advantages_max'] = advantages_global.max()
    metrics['ppo/advantages_min'] = advantages_global.min()
    metrics['ppo/advantages_mean'] = advantages_global.mean()

    try:
        correct_idx = reward_func_names.index('reward_correct')
        gathered_correct_rewards = rewards_per_func_gathered[correct_idx]
        if 'reward_correct' in config.reward_funcs_weights and config.reward_funcs_weights['reward_correct'] != 0:
             correct_mask_global = (gathered_correct_rewards / config.reward_funcs_weights['reward_correct']) == 1.0
        else:
             correct_mask_global = gathered_correct_rewards == 1.0
    except (ValueError, KeyError, IndexError):
        logger.warning("Could not find 'reward_correct' for length stats.")
        correct_mask_global = np.zeros(rewards_global.shape, dtype=bool)

    completion_lengths = completion_ids_global.sum(axis=-1)
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

    # Log entropy if available
    current_entropy = np.nan # Default if not found
    if 'entropy' in final_ppo_metadata:
        current_entropy = float(np.asarray(final_ppo_metadata['entropy']))
        metrics['ppo/entropy'] = current_entropy

    if 'entropy_loss' in final_ppo_metadata:
        current_entropy = float(np.asarray(final_ppo_metadata['entropy_loss']))
        metrics['ppo/entropy_loss'] = current_entropy

    if 'per_token_kl' in final_ppo_metadata:
        current_entropy = float(np.asarray(final_ppo_metadata['per_token_kl']))
        metrics['ppo/per_token_kl'] = current_entropy

    # Log to WandB
    if jax.process_index() == 0:
        # Include current entropy in log message
        formatted_metrics = {k: f'{v:.4f}' if isinstance(v, (float, np.floating)) else v for k, v in metrics.items()}
        logger.info(f"Step {step}: Entropy={current_entropy:.4f}, Estimator={advantage_estimator_used}, Metrics: {formatted_metrics}")
        try:
            if wandb.run is not None and wandb.run.mode != "disabled":
                 wandb.log(metrics, step=step)
        except Exception as e:
             logger.error(f"WandB logging failed: {e}")


# %% Main Training Loop (MODIFIED ADVANTAGE CALCULATION)
def main():
    """Main function to run the PPO training loop."""
    # --- Initialization ---
    try:
        jax.distributed.initialize()
        process_count = jax.process_count()
        process_index = jax.process_index()
        logger.info(f"JAX Initialized. Process count: {process_count}, Index: {process_index}")
    except Exception as e:
        process_count = 1
        process_index = 0
        logger.warning(f"Could not initialize JAX distributed: {e}. Running in single-process mode.")
    rng=jax.random.PRNGKey(0)
    config = TrainingConfig()
    reward_functions, reward_weights = reward_setup()
    config.reward_funcs_weights = {func.__name__: weight for func, weight in zip(reward_functions, reward_weights)}

    jax_setup = setup_jax(config)
    tokenizer = jax_setup["tokenizer"]

    qas_data = load_data(config)
    if not qas_data:
         logger.error(f"No data loaded for process {process_index}. Exiting.")
         return

    if process_index == 0:
        try:
            wandb.init(name=config.wandb_run_name, project=config.wandb_project, config=vars(config))
            logger.info("WandB initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}")
            wandb.init(mode="disabled")
    else:
        wandb.init(mode="disabled")

    replay_buffer: List[ReplayBufferEntry] = []
    logger.warning("Replay buffer size is infinite. Monitor memory usage.")

    # --- Initialize last_entropy for dynamic advantage selection ---
    last_entropy = 1.0 # Default value for the first step (ensures grpo_clip2 is used)
    # --------------------------------------------------------------

    # --- Training Loop ---
    logger.info("Starting training loop...")
    counter=30
    for step in range(config.training_steps):
        logger.info(f"--- Step {step}/{config.training_steps} ---")


        # --- Determine Advantage Estimator based on last_entropy ---
        if last_entropy < 0.2  and counter<0 :
            config.batch_size = 16
            config.num_pre_q = 1
        else:
            config.batch_size = 1
            config.num_pre_q = 16




        # 1. Data Selection (Dataset or Replay Buffer)
        key, rng = jax.random.split(rng)
        use_buffer = (
            step >= config.initial_buffer_fill_steps and
            len(replay_buffer) >= config.batch_size and
            jax.random.uniform(key,(1,),dtype=jnp.float32,minval=0,maxval=1) < config.sample_from_buffer_prob
        )


        prompts_for_generation: List[str] = []
        repeated_inputs: List[Dict[str, str]] = []
        base_prompts: List[str] = []  # Store base prompts before repeating
        batch_inputs: List[Dict[str, str]] = []  # Store base inputs before repeating

        # --- MODIFIED: Construct history differently based on source ---
        if use_buffer:
            sampled_entries = random.sample(replay_buffer, config.batch_size)
            logger.info(f"Sampling {config.batch_size} entries from replay buffer (size {len(replay_buffer)}) for continuation.")
            follow_up_text = "I think this maybe correct." # Define the follow-up

            for entry in sampled_entries:
                # Construct the 4-turn history for replay buffer entries
                history = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": entry.original_input['Q']},
                    {"role": "assistant", "content": entry.generated_answer},
                    {"role": "user", "content": follow_up_text}
                    # The add_generation_prompt=True in apply_chat_template
                    # should prompt the model for the next assistant turn
                ]
                # Apply the template to this specific history
                base_prompts.append(apply_chat_template(tokenizer, history))
                batch_inputs.append(entry.original_input)

            # Repeat the generated base prompts and corresponding inputs
            prompts_for_generation = repeat(base_prompts, config.num_pre_q)
            repeated_inputs = repeat(batch_inputs, config.num_pre_q)

        else:
            batch_inputs_base = random.sample(qas_data, config.batch_size)
            logger.info(f"Sampling {config.batch_size} inputs from dataset.")

            for item in batch_inputs_base:
                # Construct the standard 2-turn history for dataset entries
                history = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": item['Q']}
                ]
                # Apply the template to this specific history
                base_prompts.append(apply_chat_template(tokenizer, history))
                batch_inputs.append(item) # Store the original item

            # Repeat the generated base prompts and corresponding inputs
            prompts_for_generation = repeat(base_prompts, config.num_pre_q)
            repeated_inputs = repeat(batch_inputs, config.num_pre_q)
        # ----------------------------------------------------------

        # 2. Generate Answers
        generated_answers, datas = run_generation_step(prompts_for_generation, jax_setup, config)

        # 3. Calculate Rewards
        total_rewards_local, rewards_per_func_local = calculate_rewards(
            repeated_inputs,
            generated_answers,
            reward_functions,
            reward_weights
        )
        datas['rewards'] = total_rewards_local

        # 4. Update Replay Buffer (Only if sampling from dataset originally)
        # --- MODIFIED: Always update buffer based on current logic ---
        if not use_buffer: # Original logic - only update if from dataset
            update_replay_buffer(
                replay_buffer,
                repeated_inputs,
                prompts_for_generation,
                generated_answers,
                total_rewards_local,
                rewards_per_func_local,
                reward_functions,
                step,
                config
            )
        # ----------------------------------------------------------

        # 5. Calculate Advantages (Dynamic Selection)
        try:
            rewards_global = process_allgather(total_rewards_local)
        except Exception as e:
             logger.error(f"Error during process_allgather for advantages: {e}. Using local rewards.", exc_info=True)
             rewards_global = total_rewards_local

        mean_global = rewards_global.mean()
        std_global = max(rewards_global.std(), 1e-6)
        logger.info(f"Step {step}: Local Rewards Mean: {total_rewards_local.mean():.4f}, Global Rewards Mean: {mean_global:.4f}, Std: {std_global:.4f}")

        # --- Determine Advantage Estimator based on last_entropy ---
        # if last_entropy > 0.2  :
        #     advantage_estimator = 'grpo_clip2'
        # else:
        #     advantage_estimator = 'reinforce'

        if last_entropy < 0.2  and counter<0 :
            advantage_estimator = 'reinforce'

        else:
            advantage_estimator = 'grpo_clip2'
            if counter<0 and last_entropy>=0.2:
                counter=5
            counter -= 1



        logger.info(f"Using advantage estimator: {advantage_estimator} (based on last entropy: {last_entropy:.4f})")
        # ---------------------------------------------------------

        # Call the JIT'd function with the selected estimator and necessary args
        advantages_local = jax_setup["get_advantages_jitted_funcs"][advantage_estimator](
            rewards=datas['rewards'], # Pass rewards array
            groups=config.num_pre_q, # Pass groups (static)
            alpha=config.advantage_alpha, # Pass alpha from config
            mean_global=mean_global, # Pass global mean
            std_global=std_global # Pass global std
        )
        datas['advantages'] = advantages_local # Add advantages (JAX array)

        # 6. Perform PPO Update
        datas_jax = jax.tree_util.tree_map(
             lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x,
             datas
        )
        final_ppo_metadata = perform_ppo_update(jax_setup, datas_jax, config)

        # --- Update last_entropy for the next step ---
        if 'entropy' in final_ppo_metadata:
            last_entropy = float(np.asarray(final_ppo_metadata['entropy']))
            logger.info(f"Updated last_entropy to: {last_entropy:.4f}")
        else:
            # Keep the previous value if entropy wasn't returned
            logger.warning(f"Entropy not found in PPO metadata for step {step}. Using previous value: {last_entropy:.4f}")
        # -------------------------------------------

        # 7. Log Metrics
        collect_and_log_metrics(
            step,
            total_rewards_local,
            rewards_per_func_local,
            reward_functions,
            advantages_local, # Pass JAX array or np array
            datas['labels'],
            final_ppo_metadata,
            config,
            len(replay_buffer),
            advantage_estimator # Pass the estimator used this step for logging
        )

    # --- End of Training ---
    logger.info("Training finished.")
    if process_index == 0 and wandb.run is not None and wandb.run.mode != "disabled":
        wandb.finish()
        logger.info("WandB run finished.")

if __name__ == "__main__":
    # Ensure cache dir exists or is accessible if using it
    # jax.config.update("jax_compilation_cache_dir", "gs://luck-central-2b/jax-cache") # Example GCS path
    main()
