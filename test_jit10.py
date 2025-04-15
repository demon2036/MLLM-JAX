# main.py
import functools
import os
import random
import logging
from typing import List, Dict, Tuple, Any, Callable # Adjusted imports
import numpy as np
import jax
import jax.numpy as jnp
import wandb
import warnings
warnings.filterwarnings("ignore")

# Import necessary components
from dataclasses import dataclass, field
from functools import partial
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
from jax.sharding import NamedSharding
from jax.experimental.multihost_utils import process_allgather

# Assuming these are correctly importable
from prompts.prompts import system_prompt
# Ensure correct import path if training2 is split
from training2 import (reward_correct, reward_format, get_state, training_step,
                       repeat, slice_data as util_slice_data, get_advantages, tag_count_reward, init_fn)
from MLLM_JAX.utils import (get_jax_mesh2, _form_global_array, match_partition_rules,
                            get_partition_rules_llama, )

# %% Configuration & Data Structures
@dataclass
class ReplayBufferEntry:
    """Represents a single entry in the replay buffer."""
    original_input: Dict[str, str] # Stores the original {Q: ..., A: ...} pair
    prompt_used: str # Stores the actual prompt fed to the model (can be Q or Q+TruncatedA)
    generated_answer: str # Stores the model's output (full answer or completion)
    total_reward: float
    rewards_per_func: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Add field to store the full original answer if needed for reconstruction context
    full_original_answer_for_reward: str = "" # Store the full answer context for reward calc if entry is from completion

@dataclass
class TrainingConfig:
    """Configuration settings for the training script."""
    model_path: str = 'Qwen/Qwen2.5-1.5B-Instruct'
    max_length_sample: int = 1024
    max_length_total: int = max_length_sample + 512 # Ensure this is large enough
    dataset_name: str = "openai/gsm8k"
    dataset_split: str = "train"
    num_pre_q: int = 8 # Set to 8 as requested
    batch_size: int = 1
    training_steps: int = 400
    grad_accum_steps: int = 1
    ppo_epochs: int = 2
    mesh_shape_dp: str = "-1,1,1"
    mesh_shape_fsdp: str = "1,-1,1"
    sample_from_buffer_prob: float = 1.0 # Example probability to sample from buffer
    initial_buffer_fill_steps: int = 2
    advantage_alpha: float = 0.02
    reward_funcs_weights: Dict[str, float] = field(default_factory=dict)
    wandb_project: str = 'grop-gsm8k-2'
    wandb_run_name: str = 'refactored_v6_completion_task' # Updated run name
    log_level: int = logging.INFO
    # Completion task specific settings
    completion_trunc_min_perc: float = 0.3
    completion_trunc_max_perc: float = 0.5


# Setup Logging
logging.basicConfig(level=TrainingConfig.log_level, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# %% Reward Functions Setup
def reward_setup() -> Tuple[List[Callable], List[float]]:
    """Defines the reward functions and their corresponding weights."""
    reward_functions = [reward_correct, reward_format, tag_count_reward]
    reward_weights = [1.0, 0.5, 0.5] # Keep weights as before
    assert len(reward_functions) == len(reward_weights), "Mismatch between reward functions and weights."
    return reward_functions, reward_weights

# %% Helper Functions
def apply_chat_template(tokenizer: PreTrainedTokenizerBase, conversation_history: List[Dict[str, str]]) -> str:
    """Applies the chat template to a given conversation history."""
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
    # Store both Q and the full A for potential use in reward checking
    qas = [{'Q': x, 'A': y, 'A_final': y.split('####')[-1].strip()} for x, y in zip(dataset['question'], dataset['answer'])]
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
        num_pre_q=config.num_pre_q, # Pass num_pre_q if needed by get_state
        max_lengths=config.max_length_total
    )

    params_shapes = jax.eval_shape(init_fn, state.params)
    params_partition = match_partition_rules(get_partition_rules_llama(), params_shapes)
    params_sharding_dp = jax.tree_util.tree_map(lambda x: NamedSharding(mesh_dp, x), params_partition)
    params_sharding_fsdp = jax.tree_util.tree_map(lambda x: NamedSharding(mesh_fsdp, x), params_partition)

    params_to_dp = jax.jit(init_fn, out_shardings=params_sharding_dp)
    # params_to_fsdp = jax.jit(init_fn, out_shardings=params_sharding_fsdp) # If needed

    get_advantages_jitted_funcs = {
        'grpo_clip2': jax.jit(functools.partial(get_advantages,advantage_estimator='grpo_clip2'), static_argnames=('groups',)),
        'grpo': jax.jit(functools.partial(get_advantages,advantage_estimator='grpo'), static_argnames=('groups',)),
    }

    train_fn_jit = jax.jit(training_step, donate_argnums=(0,))

    logger.info("JAX setup complete.")
    return {
        "state": state,
        "sampler": sampler,
        "mesh_dp": mesh_dp,
        "params_to_dp": params_to_dp,
        "get_advantages_jitted_funcs": get_advantages_jitted_funcs,
        "train_fn_jit": train_fn_jit,
        "tokenizer": sampler.tokenizer # Expose tokenizer
    }

# %% Core Logic Functions (run_generation_step - check labels mask)
def run_generation_step(
    prompts: List[str],
    jax_setup: Dict[str, Any],
    config: TrainingConfig
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Generates answers (completions) using the model. Returns generated text and data dict."""
    sampler = jax_setup["sampler"]
    state = jax_setup["state"]
    params_to_dp = jax_setup["params_to_dp"]
    tokenizer = jax_setup["tokenizer"]

    params_dp_bf16 = params_to_dp(jax.tree_util.tree_map(lambda x: jnp.astype(x, jnp.bfloat16), state.params))

    # Tokenize prompts (which might be Q or Q+TruncatedA)
    inputs = tokenizer(prompts, return_tensors="jax", padding=True, padding_side="right", truncation=True, max_length=config.max_length_total - config.max_length_sample) # Truncate prompt if too long
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    position_ids = attention_mask.cumsum(-1) - 1
    position_ids = jnp.where(attention_mask == 0, 1, position_ids)

    # Determine true length of the prompts *before* padding
    # Use original untruncated inputs length for true_length_prompts
    unpadded_inputs = tokenizer(prompts, return_tensors="np", padding=False)
    true_length_prompts = np.array([len(ids) for ids in unpadded_inputs['input_ids']])


    # Determine prefill length based on max prompt length in the batch
    max_prompt_len_in_batch = input_ids.shape[1]
    prefill_length = sampler.find_ceil(max_prompt_len_in_batch)

    # Pad inputs to prefill_length if necessary (should already be handled by tokenizer padding='max_length' or similar if max_length is set, but manual check is safer)
    pad_width = max(0, prefill_length - input_ids.shape[1])
    if pad_width > 0:
         input_ids = jnp.pad(input_ids, ((0, 0), (0, pad_width)), constant_values=tokenizer.pad_token_id)
         attention_mask = jnp.pad(attention_mask, ((0, 0), (0, pad_width)), constant_values=0)
         position_ids = jnp.pad(position_ids, ((0, 0), (0, pad_width)), constant_values=1)


    logger.info(f"Generating completions for {len(prompts)} prompts (max prompt len: {max_prompt_len_in_batch}, prefill_len: {prefill_length})...")

    outputs = sampler.generate(
        input_ids_pad=input_ids,
        pad_attention=attention_mask,
        position_ids=position_ids,
        prefill_length=prefill_length,
        max_length=config.max_length_sample, # Max *new* tokens to generate
        params=params_dp_bf16
    )
    logger.info("Generation complete.")

    # --- Prepare data for PPO update ---
    # total_len includes prompt + generated part (up to max_length_total)
    total_len = min(prefill_length + config.max_length_sample, config.max_length_total)
    train_input_ids = np.full((len(prompts), total_len), fill_value=tokenizer.pad_token_id, dtype=np.int32)
    train_attention_mask = np.zeros_like(train_input_ids, dtype=np.int32)
    # *** CRUCIAL: labels mask should ONLY mark the generated completion tokens ***
    train_completion_labels = np.full_like(train_input_ids, fill_value=-100, dtype=np.int32) # Use -100 for ignored tokens
    generated_completions_text = [] # Store only the generated part

    for i, (true_len, gen_step) in enumerate(zip(true_length_prompts, outputs['local_sample_step'])):
        # true_len is the length of the original prompt (Q or Q+TruncatedA)
        # start_idx is where generation begins in the model's internal buffer
        start_idx_buffer = prefill_length # Generation starts after prefill
        end_idx_buffer = start_idx_buffer + gen_step + 1 # End of generation in buffer
        start_idx_buffer = min(start_idx_buffer, outputs['local_token_buffer'].shape[1])
        end_idx_buffer = min(end_idx_buffer, outputs['local_token_buffer'].shape[1])

        # Extract generated tokens from the buffer
        generated_tokens = outputs['local_token_buffer'][i, start_idx_buffer:end_idx_buffer]
        completion_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        generated_completions_text.append(completion_text)

        # Fill the training arrays
        current_prompt_tokens = input_ids[i, :true_len] # Get the actual prompt tokens used (padded to max_prompt_len_in_batch)
        actual_prompt_len = min(true_len, total_len) # Ensure prompt doesn't exceed total length

        # Copy prompt
        train_input_ids[i, :actual_prompt_len] = current_prompt_tokens[:actual_prompt_len]
        train_attention_mask[i, :actual_prompt_len] = 1
        # Prompt part has label -100 (ignored in loss)

        # Copy generated completion and set labels
        gen_len = len(generated_tokens)
        copy_len = min(gen_len, total_len - actual_prompt_len) # How many generated tokens fit
        if copy_len > 0:
            start_gen_idx_train = actual_prompt_len
            end_gen_idx_train = actual_prompt_len + copy_len
            train_input_ids[i, start_gen_idx_train:end_gen_idx_train] = generated_tokens[:copy_len]
            train_attention_mask[i, start_gen_idx_train:end_gen_idx_train] = 1
            # *** Set labels ONLY for the generated part ***
            train_completion_labels[i, start_gen_idx_train:end_gen_idx_train] = generated_tokens[:copy_len]

    logger.info(f"Sample generated completions (last 2): {generated_completions_text[-2:]}")

    # Note: 'labels' now correctly marks only the completion tokens with their IDs,
    # and uses -100 elsewhere. The training_step loss function should handle this.
    data = {
        'input_ids': train_input_ids,
        'attention_mask': train_attention_mask,
        'labels': train_completion_labels, # This is the crucial mask for PPO loss
    }

    # Return the generated completions (text) and the prepared data dictionary
    return generated_completions_text, data


# %% Calculate Rewards (MODIFIED to handle reconstruction)
def calculate_rewards(
    repeated_original_inputs: List[Dict[str, str]], # Original {Q:A} pairs
    truncated_prompts: List[str], # The actual prompts used (Q+TruncA)
    generated_completions: List[str], # Model outputs (the completions)
    reward_functions: List[Callable],
    reward_weights: List[float],
    tokenizer: PreTrainedTokenizerBase # Needed to find prompt length
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates rewards based on reconstructed answers."""
    num_answers = len(generated_completions)
    num_funcs = len(reward_functions)
    rewards_per_func = np.zeros((num_funcs, num_answers))
    reconstructed_answers = []

    # Reconstruct the full answer for reward calculation
    for i in range(num_answers):
        # Find the length of the assistant's part in the prompt (truncated answer)
        # This assumes the last turn in the prompt template is the assistant
        prompt_tokens = tokenizer(truncated_prompts[i], return_tensors="np").input_ids[0]
        # This is complex if template adds special tokens. A simpler, though less robust,
        # way might be to find the truncated answer string *before* templating.
        # Let's assume reconstruction based on combining known prefix and generation for now.
        # A better way: Store the truncated_answer_str alongside the prompt.
        # For now, we approximate by finding the end of the prompt.
        # We need the original truncated answer string here. This requires passing it.
        # ---- MODIFICATION NEEDED in main loop to pass truncated_answer_str ----
        # ---- Assuming it's passed via repeated_original_inputs for now (HACK) ----
        truncated_answer_str = repeated_original_inputs[i].get("truncated_answer_for_reward", "") # Needs to be added in main
        reconstructed = truncated_answer_str + generated_completions[i]
        reconstructed_answers.append(reconstructed)

    # Calculate rewards using the reconstructed answer
    for i, reward_func in enumerate(reward_functions):
        func_weight = reward_weights[i]
        func_name = reward_func.__name__
        for j, (orig_inp, recon_ans) in enumerate(zip(repeated_original_inputs, reconstructed_answers)):
            try:
                # Pass the original input context and the RECONSTRUCTED answer
                reward_val = reward_func(orig_inp, recon_ans)
                rewards_per_func[i, j] = func_weight * reward_val
            except Exception as e:
                logger.error(f"Error calculating reward '{func_name}' for sample {j} with reconstructed answer: {e}", exc_info=False)
                rewards_per_func[i, j] = -1.0 # Assign penalty for error

    total_rewards = rewards_per_func.sum(axis=0)
    return total_rewards, rewards_per_func


# %% Update Replay Buffer (MODIFIED for completion task results)
def update_replay_buffer(
    replay_buffer: List[ReplayBufferEntry],
    # Information about the generation that just happened
    original_inputs_for_batch: List[Dict[str, str]], # Base {Q:A} used for this batch
    prompts_actually_used: List[str], # The Q or Q+TruncA prompts
    generated_outputs: List[str], # Full answers OR completions
    total_rewards: np.ndarray, # Rewards for the generated_outputs
    rewards_per_func_values: np.ndarray, # Per-func rewards
    reward_functions: List[Callable],
    step: int,
    config: TrainingConfig,
    was_completion_task: bool, # Flag to know the source
    full_original_answers_for_reward: List[str] = None # Pass the full answers if completion task
) -> None:
    """Adds new experiences to the replay buffer (infinite size)."""
    num_new_entries = len(generated_outputs)
    added_count = 0
    reward_func_names = [func.__name__ for func in reward_functions]

    for i in range(num_new_entries):
        rewards_dict = {name: rewards_per_func_values[k, i] for k, name in enumerate(reward_func_names)}

        # Determine what to store based on whether it was a completion task
        input_to_store = original_inputs_for_batch[i] # The base {Q: A} pair
        prompt_to_store = prompts_actually_used[i]
        output_to_store = generated_outputs[i]
        full_answer_context = ""
        if was_completion_task:
            # For completion tasks, store the completion as generated_answer
            # Store the prompt used (Q+TruncA)
            # Store the reward (calculated on reconstructed answer)
            # Store the *full* original answer for potential future reconstruction needs
             if full_original_answers_for_reward:
                  full_answer_context = full_original_answers_for_reward[i]
             else:
                  # Fallback to original input if list wasn't passed (should be fixed)
                  full_answer_context = input_to_store['A']

        else:
            # For initial dataset tasks, store the full answer generated
             full_answer_context = output_to_store # The full answer is the context itself


        entry = ReplayBufferEntry(
            original_input=input_to_store,
            prompt_used=prompt_to_store,
            generated_answer=output_to_store, # Stores completion if was_completion_task=True
            total_reward=total_rewards[i],
            rewards_per_func=rewards_dict,
            metadata={'step_added': step, 'was_completion': was_completion_task},
            full_original_answer_for_reward=full_answer_context # Store context
        )
        replay_buffer.append(entry)
        added_count += 1

    if added_count > 0:
        logger.debug(f"Added {added_count} entries to replay buffer.")
    logger.info(f"Replay buffer size: {len(replay_buffer)}")


# %% Perform PPO Update (Unchanged, relies on correct 'labels' mask)
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

    # Calculate total valid *completion* tokens (where labels != -100)
    total_valid_token_count = (datas['labels'] != -100).sum()


    per_token_logps_list = []
    final_meta_data = {}

    logger.info(f"Starting PPO updates ({config.ppo_epochs} epochs) on {total_valid_token_count} completion tokens...")
    for ppo_epoch in range(config.ppo_epochs):
        datas_copy = datas.copy() # Create copy for ppo epochs if needed
        for accum_step in range(config.grad_accum_steps):
            local_data = jax.tree_util.tree_map(
                lambda x: util_slice_data(x, config.grad_accum_steps, accum_step),
                datas_copy # Use the copy inside accumulation loop
            )
            # Pass total count if needed by training_step, otherwise it uses local mask sum
            # local_data['total_valid_token_count'] = total_valid_token_count

            state, meta_data = train_fn_jit(state, local_data)
            final_meta_data = meta_data # Keep track of last metadata

            # Store logps from first PPO epoch for subsequent epochs
            if ppo_epoch == 0 and 'per_token_logps' in meta_data:
                 per_token_logps_list.append(meta_data['per_token_logps'])

        # After first PPO epoch, add 'old_per_token_logps' for subsequent epochs
        if ppo_epoch == 0 and per_token_logps_list:
            # Ensure concatenation happens correctly even with accumulation
            full_batch_logps = jnp.concatenate(per_token_logps_list, axis=0)
            # Reshape if necessary to match original batch structure before accumulation split
            # This assumes the batch dim was split by grad_accum_steps
            original_batch_size = datas['input_ids'].shape[0] // config.num_pre_q // config.grad_accum_steps
            logps_reshaped = full_batch_logps.reshape(config.grad_accum_steps, original_batch_size * config.num_pre_q, -1)
            # We need to interleave the results - this logic might be complex.
            # Simpler: Assume train_fn_jit returns logps matching local_data batch size
            # And we just need to pass the full batch logps for the *next* epoch.
            datas['old_per_token_logps'] = jnp.concatenate(per_token_logps_list, axis=0) # Pass concatenated logps
            logger.info(f"Stored old log probabilities shape: {datas['old_per_token_logps'].shape}")


    logger.info("PPO updates finished.")
    jax_setup["state"] = state # Update state in the setup dict
    return final_meta_data

# %% Collect and Log Metrics (Adjusted completion length calculation)
def collect_and_log_metrics(
    step: int,
    total_rewards_local: np.ndarray,
    rewards_per_func_local: np.ndarray,
    reward_functions: List[Callable],
    advantages_local: np.ndarray,
    # Pass the 'labels' mask which indicates completion tokens
    completion_labels_local: np.ndarray,
    final_ppo_metadata: Dict[str, Any],
    config: TrainingConfig,
    buffer_size: int,
    advantage_estimator_used: str,
    was_completion_task: bool # Log if it was completion
) -> None:
    """Gathers data across hosts, calculates metrics, and logs them."""
    metrics = {}
    advantages_local_np = np.asarray(advantages_local)

    try:
        # Gather necessary data across hosts
        rewards_global = process_allgather(total_rewards_local)
        advantages_global = process_allgather(advantages_local_np)
        # Gather the labels mask to calculate global completion length
        completion_labels_global = process_allgather(completion_labels_local)
        rewards_per_func_gathered = [process_allgather(rewards_per_func_local[i]) for i in range(rewards_per_func_local.shape[0])]
    except Exception as e:
        logger.error(f"Error during process_allgather: {e}. Using local data.", exc_info=True)
        # Fallback to local data if allgather fails
        rewards_global = total_rewards_local
        advantages_global = advantages_local_np
        completion_labels_global = completion_labels_local
        rewards_per_func_gathered = [rewards_per_func_local[i] for i in range(rewards_per_func_local.shape[0])]

    # Basic metrics
    mean_global = rewards_global.mean()
    std_global = rewards_global.std()
    metrics['reward/global_mean'] = mean_global
    metrics['reward/global_std'] = std_global
    metrics['replay_buffer/size'] = buffer_size
    metrics['ppo/advantage_estimator'] = advantage_estimator_used
    metrics['step_info/was_completion_task'] = was_completion_task

    # Per-function reward metrics
    reward_func_names = [func.__name__ for func in reward_functions]
    for i, name in enumerate(reward_func_names):
        metrics[f'reward/{name}_mean'] = rewards_per_func_gathered[i].mean()

    # Advantage metrics
    metrics['ppo/advantages_max'] = advantages_global.max()
    metrics['ppo/advantages_min'] = advantages_global.min()
    metrics['ppo/advantages_mean'] = advantages_global.mean()

    # Completion length metrics (based on the labels mask)
    # Calculate length by summing non -100 labels per sequence
    completion_lengths = (completion_labels_global != -100).sum(axis=-1)
    metrics['completion/length_mean'] = completion_lengths.mean()
    metrics['completion/length_max'] = completion_lengths.max()
    metrics['completion/length_min'] = completion_lengths.min()


    # --- Correctness-based length stats (Requires careful handling of reward scaling) ---
    try:
        correct_idx = reward_func_names.index('reward_correct')
        gathered_correct_rewards = rewards_per_func_gathered[correct_idx]

        # Determine the threshold for correctness based on the weight
        correct_weight = config.reward_funcs_weights.get('reward_correct', 1.0)
        # Handle potential division by zero if weight is 0
        if correct_weight != 0:
            correct_threshold = 1.0 * correct_weight
            # Use a small tolerance for floating point comparisons
            correct_mask_global = np.abs(gathered_correct_rewards - correct_threshold) < 1e-5
        else:
             # If weight is 0, correctness might be judged differently or unavailable
             correct_mask_global = np.zeros(rewards_global.shape, dtype=bool)
             logger.warning("Reward weight for 'reward_correct' is 0. Cannot determine correctness mask reliably.")

    except (ValueError, KeyError, IndexError):
        logger.warning("Could not find 'reward_correct' or its weight for length stats.")
        correct_mask_global = np.zeros(rewards_global.shape, dtype=bool)

    # Log length stats based on correctness mask
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
    # ------------------------------------------------------------------------------------

    # Log PPO metadata (entropy, etc.)
    current_entropy = np.nan # Default if not found
    if 'batch_avg_entropy' in final_ppo_metadata: # Use the batch average entropy
        current_entropy = float(np.asarray(final_ppo_metadata['batch_avg_entropy']))
        metrics['ppo/batch_avg_entropy'] = current_entropy

    # Log other PPO metrics if needed (e.g., the conditionally calculated losses)
    if 'loss_low_entropy_calc' in final_ppo_metadata:
         metrics['ppo/loss_if_low_entropy'] = float(np.asarray(final_ppo_metadata['loss_low_entropy_calc']))
    if 'loss_high_entropy_calc' in final_ppo_metadata:
         metrics['ppo/loss_if_high_entropy'] = float(np.asarray(final_ppo_metadata['loss_high_entropy_calc']))


    # Log to WandB (only on process 0)
    if jax.process_index() == 0:
        formatted_metrics = {k: f'{v:.4f}' if isinstance(v, (float, np.floating)) else v for k, v in metrics.items()}
        logger.info(f"Step {step}: Entropy={current_entropy:.4f}, Estimator={advantage_estimator_used}, CompletionTask={was_completion_task}, Metrics: {formatted_metrics}")
        try:
            if wandb.run is not None and wandb.run.mode != "disabled":
                 wandb.log(metrics, step=step)
        except Exception as e:
             logger.error(f"WandB logging failed: {e}")


# %% Main Training Loop (MODIFIED for Completion Task)
def main():
    """Main function to run the PPO training loop."""
    try:
        jax.distributed.initialize()
        process_count = jax.process_count()
        process_index = jax.process_index()
        logger.info(f"JAX Initialized. Process count: {process_count}, Index: {process_index}")
    except Exception as e:
        process_count = 1
        process_index = 0
        logger.warning(f"Could not initialize JAX distributed: {e}. Running in single-process mode.")

    rng = jax.random.PRNGKey(42) # Use a fixed seed for reproducibility if needed
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

    last_entropy = 1.0 # Default

    logger.info("Starting training loop...")
    for step in range(config.training_steps):
        logger.info(f"--- Step {step}/{config.training_steps} ---")

        step_rng, rng = jax.random.split(rng) # Get a unique key for this step
        use_buffer_rng, generation_rng = jax.random.split(step_rng)

        use_buffer = (
            step >= config.initial_buffer_fill_steps and
            len(replay_buffer) >= config.batch_size and
            jax.random.uniform(use_buffer_rng, (1,), dtype=jnp.float32, minval=0, maxval=1)[0] < config.sample_from_buffer_prob
        )
        was_completion_task = bool(use_buffer) # Flag for logging/buffer update

        prompts_for_generation: List[str] = []
        # Store original {Q:A} pairs associated with each prompt for reward calc
        original_inputs_for_reward: List[Dict[str, str]] = []
        # Store the full original answer if doing completion task, needed for reward calc
        full_original_answers_for_reward: List[str] = []
        # Store the truncated answer string used in the prompt for reconstruction
        truncated_answers_for_reconstruction: List[str] = []


        if use_buffer:
            # --- Completion Task Logic ---
            sampled_entries = random.sample(replay_buffer, config.batch_size)
            logger.info(f"Sampling {config.batch_size} entries from replay buffer (size {len(replay_buffer)}) for COMPLETION task.")

            for entry in sampled_entries:
                original_question = entry.original_input['Q']
                 # Use the full answer stored in the entry for truncation base
                original_full_answer = entry.full_original_answer_for_reward
                if not original_full_answer: # Fallback if not stored correctly
                    original_full_answer = entry.generated_answer # Might be completion itself

                # Tokenize the full original answer ONCE
                try:
                    answer_tokens = tokenizer(original_full_answer, add_special_tokens=False).input_ids
                except Exception as e:
                    logger.error(f"Tokenization error for answer: {original_full_answer[:50]}... Error: {e}")
                    answer_tokens = [] # Skip this entry if tokenization fails

                if not answer_tokens:
                     logger.warning(f"Skipping entry due to empty/failed tokenization of answer: {entry.original_input['Q']}")
                     continue # Skip this entry if answer is empty or failed

                # Repeat N times with random truncation for the *same* original entry
                for _ in range(config.num_pre_q):
                    answer_len_chars = len(original_full_answer)
                    # 1. 直接抽取一个 0.3 到 0.5 之间的随机小数比例
                    trunc_fraction = random.uniform(config.completion_trunc_min_perc, config.completion_trunc_max_perc)
                    # 2. 根据比例计算截断的字符数 (取整)
                    trunc_char_len = int(answer_len_chars * trunc_fraction)
                    # 3. 直接进行字符串切片
                    truncated_answer_str = original_full_answer[:trunc_char_len]  # Direct string slicing
                    # Construct prompt history (System, User Q, Assistant Truncated A)
                    history = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": original_question},
                        {"role": "assistant", "content": truncated_answer_str}
                    ]
                    prompt = apply_chat_template(tokenizer, history)
                    prompts_for_generation.append(prompt)
                    # Store original Q/A for reward context
                    original_inputs_for_reward.append(entry.original_input)
                    # Store the full original answer for reward calculation context
                    full_original_answers_for_reward.append(original_full_answer)
                    # Store the specific truncated answer used for this prompt
                    truncated_answers_for_reconstruction.append(truncated_answer_str)

        else:
            # --- Standard Generation Task Logic ---
            batch_inputs_base = random.sample(qas_data, config.batch_size)
            logger.info(f"Sampling {config.batch_size} inputs from dataset for standard generation.")

            for item in batch_inputs_base:
                history = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": item['Q']}
                ]
                base_prompt = apply_chat_template(tokenizer, history)
                # Repeat N times for the same base prompt
                for _ in range(config.num_pre_q):
                    prompts_for_generation.append(base_prompt)
                    original_inputs_for_reward.append(item) # Store original Q/A
                    # No truncation needed here for reconstruction list
                    truncated_answers_for_reconstruction.append("") # Placeholder
                    full_original_answers_for_reward.append(item['A']) # Store full answer context

        # Check if any prompts were generated before proceeding
        if not prompts_for_generation:
             logger.warning(f"Step {step}: No prompts generated (possibly due to empty buffer or data sampling issues). Skipping step.")
             continue

        # 2. Generate Answers / Completions
        generated_outputs_text, datas = run_generation_step(prompts_for_generation, jax_setup, config)

        # 3. Calculate Rewards (Needs reconstructed answers if completion task)
        # --- Pass necessary info to calculate_rewards ---
        # HACK: Modify original_inputs_for_reward temporarily to pass truncated string
        temp_inputs_for_reward = []
        for i in range(len(original_inputs_for_reward)):
             temp_inp = original_inputs_for_reward[i].copy()
             if was_completion_task:
                  temp_inp["truncated_answer_for_reward"] = truncated_answers_for_reconstruction[i]
             else:
                  temp_inp["truncated_answer_for_reward"] = "" # Not needed
             temp_inputs_for_reward.append(temp_inp)

        total_rewards_local, rewards_per_func_local = calculate_rewards(
            temp_inputs_for_reward, # Pass modified dict with truncated answer
            prompts_for_generation, # Pass the actual prompts used
            generated_outputs_text, # Pass the model's raw output (completions or full answers)
            reward_functions,
            reward_weights,
            tokenizer
        )
        datas['rewards'] = total_rewards_local # Add rewards to data dict for advantage calculation

        # 4. Update Replay Buffer
        # We store the experience generated in this step, regardless of source
        update_replay_buffer(
            replay_buffer,
            original_inputs_for_reward, # Pass the *base* {Q:A} pairs
            prompts_for_generation, # Pass the actual prompts used (Q or Q+TruncA)
            generated_outputs_text, # Pass the raw model output (completion or full answer)
            total_rewards_local,
            rewards_per_func_local,
            reward_functions,
            step,
            config,
            was_completion_task=was_completion_task, # Indicate if it was completion
            full_original_answers_for_reward=full_original_answers_for_reward if was_completion_task else None # Pass full answers only if completion
        )

        # 5. Calculate Advantages (Dynamic Selection)
        try:
            # Use rewards from the current step for advantage calculation
            rewards_to_gather = datas['rewards']
            rewards_global = process_allgather(rewards_to_gather)
        except Exception as e:
             logger.error(f"Error during process_allgather for advantages: {e}. Using local rewards.", exc_info=True)
             rewards_global = datas['rewards'] # Use local as fallback

        mean_global = rewards_global.mean()
        std_global = max(rewards_global.std(), 1e-6)
        logger.info(f"Step {step}: Local Rewards Mean: {np.mean(datas['rewards']):.4f}, Global Rewards Mean: {mean_global:.4f}, Std: {std_global:.4f}")

        # Determine Advantage Estimator based on last_entropy
        # --- Keep previous dynamic logic ---
        if last_entropy > 0.4 and step > 300 : # Example threshold
            advantage_estimator = 'grpo_clip2'
        else:
            advantage_estimator = 'grpo'
        # advantage_estimator = 'grpo' # Keep override if needed for testing
        logger.info(f"Using advantage estimator: {advantage_estimator} (based on last entropy: {last_entropy:.4f})")
        # ----------------------------------

        advantages_local = jax_setup["get_advantages_jitted_funcs"][advantage_estimator](
            rewards=datas['rewards'], # Use rewards from current step data
            groups=config.num_pre_q,
            alpha=config.advantage_alpha,
            mean_global=mean_global,
            std_global=std_global
        )
        datas['advantages'] = advantages_local # Add advantages (JAX array)

        # 6. Perform PPO Update
        datas_jax = jax.tree_util.tree_map(
             lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x,
             datas # Use the data dict constructed in this step
        )
        final_ppo_metadata = perform_ppo_update(jax_setup, datas_jax, config)

        # 7. Update last_entropy for the next step
        if 'batch_avg_entropy' in final_ppo_metadata: # Check the key returned by collect_metrics
            last_entropy = float(np.asarray(final_ppo_metadata['batch_avg_entropy']))
            logger.info(f"Updated last_entropy to: {last_entropy:.4f}")
        elif 'entropy' in final_ppo_metadata: # Fallback check for older key
             last_entropy = float(np.asarray(final_ppo_metadata['entropy']))
             logger.info(f"Updated last_entropy using 'entropy' key to: {last_entropy:.4f}")
        else:
            logger.warning(f"Entropy not found in PPO metadata for step {step}. Using previous value: {last_entropy:.4f}")


        # 8. Log Metrics
        collect_and_log_metrics(
            step,
            total_rewards_local, # Rewards calculated in this step
            rewards_per_func_local, # Per-func rewards calculated
            reward_functions,
            advantages_local, # Advantages calculated
            datas['labels'], # Pass the labels mask (marks completions)
            final_ppo_metadata, # Metadata from PPO update
            config,
            len(replay_buffer),
            advantage_estimator,
            was_completion_task # Log whether it was completion
        )

    # End of Training
    logger.info("Training finished.")
    if process_index == 0 and wandb.run is not None and wandb.run.mode != "disabled":
        wandb.finish()
        logger.info("WandB run finished.")

if __name__ == "__main__":
    main()