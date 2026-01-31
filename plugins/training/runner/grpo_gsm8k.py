from __future__ import annotations

import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import math

import numpy as np

from plugins.common.data.padding import pad_2d_right
from plugins.common.sharding.batch import local_from_global, make_form_training_global_array
from plugins.common.wandb_utils import maybe_init_wandb
from plugins.common.tokenizer import prepare_tokenizer
from plugins.training.advantage.estimators import compute_gae_advantages
from plugins.training.update.optimizer import OptimizerConfig
from plugins.training.algorithms import AlgoConfig, create_algorithm
from plugins.training.ppo import get_ppo_state, ppo_training_step

@dataclass(frozen=True)
class GRPORolloutConfig:
    # Prompt batch size per training step (global, across all processes).
    #
    # Each prompt is expanded to `n` sampled completions, so the global
    # sequence batch is: `batch_size * n`.
    batch_size: int = 32
    # Number of samples per prompt (GRPO group size, a.k.a. K / num_pre_q).
    n: int = 8
    global_length: int = 512
    max_length_sample: int = 64
    # Rollout backend selector (swappable generation engine).
    backend: str = "naive"


@dataclass(frozen=True)
class GRPOTrainConfig:
    # Optional: sequences per process per micro-step.
    micro_batch_size: int | None = None
    # Optional: sequences per device per micro-step.
    micro_batch_size_per_device: int | None = None
    # Remat policy for the training module (memory vs compute trade-off).
    #
    # Values:
    # - "dots_with_no_batch_dims": default transformer heuristic (historical)
    # - "nothing_saveable": rematerialize everything (lower HBM, more compute)
    remat_policy: str = "dots_with_no_batch_dims"
    # Optional model memory optimization (compute-for-memory).
    #
    # When enabled, patches `LlamaMLP.__call__` to wrap the SwiGLU gate/up
    # activation (`silu(gate_proj(x)) * up_proj(x)`) in
    # `jax.checkpoint(..., policy=nothing_saveable)`.
    #
    # Intended to reduce peak HBM at long sequence lengths / larger micro-batches.
    mlp_checkpoint_gate_up: bool = False
    max_length_total: int = 0
    ppo_epochs: int = 1
    grad_accum_steps: int = 1
    beta: float = 0.0
    # Policy loss implementation:
    # - "jax": use the legacy pure-JAX loss inside TrainGRPOModule
    # - "pallas": use the Pallas GRPO kernel (plugins/training/kernels)
    policy_loss_impl: str = "jax"
    # Pallas GRPO kernel tuning / numerics (only used when policy_loss_impl=="pallas").
    pallas_block_size: int = 2048
    pallas_time_block: int = 128
    pallas_compute_dtype: str = "bf16"
    pallas_bwd_output_alias_logits: bool = False
    # Optional: log TPU memory stats (jax.devices()[0].memory_stats()) to W&B.
    log_tpu_memory: bool = False
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass(frozen=True)
class GRPOGsm8kConfig:
    # The YAML config path used to construct this run (as passed to the CLI).
    # Kept for W&B traceability (so runs can be mapped back to a committed file).
    config_path: str

    model_path: str
    steps: int
    rollout: GRPORolloutConfig
    train: GRPOTrainConfig
    mesh_shape: str

    wandb_project: str
    wandb_mode: str
    wandb_name: str
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    reward_weights: tuple[float, float, float] = (1.0, 0.5, 0.5)
    eval_every_steps: int = 0
    eval_batches_per_process: int = 1
    eval_split: str = "test"
    # Optional: explicit epoch length override (in training steps).
    #
    # If set > 0, this value is used as the epoch length for triggering
    # `eval_full_every_epochs`. Otherwise, epoch length is derived from the
    # training dataset size and `rollout.batch_size`.
    train_steps_per_epoch: int = 0
    # Full eval sweep (entire eval split) frequency, in epochs.
    #
    # Epoch length is derived from the training dataset size and the effective
    # global prompt batch size (`rollout.batch_size`).
    #
    # - 0: disabled
    # - 1: run full eval at the end of every epoch (and at the end of training)
    eval_full_every_epochs: int = 0
    # Number of completions per question during the full eval sweep.
    eval_full_num_pre_q: int = 1
    # Use greedy decoding for full eval sweep (recommended for deterministic accuracy).
    eval_full_greedy: bool = False


def _ensure_batch_multiple_of_local_devices(local_batch: int, local_device_count: int) -> int:
    if local_batch % local_device_count == 0:
        return local_batch
    return ((local_batch + local_device_count - 1) // local_device_count) * local_device_count


def _as_float(x: Any) -> float:
    return float(np.asarray(x))


def _as_int(x: Any) -> int:
    return int(np.asarray(x))


def _stats_1d(x: np.ndarray) -> dict[str, float]:
    if x.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def _group_correct_per_prompt(correct_per_completion: np.ndarray, *, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert per-completion correctness into per-prompt metrics.

    Returns three float32 vectors of shape [prompt_batch_size]:
    - pass@1: correctness of the first completion in each group
    - pass@K: any correctness within the group
    - mean@K: mean correctness within the group
    """
    correct_per_completion = np.asarray(correct_per_completion, dtype=np.float32).reshape(-1)
    k = int(n)
    if k <= 0:
        raise ValueError("n must be > 0")
    if correct_per_completion.size % k != 0:
        raise ValueError(f"Expected correct_per_completion.size divisible by n={k}, got size={correct_per_completion.size}")
    prompt_batch_size = int(correct_per_completion.size // k)
    grouped = correct_per_completion.reshape(prompt_batch_size, k)
    pass_at_1 = grouped[:, 0]
    pass_at_k = grouped.max(axis=1)
    mean_at_k = grouped.mean(axis=1)
    return pass_at_1, pass_at_k, mean_at_k


def _masked_sum_and_count_1d(values: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    mask = np.asarray(mask, dtype=np.float32).reshape(-1)
    if values.shape != mask.shape:
        raise ValueError(f"values and mask must have the same shape, got {values.shape} vs {mask.shape}")
    masked = values * mask
    return float(masked.sum()), int(mask.sum())


def run_grpo_gsm8k(cfg: GRPOGsm8kConfig) -> None:
    """End-to-end GRPO training loop (rollout → reward → advantages → update)."""
    import jax
    import jax.numpy as jnp
    from datasets import load_dataset
    from jax.experimental.multihost_utils import process_allgather
    from transformers import AutoTokenizer

    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as PS

    from prompts.prompts import system_prompt
    from plugins.training.advantage.modules import GroupIdGRPOAdvantageModule
    from plugins.training.mesh import create_mesh
    from plugins.training.reward.modules import WeightedRewardModule
    from plugins.training.rollout.modules import RolloutBackendModule
    from plugins.training.update.optimizer import build_tx
    from plugins.training.update.train_step import training_step
    from training2 import (
        get_state,
        reward_correct,
        reward_format,
        tag_count_reward,
    )

    # IMPORTANT: slice micro-batches inside a jitted function to preserve sharding.
    #
    # If we slice a sharded global batch with Python indexing (outside jit), JAX may
    # replicate the slice across devices (PartitionSpec()), which increases both
    # per-device compute and memory. On v4-8 co-located train+rollout, that can
    # be the difference between loading `jit_training_step` successfully or OOM.
    def _slice_data_impl(x, accumulate_steps: int, i: int):
        micro_batch_size = int(x.shape[0]) // int(accumulate_steps)
        start = int(i) * micro_batch_size
        return jax.lax.dynamic_slice_in_dim(x, start, micro_batch_size, axis=0)

    slice_data = jax.jit(_slice_data_impl, static_argnums=(1, 2))

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    require_multihost = os.environ.get("REQUIRE_MULTIHOST") == "1"
    require_process_count_raw = os.environ.get("REQUIRE_JAX_PROCESS_COUNT")
    require_process_count = int(require_process_count_raw) if require_process_count_raw else 0

    try:
        jax.distributed.initialize()
    except Exception as e:
        if require_multihost or require_process_count > 0:
            raise RuntimeError(
                "jax.distributed.initialize() failed but a multi-host runtime is required "
                "(REQUIRE_MULTIHOST=1 or REQUIRE_JAX_PROCESS_COUNT is set). "
                "Start the job on all workers (`gcloud ... tpu-vm ssh --worker=all`) "
                "or launch one process per worker."
            ) from e
        if os.environ.get("PRINT_JAX_DISTRIBUTED_INIT_ERROR") == "1":
            print(f"jax.distributed.initialize() skipped: {e}")

    if require_multihost and int(jax.process_count()) <= 1:
        raise RuntimeError(
            "Expected a multi-host JAX runtime (REQUIRE_MULTIHOST=1) but got jax.process_count()==1. "
            "This usually means only worker 0 launched the program."
        )

    if require_process_count > 0:
        actual_process_count = int(jax.process_count())
        if actual_process_count != require_process_count:
            raise RuntimeError(
                f"Expected jax.process_count()=={require_process_count} (from REQUIRE_JAX_PROCESS_COUNT), "
                f"got {actual_process_count}. This usually means only a subset of TPU workers launched the program."
            )

    mesh = create_mesh(cfg.mesh_shape)
    local_device_count = len(mesh.local_devices)
    process_count = int(jax.process_count())
    tp_size = int(mesh.shape.get("tp", 1))

    # Detect the common misconfiguration that makes v6e-16 look dramatically
    # slower than v6e-8: cross-host FSDP sharding from `mesh_shape: 1,-1,1`.
    #
    # This is only a warning (cross-host sharding can be necessary for larger
    # models), but for decode-heavy GRPO it can dominate step time.
    if process_count > 1:
        fsdp_size = int(mesh.shape.get("fsdp", 1))
        if fsdp_size > int(jax.local_device_count()) and int(jax.process_index()) == 0:
            print(
                "WARNING: mesh fsdp axis spans hosts "
                f"(fsdp={fsdp_size} > local_device_count={int(jax.local_device_count())}). "
                "For v6e multi-host GRPO, this often slows rollout decode. "
                "Consider `mesh_shape: auto` (dp=process_count, fsdp=local_device_count) "
                "or an explicit host-local mesh like `mesh_shape: 4,4,1` on v6e-16."
            )

    _form_training_global_array = make_form_training_global_array(mesh)

    # --- Resolve rollout batch sizes ---
    # New semantics: `rollout.batch_size` is global prompts per training step (across all processes).
    requested_global_prompts_per_step = int(cfg.rollout.batch_size)
    if requested_global_prompts_per_step <= 0:
        raise ValueError("rollout.batch_size must be > 0 (global prompts per training step).")

    n = int(cfg.rollout.n)
    if n <= 0:
        raise ValueError("rollout.n must be > 0")

    # We keep rollout passes as an internal mechanism, but expose only a single
    # user-facing knob: global prompt `batch_size`.
    #
    # Each process uses the same local prompt batch size to keep shapes consistent.
    prompts_per_pass = (int(requested_global_prompts_per_step) + int(process_count) - 1) // int(process_count)
    prompts_per_pass = max(1, int(prompts_per_pass))
    effective_global_prompts_per_step = int(prompts_per_pass) * int(process_count)
    if int(jax.process_index()) == 0 and int(effective_global_prompts_per_step) != int(requested_global_prompts_per_step):
        print(
            f"Padding global rollout.batch_size {int(requested_global_prompts_per_step)} -> {int(effective_global_prompts_per_step)} "
            f"so each process gets an equal prompt batch (process_count={int(process_count)})."
        )

    # Ensure per-pass local batch (sequences) can be evenly split across local devices.
    required_multiple = local_device_count // math.gcd(n, local_device_count)
    if required_multiple <= 0:
        raise ValueError("Invalid rollout batching: local_device_count must be > 0")
    if prompts_per_pass % required_multiple != 0:
        padded_prompts_per_pass = ((int(prompts_per_pass) + int(required_multiple) - 1) // int(required_multiple)) * int(
            required_multiple
        )
        effective_global_prompts_per_step = int(padded_prompts_per_pass) * int(process_count)
        if int(jax.process_index()) == 0:
            print(
                f"Padding per-process rollout prompt batch {int(prompts_per_pass)} -> {int(padded_prompts_per_pass)} so that "
                f"(prompts_per_process * rollout.n) is divisible by local_device_count={int(local_device_count)} "
                f"(effective global prompts/step = {int(effective_global_prompts_per_step)})."
            )
        prompts_per_pass = padded_prompts_per_pass

    local_batch_per_pass = int(prompts_per_pass) * int(n)
    rollout_passes = 1

    # --- Resolve train micro-batch & grad accumulation (sequences) ---
    micro_batch_size = cfg.train.micro_batch_size
    if micro_batch_size is not None:
        micro_batch_size = int(micro_batch_size)
        if micro_batch_size <= 0:
            raise ValueError("train.micro_batch_size must be > 0.")

    micro_batch_size_per_device = cfg.train.micro_batch_size_per_device
    if micro_batch_size_per_device is not None:
        micro_batch_size_per_device = int(micro_batch_size_per_device)
        if micro_batch_size_per_device <= 0:
            raise ValueError("train.micro_batch_size_per_device must be > 0.")

        expected_micro_batch_size = int(micro_batch_size_per_device) * int(local_device_count)
        if micro_batch_size is not None and int(micro_batch_size) != expected_micro_batch_size:
            raise ValueError(
                "train.micro_batch_size must match train.micro_batch_size_per_device * local_device_count, got "
                f"{int(micro_batch_size)} vs {int(micro_batch_size_per_device)}*{int(local_device_count)} "
                f"({expected_micro_batch_size})."
            )
        micro_batch_size = expected_micro_batch_size

    if micro_batch_size is not None and micro_batch_size_per_device is None:
        if int(micro_batch_size) % int(local_device_count) == 0:
            micro_batch_size_per_device = int(micro_batch_size) // int(local_device_count)

    local_batch = int(rollout_passes) * int(local_batch_per_pass)
    global_batch = int(local_batch) * int(process_count)

    grad_accum_steps = int(cfg.train.grad_accum_steps)
    if micro_batch_size is not None:
        if local_batch % micro_batch_size != 0:
            raise ValueError(
                f"train.micro_batch_size={micro_batch_size} must divide local_batch={local_batch} "
                f"(prompts_per_process={int(prompts_per_pass)} * rollout.n={n}; "
                f"requested_global_prompts={int(requested_global_prompts_per_step)} effective_global_prompts={int(effective_global_prompts_per_step)})."
            )
        micro_steps = local_batch // micro_batch_size
        if grad_accum_steps != 1 and grad_accum_steps != micro_steps:
            print(
                f"Overriding train.grad_accum_steps {grad_accum_steps} -> {micro_steps} "
                f"to respect train.micro_batch_size={micro_batch_size}."
            )
        grad_accum_steps = micro_steps

    if local_batch % grad_accum_steps != 0:
        raise ValueError(
            f"train.grad_accum_steps={grad_accum_steps} must divide local_batch={local_batch} "
            f"(rollout_passes={rollout_passes} * prompts_per_pass={prompts_per_pass} * rollout.n={n})."
        )

    # Finalize resolved config (for logging/debug): keep `rollout.batch_size` as the
    # effective global prompt batch size per training step.
    resolved_rollout_batch_size = int(effective_global_prompts_per_step)

    cfg = GRPOGsm8kConfig(
        **{
            **cfg.__dict__,
            "rollout": GRPORolloutConfig(
                **{
                    **cfg.rollout.__dict__,
                    "batch_size": int(resolved_rollout_batch_size),
                }
            ),
            "train": GRPOTrainConfig(
                **{
                    **cfg.train.__dict__,
                    "micro_batch_size": micro_batch_size,
                    "micro_batch_size_per_device": micro_batch_size_per_device,
                    "grad_accum_steps": grad_accum_steps,
                }
            ),
        }
    )

    print(f"backend={jax.default_backend()} process={jax.process_index()}/{jax.process_count()}")
    print(f"device_count={jax.device_count()} local_device_count={local_device_count}")
    print(
        "config="
        + str(
            dict(
                model_path=cfg.model_path,
                steps=cfg.steps,
                rollout={
                    **asdict(cfg.rollout),
                    "prompts_per_pass_per_process": int(prompts_per_pass),
                    "sequences_per_pass_per_process": int(local_batch_per_pass),
                    "passes_per_step": int(rollout_passes),
                },
                train=asdict(cfg.train),
                local_batch=local_batch,
                global_batch=global_batch,
                mesh_shape=cfg.mesh_shape,
                wandb_project=cfg.wandb_project,
                wandb_name=cfg.wandb_name,
                reward_weights=cfg.reward_weights,
                eval_every_steps=cfg.eval_every_steps,
                eval_batches_per_process=cfg.eval_batches_per_process,
                eval_split=cfg.eval_split,
            )
        )
    )

    if bool(getattr(cfg.train, "mlp_checkpoint_gate_up", False)):
        from plugins.sample.optimizations.llama_mlp_checkpoint import patch_llama_mlp_checkpoint_gate_up

        patch_llama_mlp_checkpoint_gate_up()
        if jax.process_index() == 0:
            print("mlp_checkpoint_gate_up=1 (patched LlamaMLP.__call__ with jax.checkpoint)")

    dataset = load_dataset("openai/gsm8k", "main", split="train")
    qas = [{"Q": q, "A": a.split("####")[-1].strip()} for q, a in zip(dataset["question"], dataset["answer"])]
    if jax.process_count() > 1:
        qas = qas[jax.process_index() :: jax.process_count()]
    if not qas:
        raise RuntimeError("No GSM8K data after sharding.")

    full_sweep_enabled_env = os.environ.get("EVAL_FULL_SWEEP") == "1"
    full_sweep_enabled_cfg = int(getattr(cfg, "eval_full_every_epochs", 0) or 0) > 0
    full_sweep_enabled = bool(full_sweep_enabled_env or full_sweep_enabled_cfg)
    eval_qas: list[dict[str, str]] = []
    if int(cfg.eval_every_steps) > 0 or full_sweep_enabled:
        eval_dataset = load_dataset("openai/gsm8k", "main", split=str(cfg.eval_split))
        eval_qas = [{"Q": q, "A": a.split("####")[-1].strip()} for q, a in zip(eval_dataset["question"], eval_dataset["answer"])]
        if jax.process_count() > 1:
            eval_qas = eval_qas[jax.process_index() :: jax.process_count()]
        if not eval_qas and jax.process_index() == 0:
            print(f"WARNING: eval enabled but no eval data after sharding (split={cfg.eval_split!r}).")

    from plugins.training.rollout.backends import create_rollout_backend

    rollout_backend_name = str(cfg.rollout.backend).strip().lower()
    if rollout_backend_name == "":
        rollout_backend_name = "naive"

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    tokenizer, pad_token_id = prepare_tokenizer(tokenizer, padding_side="right")

    reward_funcs = [reward_correct, reward_format, tag_count_reward]
    reward_func_names = [fn.__name__ for fn in reward_funcs]
    algo = create_algorithm(cfg.algo, reward_funcs=reward_funcs, reward_weights=cfg.reward_weights)
    reward_module = algo.reward_module
    advantage_module = algo.advantage_module
    update_module = algo.update_module
    use_value_head = algo.requires_value_head
    use_gae_advantage = algo.estimator_name == "gae"
    if jax.process_index() == 0:
        print(
            " ".join(
                [
                    f"algo={algo.name}",
                    f"estimator={algo.estimator_name}",
                    f"update={algo.update_name}",
                    f"value_head={int(use_value_head)}",
                ]
            )
        )

    tx = build_tx(training_steps=cfg.steps, cfg=cfg.train.optimizer)

    value_fn = None
    if use_value_head:
        state, sampler, ppo_module = get_ppo_state(
            mesh,
            training_steps=cfg.steps,
            grad_accum_steps=grad_accum_steps,
            model_path=cfg.model_path,
            update_cfg=cfg.algo.update,
            beta=cfg.train.beta,
            create_sampler=True,
            tx=tx,
        )
        train_fn = jax.jit(ppo_training_step, donate_argnums=(0,))

        def _value_forward(params, input_ids, attention_mask):
            return ppo_module.apply(
                {"params": params},
                input_ids=input_ids,
                attention_mask=attention_mask,
                method=ppo_module.value,
            )

        value_fn = jax.jit(_value_forward)
    else:
        state, sampler, _state_sharding = get_state(
            mesh,
            training_steps=cfg.steps,
            grad_accum_steps=grad_accum_steps,
            model_path=cfg.model_path,
            num_pre_q=cfg.rollout.n,
            max_lengths=cfg.train.max_length_total,
            beta=cfg.train.beta,
            remat_policy=cfg.train.remat_policy,
            policy_loss_impl=cfg.train.policy_loss_impl,
            pallas_block_size=cfg.train.pallas_block_size,
            pallas_time_block=cfg.train.pallas_time_block,
            pallas_compute_dtype=cfg.train.pallas_compute_dtype,
            pallas_bwd_output_alias_logits=cfg.train.pallas_bwd_output_alias_logits,
            create_sampler=True,
            tx=tx,
        )
        train_fn = jax.jit(training_step, donate_argnums=(0,))
    if os.environ.get("ROLLOUT_FAST_QWEN2_DECODE_ATTENTION") == "1":
        from plugins.training.rollout.optimizations import patch_qwen2_attention_decode_fast

        patch_qwen2_attention_decode_fast()
        if jax.process_index() == 0:
            print("rollout_fast_qwen2_decode_attention=1 (patched attention._naive_sdpa for decode)")
    if os.environ.get("ROLLOUT_FAST_GENERATE") == "1":
        from plugins.training.rollout.optimizations import patch_sampler_generate_fast

        patch_sampler_generate_fast(sampler)
        if jax.process_index() == 0:
            print("rollout_fast_generate=1 (patched sampler.generate)")
    rollout_backend = create_rollout_backend(
        name=rollout_backend_name,
        sampler=sampler,
        tokenizer=None,
        model_path=cfg.model_path,
    )
    eval_sampler = sampler
    eval_rollout_backend = rollout_backend
    eval_greedy_enabled = bool(getattr(cfg, "eval_full_greedy", False)) or os.environ.get("EVAL_GREEDY") == "1" or os.environ.get("EVAL_FULL_GREEDY") == "1"
    if eval_greedy_enabled and eval_qas:
        from jax.experimental.shard_map import shard_map
        from plugins.sample.mllm_sampler import Sampler as SamplerImpl

        eval_sampler = SamplerImpl(sampler.model, sampler.tokenizer, mesh=mesh)

        def _greedy_sample(rng, logits):
            rngs = jax.random.split(rng, jax.device_count())

            def sample_inner(_rng, logits_local):
                del _rng
                return jnp.argmax(logits_local, axis=-1)

            sample_fn = shard_map(
                sample_inner,
                mesh=mesh,
                in_specs=(PS(["dp", "fsdp"]), PS(["dp", "fsdp"], "tp")),
                out_specs=PS(["dp", "fsdp"]),
                check_rep=False,
            )
            return sample_fn(rngs, logits)

        eval_sampler.sample_fn = jax.jit(_greedy_sample)
        eval_sampler.jit_infer_step = jax.jit(eval_sampler.infer, donate_argnums=(0,))

        if os.environ.get("ROLLOUT_FAST_GENERATE") == "1":
            from plugins.training.rollout.optimizations import patch_sampler_generate_fast

            patch_sampler_generate_fast(eval_sampler)

        eval_rollout_backend = create_rollout_backend(
            name=rollout_backend_name,
            sampler=eval_sampler,
            tokenizer=None,
            model_path=cfg.model_path,
        )
        if jax.process_index() == 0:
            print("eval_greedy=1 (eval rollout uses argmax sampling)")

    rollout_module = RolloutBackendModule(backend=rollout_backend)
    eval_rollout_module = RolloutBackendModule(backend=eval_rollout_backend)

    wandb = maybe_init_wandb(
        cfg=cfg,
        project=cfg.wandb_project,
        name=cfg.wandb_name,
        mode=cfg.wandb_mode,
        process_index=jax.process_index(),
    )

    def _policy_params_for_rollout(params):
        if not use_value_head:
            return params
        return {"model": params["model"], "lm_head": params["lm_head"]}

    # --- Epoch length (derived) ---
    #
    # Define 1 "epoch" as consuming ~one full pass worth of prompts from the
    # training split, based on the effective global prompt batch size per step.
    local_train_question_count = int(len(qas))
    local_train_count_arr = np.asarray([local_train_question_count], dtype=np.int32)
    global_train_question_count = int(np.asarray(process_allgather(local_train_count_arr)).sum())
    global_prompts_per_step = max(int(cfg.rollout.batch_size), 1)
    steps_per_epoch_derived = max(int(math.ceil(float(global_train_question_count) / float(global_prompts_per_step))), 1)
    steps_per_epoch_override = int(getattr(cfg, "train_steps_per_epoch", 0) or 0)
    steps_per_epoch = int(steps_per_epoch_override) if steps_per_epoch_override > 0 else int(steps_per_epoch_derived)
    if jax.process_index() == 0:
        print(
            "epoch_length "
            + " ".join(
                [
                    f"train_questions_global={global_train_question_count}",
                    f"rollout_batch_size_global={global_prompts_per_step}",
                    f"steps_per_epoch={steps_per_epoch}",
                    f"steps_per_epoch_override={steps_per_epoch_override}",
                ]
            )
        )

    def _full_eval_num_pre_q() -> int:
        # Config first; env override kept for backwards compatibility.
        cfg_eval_n = int(getattr(cfg, "eval_full_num_pre_q", 1) or 1)
        env_eval_n_raw = os.environ.get("EVAL_FULL_NUM_PRE_Q")
        if env_eval_n_raw is None:
            return cfg_eval_n
        try:
            env_eval_n = int(env_eval_n_raw)
        except Exception as e:
            raise ValueError(f"EVAL_FULL_NUM_PRE_Q must be an int, got {env_eval_n_raw!r}") from e
        if jax.process_index() == 0 and env_eval_n != cfg_eval_n:
            print(f"WARNING: EVAL_FULL_NUM_PRE_Q overrides config eval_full_num_pre_q ({cfg_eval_n} -> {env_eval_n})")
        return env_eval_n

    def _run_full_eval_sweep(*, step: int) -> None:
        if not eval_qas:
            return

        # Keep compilation stable by matching the per-process sequence batch size of training:
        #   (prompt_batch_size_eval * eval_n) == local_seq_batch_train
        train_prompts_per_pass = int(prompts_per_pass)
        train_n = int(n)
        local_seq_batch_train = train_prompts_per_pass * train_n

        eval_n = int(_full_eval_num_pre_q())
        if eval_n < 1:
            raise ValueError(f"eval_full_num_pre_q must be >= 1, got {eval_n}")
        if local_seq_batch_train % eval_n != 0:
            raise ValueError(
                "eval_full_num_pre_q must divide the per-process sequence batch size: "
                f"{eval_n=} local_seq_batch_train={local_seq_batch_train}"
            )

        prompt_batch_size = local_seq_batch_train // eval_n

        t_eval0 = time.perf_counter()
        t_eval_sync_s = 0.0
        t_eval_rollout_generate_s = 0.0
        t_eval_rollout_flush_s = 0.0
        t_eval_reward_s = 0.0
        t_eval_release_s = 0.0

        local_question_count = int(len(eval_qas))
        local_prompt_sum_accuracy = 0.0
        local_prompt_count = 0

        local_completion_sum_correct = 0.0
        local_completion_sum_format = 0.0
        local_completion_sum_tag = 0.0
        local_completion_count = 0

        t0 = time.perf_counter()
        eval_policy_params = _policy_params_for_rollout(state.params)
        eval_rollout_module.sync_weights(eval_policy_params)
        t_eval_sync_s += time.perf_counter() - t0

        pad_item = eval_qas[0]
        for start in range(0, local_question_count, prompt_batch_size):
            batch_items = eval_qas[start : start + prompt_batch_size]
            valid_prompts = int(len(batch_items))
            if valid_prompts <= 0:
                continue
            if valid_prompts < prompt_batch_size:
                batch_items = batch_items + [pad_item] * (prompt_batch_size - valid_prompts)

            eval_prompts_base = [item["Q"] for item in batch_items]
            eval_repeated_prompts = [p for p in eval_prompts_base for _ in range(eval_n)]
            eval_repeated_items = [item for item in batch_items for _ in range(eval_n)]

            t0 = time.perf_counter()
            eval_rollout = eval_rollout_module.rollout(
                prompts=eval_repeated_prompts,
                sampler=eval_sampler,
                params=eval_policy_params,
                system_prompt=system_prompt,
                global_length=int(cfg.rollout.global_length),
                max_length_sample=cfg.rollout.max_length_sample,
            )
            eval_answers = eval_rollout.answers
            t_eval_rollout_generate_s += time.perf_counter() - t0

            t0 = time.perf_counter()
            eval_rollout_module.flush_cache()
            t_eval_rollout_flush_s += time.perf_counter() - t0

            t0 = time.perf_counter()
            eval_rewards_out = reward_module.compute(inputs=eval_repeated_items, answers=eval_answers)
            eval_rewards_per_func = eval_rewards_out.rewards_per_func
            t_eval_reward_s += time.perf_counter() - t0

            correct_per_completion = np.asarray(eval_rewards_per_func[0], dtype=np.float32).reshape(-1)
            format_per_completion = np.asarray(eval_rewards_per_func[1], dtype=np.float32).reshape(-1)
            tag_per_completion = np.asarray(eval_rewards_per_func[2], dtype=np.float32).reshape(-1)

            prompt_mask = np.zeros((prompt_batch_size,), dtype=np.float32)
            prompt_mask[:valid_prompts] = 1.0
            completion_mask = np.repeat(prompt_mask, eval_n)

            # "Accuracy" is defined as correctness of the first (or only)
            # completion per prompt, matching the "each question once" metric.
            if eval_n == 1:
                local_prompt_sum_accuracy += float((correct_per_completion * prompt_mask).sum())
            else:
                pass_at_1, _pass_at_k, _mean_at_k = _group_correct_per_prompt(correct_per_completion, n=eval_n)
                local_prompt_sum_accuracy += float((pass_at_1 * prompt_mask).sum())

            local_prompt_count += int(valid_prompts)
            local_completion_sum_correct += float((correct_per_completion * completion_mask).sum())
            local_completion_sum_format += float((format_per_completion * completion_mask).sum())
            local_completion_sum_tag += float((tag_per_completion * completion_mask).sum())
            local_completion_count += int(valid_prompts) * int(eval_n)

        t0 = time.perf_counter()
        eval_rollout_module.release_weights()
        t_eval_release_s += time.perf_counter() - t0

        local_stats = np.asarray(
            [
                float(local_question_count),
                float(local_prompt_count),
                float(local_prompt_sum_accuracy),
                float(local_completion_count),
                float(local_completion_sum_correct),
                float(local_completion_sum_format),
                float(local_completion_sum_tag),
            ],
            dtype=np.float64,
        )
        gathered = np.asarray(process_allgather(local_stats))
        global_stats = gathered.sum(axis=0)

        global_question_count = int(global_stats[0])
        global_prompt_count = int(global_stats[1])
        global_prompt_sum_accuracy = float(global_stats[2])
        global_completion_count = int(global_stats[3])
        global_completion_sum_correct = float(global_stats[4])
        global_completion_sum_format = float(global_stats[5])
        global_completion_sum_tag = float(global_stats[6])

        t_eval_s = time.perf_counter() - t_eval0
        prompt_denom = max(global_prompt_count, 1)
        completion_denom = max(global_completion_count, 1)

        accuracy_global = global_prompt_sum_accuracy / float(prompt_denom)
        mean_correct_per_completion = global_completion_sum_correct / float(completion_denom)
        mean_format_per_completion = global_completion_sum_format / float(completion_denom)
        mean_tag_per_completion = global_completion_sum_tag / float(completion_denom)

        epoch = int((int(step) + 1) // steps_per_epoch) if steps_per_epoch > 0 else 0

        full_eval_logs: dict[str, Any] = {
            "eval_full/enabled": 1,
            "eval_full/epoch": int(epoch),
            "eval_full/split": str(cfg.eval_split),
            "eval_full/questions_global": global_question_count,
            "eval_full/samples_per_question": int(eval_n),
            "eval_full/accuracy": float(accuracy_global),
            # Backward-compatible alias (previously logged as pass@1).
            "eval_full/accuracy/pass_at_1": float(accuracy_global),
            # Per-completion means (kept for reference; NOT the main "accuracy").
            "eval_full/reward_correct/mean_per_completion": float(mean_correct_per_completion),
            "eval_full/reward_format/mean_per_completion": float(mean_format_per_completion),
            "eval_full/tag_count_reward/mean_per_completion": float(mean_tag_per_completion),
            "time/eval_full/sync_s": float(t_eval_sync_s),
            "time/eval_full/rollout_generate_s": float(t_eval_rollout_generate_s),
            "time/eval_full/rollout_flush_s": float(t_eval_rollout_flush_s),
            "time/eval_full/reward_s": float(t_eval_reward_s),
            "time/eval_full/release_s": float(t_eval_release_s),
            "time/eval_full/step_s": float(t_eval_s),
        }
        if global_question_count > 0:
            full_eval_logs["time/eval_full/s_per_question"] = float(t_eval_s) / float(global_question_count)

        full_eval_logs.update(
            {
                "eval/accuracy/full_sweep": 1,
                "eval/accuracy/epoch": int(epoch),
                "eval/accuracy/split": str(cfg.eval_split),
                "eval/accuracy/questions_global": global_question_count,
                "eval/accuracy/samples_per_question": int(eval_n),
                "eval/accuracy/accuracy": float(accuracy_global),
                "eval/accuracy/pass_at_1": float(accuracy_global),
            }
        )

        if jax.process_index() == 0:
            print(
                "eval_full "
                + " ".join(
                    [
                        f"epoch={epoch}",
                        f"split={cfg.eval_split}",
                        f"questions={global_question_count}",
                        f"samples_per_question={eval_n}",
                        f"accuracy={accuracy_global:.4f}",
                        f"t={t_eval_s:.2f}s",
                    ]
                )
            )
        if wandb is not None and jax.process_index() == 0:
            wandb.log(full_eval_logs, step=int(step))

    rng = random.Random(0xC0FFEE + jax.process_index())
    step_times: list[float] = []
    for step in range(cfg.steps):
        t_step0 = time.perf_counter()

        # --- Rollout (sampling) ---
        # Keep rollout logic local to avoid cross-host coupling; global sync happens at the batch->global array step.
        answers_all: list[str] = []
        datas_np_all: list[dict[str, np.ndarray]] = []
        rewards_per_func_all: list[np.ndarray] = []
        rewards_all: list[np.ndarray] = []
        advantages_all: list[np.ndarray] = []
        group_ids_all: list[np.ndarray] = []

        t_rollout = 0.0
        t_rollout_sync = 0.0
        t_rollout_generate = 0.0
        t_rollout_flush = 0.0
        t_rollout_release = 0.0
        t_reward = 0.0
        t_adv = 0.0

        for pass_idx in range(int(rollout_passes)):
            batch_items = [rng.choice(qas) for _ in range(int(prompts_per_pass))]
            prompts_base = [item["Q"] for item in batch_items]
            repeated_prompts = [p for p in prompts_base for _ in range(int(n))]
            repeated_items = [item for item in batch_items for _ in range(int(n))]

            group_ids = np.repeat(
                np.arange(int(prompts_per_pass), dtype=np.int32) + pass_idx * int(prompts_per_pass),
                int(n),
            )

            # --- Rollout (sampling) ---
            t_sync0 = time.perf_counter()
            policy_params = _policy_params_for_rollout(state.params)
            rollout_module.sync_weights(policy_params)
            t_rollout_sync += time.perf_counter() - t_sync0

            t_rollout0 = time.perf_counter()
            rollout = rollout_module.rollout(
                prompts=repeated_prompts,
                sampler=sampler,
                params=policy_params,
                system_prompt=system_prompt,
                global_length=int(cfg.rollout.global_length),
                max_length_sample=cfg.rollout.max_length_sample,
            )
            answers = rollout.answers
            datas_np = rollout.batch
            t_rollout_generate += time.perf_counter() - t_rollout0

            t_flush0 = time.perf_counter()
            rollout_module.flush_cache()
            t_rollout_flush += time.perf_counter() - t_flush0

            # --- Reward ---
            t_reward0 = time.perf_counter()
            rewards_out = reward_module.compute(inputs=repeated_items, answers=answers)
            rewards_per_func, rewards_np = rewards_out.rewards_per_func, rewards_out.rewards
            t_reward += time.perf_counter() - t_reward0

            # --- Advantages (group_id based) ---
            advantages_np = None
            if not use_gae_advantage:
                if advantage_module is None:
                    raise RuntimeError("advantage_module is required for non-GAE estimators")
                t_adv0 = time.perf_counter()
                advantages_np = advantage_module.compute(rewards=rewards_np, group_ids=group_ids).advantages
                t_adv += time.perf_counter() - t_adv0

            datas_np = dict(datas_np)
            datas_np["rewards"] = rewards_np
            if advantages_np is not None:
                datas_np["advantages"] = advantages_np
            datas_np["group_ids"] = group_ids

            answers_all.extend(answers)
            datas_np_all.append(datas_np)
            rewards_per_func_all.append(rewards_per_func)
            rewards_all.append(rewards_np)
            if advantages_np is not None:
                advantages_all.append(advantages_np)
            group_ids_all.append(group_ids)

        t_release0 = time.perf_counter()
        rollout_module.release_weights()
        t_rollout_release = time.perf_counter() - t_release0

        t_rollout = t_rollout_sync + t_rollout_generate + t_rollout_flush + t_rollout_release
        rewards_np = np.concatenate(rewards_all, axis=0) if rewards_all else np.asarray([], dtype=np.float32)
        advantages_np = np.concatenate(advantages_all, axis=0) if advantages_all else np.asarray([], dtype=np.float32)
        group_ids = np.concatenate(group_ids_all, axis=0) if group_ids_all else np.asarray([], dtype=np.int32)
        rewards_per_func = (
            np.concatenate(rewards_per_func_all, axis=1) if rewards_per_func_all else np.asarray([], dtype=np.float32)
        )

        # Concatenate per-pass batches. If sequence lengths differ, pad up to the global max length.
        datas_np = {}
        if datas_np_all:
            keys = datas_np_all[0].keys()
            for k in keys:
                if k in {"input_ids", "attention_mask", "labels"}:
                    max_len_local = max(int(d[k].shape[1]) for d in datas_np_all)
                    if k == "input_ids":
                        pad_value = pad_token_id
                    else:
                        pad_value = 0
                    parts = [pad_2d_right(d[k], max_len_local, pad_value) for d in datas_np_all]
                    datas_np[k] = np.concatenate(parts, axis=0)
                else:
                    datas_np[k] = np.concatenate([d[k] for d in datas_np_all], axis=0)

            seq_len_local = int(datas_np["input_ids"].shape[1])
            seq_len_global = int(np.asarray(process_allgather(np.asarray([seq_len_local], dtype=np.int32))).max())
            if seq_len_global != seq_len_local:
                datas_np["input_ids"] = pad_2d_right(datas_np["input_ids"], seq_len_global, pad_token_id)
                datas_np["attention_mask"] = pad_2d_right(datas_np["attention_mask"], seq_len_global, 0)
                datas_np["labels"] = pad_2d_right(datas_np["labels"], seq_len_global, 0)

        if use_gae_advantage and datas_np:
            t_adv0 = time.perf_counter()
            if value_fn is None:
                raise RuntimeError("GAE estimator requires value_fn (PPO update with value head)")
            value_inputs = {
                "input_ids": datas_np["input_ids"],
                "attention_mask": datas_np["attention_mask"],
            }
            value_inputs_jax = jax.tree_util.tree_map_with_path(_form_training_global_array, value_inputs)
            values = value_fn(state.params, value_inputs_jax["input_ids"], value_inputs_jax["attention_mask"])
            jax.block_until_ready(values)
            values_np = local_from_global(values)
            values_pred = np.asarray(values_np[:, :-1], dtype=np.float32)
            completion_mask = np.asarray(datas_np["labels"][:, 1:], dtype=np.float32)
            advantages_np, returns_np = compute_gae_advantages(
                rewards=rewards_np,
                values=values_pred,
                completion_mask=completion_mask,
                gamma=cfg.algo.estimator.gae_gamma,
                gae_lambda=cfg.algo.estimator.gae_lambda,
                normalize=cfg.algo.estimator.gae_normalize,
                eps=cfg.algo.estimator.eps,
                clip_range=cfg.algo.estimator.clip_range,
            )
            datas_np["values"] = values_pred
            datas_np["returns"] = returns_np
            datas_np["advantages"] = advantages_np
            t_adv += time.perf_counter() - t_adv0

        rewards_global = np.asarray(process_allgather(rewards_np)).reshape(-1)
        reward_global_stats = _stats_1d(rewards_global)

        # --- Update ---
        t_shard0 = time.perf_counter()
        datas = jax.tree_util.tree_map_with_path(
            _form_training_global_array,
            datas_np,
        )
        total_valid_token_count = datas["labels"][:, 1:].sum()
        t_shard = time.perf_counter() - t_shard0

        t_update0 = time.perf_counter()
        update_out = update_module.update(
            state=state,
            batch=datas,
            total_valid_token_count=total_valid_token_count,
            train_step=train_fn,
            slice_data=slice_data,
            grad_accum_steps=grad_accum_steps,
            ppo_steps=cfg.train.ppo_epochs,
        )
        state, datas, last_meta, entropy = (
            update_out.state,
            update_out.batch,
            update_out.last_meta,
            update_out.entropy,
        )
        jax.block_until_ready(last_meta["loss"])
        t_update = time.perf_counter() - t_update0

        t_step = time.perf_counter() - t_step0
        step_times.append(float(t_step))

        loss_value = _as_float(last_meta["loss"])
        if entropy is None:
            entropy_value = _as_float(jnp.mean(last_meta["entropy"]))
        else:
            entropy_value = _as_float(entropy)

        # --- Derived stats (global) ---
        if advantages_np.ndim == 1:
            advantages_global = np.asarray(process_allgather(advantages_np)).reshape(-1)
            adv_global_stats = _stats_1d(advantages_global)
        else:
            advantages_global = np.asarray(process_allgather(advantages_np))
            mask_global = np.asarray(process_allgather(np.asarray(datas_np["labels"][:, 1:], dtype=np.float32)))
            advantages_flat = advantages_global[mask_global > 0]
            adv_global_stats = _stats_1d(advantages_flat.astype(np.float32))

        rewards_per_func_global = np.asarray(process_allgather(rewards_per_func))
        # shape [process_count, num_funcs, B_local]
        per_func_means = rewards_per_func_global.mean(axis=(0, 2))

        labels_np = np.asarray(datas_np["labels"])
        attn_np = np.asarray(datas_np["attention_mask"])
        completion_len_local = labels_np.sum(axis=1).astype(np.float32)
        total_len_local = attn_np.sum(axis=1).astype(np.float32)
        prompt_len_local = (total_len_local - completion_len_local).astype(np.float32)

        completion_len_global = np.asarray(process_allgather(completion_len_local)).reshape(-1)
        total_len_global = np.asarray(process_allgather(total_len_local)).reshape(-1)
        prompt_len_global = np.asarray(process_allgather(prompt_len_local)).reshape(-1)

        completion_stats = _stats_1d(completion_len_global)
        prompt_stats = _stats_1d(prompt_len_global)
        total_len_stats = _stats_1d(total_len_global)

        valid_tokens_local = int(labels_np[:, 1:].sum())
        valid_tokens_global = int(np.asarray(process_allgather(np.asarray([valid_tokens_local], dtype=np.int64))).sum())
        global_batch = int(local_batch * jax.process_count())

        train_log: dict[str, Any] = {
            # Other
            "train-other/loss": loss_value,
            "train-other/entropy": entropy_value,
            "train-other/batch_global": global_batch,
            "train-other/batch_local": int(local_batch),
            "train-other/total_valid_token_count": valid_tokens_global,
            # Reward / advantage
            "train-reward/total/mean": reward_global_stats["mean"],
            "train-reward/total/std": reward_global_stats["std"],
            "train-reward/total/min": reward_global_stats["min"],
            "train-reward/total/max": reward_global_stats["max"],
            "train-reward/advantage/mean": adv_global_stats["mean"],
            "train-reward/advantage/std": adv_global_stats["std"],
            "train-reward/advantage/min": adv_global_stats["min"],
            "train-reward/advantage/max": adv_global_stats["max"],
            # Seq lengths
            "train-seq_len/prompt/mean": prompt_stats["mean"],
            "train-seq_len/prompt/max": prompt_stats["max"],
            "train-seq_len/completion/mean": completion_stats["mean"],
            "train-seq_len/completion/max": completion_stats["max"],
            "train-seq_len/total/mean": total_len_stats["mean"],
            "train-seq_len/total/max": total_len_stats["max"],
            # Timing
            "time/train/rollout_s": float(t_rollout),
            "time/train/rollout_sync_s": float(t_rollout_sync),
            "time/train/rollout_generate_s": float(t_rollout_generate),
            "time/train/rollout_flush_s": float(t_rollout_flush),
            "time/train/rollout_release_s": float(t_rollout_release),
            "time/train/reward_s": float(t_reward),
            "time/train/advantages_s": float(t_adv),
            "time/train/shard_s": float(t_shard),
            "time/train/update_s": float(t_update),
            "time/train/step_s": float(t_step),
        }
        if bool(getattr(cfg.train, "log_tpu_memory", False)) and jax.process_index() == 0:
            try:
                mem_stats = jax.devices()[0].memory_stats()
            except Exception:
                mem_stats = None
            if isinstance(mem_stats, dict):
                # Log the most useful stable keys across TPU backends.
                for k in [
                    "bytes_in_use",
                    "peak_bytes_in_use",
                    "bytes_reserved",
                    "peak_bytes_reserved",
                    "bytes_limit",
                    "bytes_reservable_limit",
                    "largest_alloc_size",
                    "largest_free_block_bytes",
                    "num_allocs",
                ]:
                    v = mem_stats.get(k)
                    if v is None:
                        continue
                    try:
                        train_log[f"tpu/mem/{k}"] = int(v)
                    except Exception:
                        train_log[f"tpu/mem/{k}"] = str(v)
        if "policy_loss" in last_meta:
            train_log["train-ppo/policy_loss"] = _as_float(last_meta["policy_loss"])
        if "value_loss" in last_meta:
            train_log["train-ppo/value_loss"] = _as_float(last_meta["value_loss"])
        if "value_pred_mean" in last_meta:
            train_log["train-ppo/value_pred_mean"] = _as_float(last_meta["value_pred_mean"])
        if "return_mean" in last_meta:
            train_log["train-ppo/return_mean"] = _as_float(last_meta["return_mean"])
        if len(step_times) >= 10:
            train_log["time/train/step_avg_last10_s"] = float(sum(step_times[-10:]) / 10.0)
        for name, mean_value in zip(reward_func_names, per_func_means):
            train_log[f"train-reward/func/{name}/mean"] = float(mean_value)

        if t_step > 0:
            train_log["throughput/train/valid_tokens_per_s"] = float(valid_tokens_global) / float(t_step)
        if t_update > 0:
            train_log["throughput/train/valid_tokens_per_s_update"] = float(valid_tokens_global) / float(t_update)

        if wandb is not None and jax.process_index() == 0:
            wandb.log(train_log, step=step)

        if jax.process_index() == 0:
            parts = [
                f"step={step}",
                f"loss={loss_value:.6f}",
                f"entropy={entropy_value:.4f}",
                f"reward_mean={reward_global_stats['mean']:.4f}",
                f"dt={t_step:.2f}s",
            ]
            if os.environ.get("PRINT_TRAIN_TIME_BREAKDOWN") == "1":
                max_length_sample = int(cfg.rollout.max_length_sample)
                if completion_len_global.size == 0:
                    completion_hit_max_frac = float("nan")
                else:
                    completion_hit_max_frac = float((completion_len_global >= max_length_sample).mean())
                parts.extend(
                    [
                        f"rollout={t_rollout:.2f}s",
                        f"rollout_sync={t_rollout_sync:.2f}s",
                        f"rollout_generate={t_rollout_generate:.2f}s",
                        f"rollout_flush={t_rollout_flush:.2f}s",
                        f"rollout_release={t_rollout_release:.2f}s",
                        f"reward={t_reward:.2f}s",
                        f"advantages={t_adv:.2f}s",
                        f"shard={t_shard:.2f}s",
                        f"update={t_update:.2f}s",
                        f"prompt_len_mean={prompt_stats['mean']:.1f}",
                        f"completion_len_mean={completion_stats['mean']:.1f}",
                        f"completion_len_max={completion_stats['max']:.0f}",
                        f"completion_hit_max_frac={completion_hit_max_frac:.2f}",
                    ]
                )
            print(" ".join(parts))

        # --- Eval (optional; no updates) ---
        if eval_qas and int(cfg.eval_every_steps) > 0 and ((step + 1) % int(cfg.eval_every_steps) == 0):
            eval_logs: dict[str, Any] = {}
            eval_rollout_s = 0.0
            eval_rollout_sync_s = 0.0
            eval_rollout_generate_s = 0.0
            eval_rollout_flush_s = 0.0
            eval_rollout_release_s = 0.0
            eval_reward_s = 0.0
            eval_step0 = time.perf_counter()
            eval_rewards_all: list[np.ndarray] = []
            eval_rewards_per_func_all: list[np.ndarray] = []

            for eval_batch_idx in range(int(cfg.eval_batches_per_process)):
                start = (step * int(cfg.eval_batches_per_process) + eval_batch_idx) * int(
                    prompts_per_pass
                )
                eval_items = [
                    eval_qas[(start + i) % len(eval_qas)] for i in range(int(prompts_per_pass))
                ]
                eval_prompts_base = [item["Q"] for item in eval_items]
                eval_repeated_prompts = [p for p in eval_prompts_base for _ in range(int(n))]
                eval_repeated_items = [item for item in eval_items for _ in range(int(n))]

                t_eval_sync0 = time.perf_counter()
                eval_policy_params = _policy_params_for_rollout(state.params)
                eval_rollout_module.sync_weights(eval_policy_params)
                eval_rollout_sync_s += time.perf_counter() - t_eval_sync0

                t_eval_rollout0 = time.perf_counter()
                eval_rollout = eval_rollout_module.rollout(
                    prompts=eval_repeated_prompts,
                    sampler=eval_sampler,
                    params=eval_policy_params,
                    system_prompt=system_prompt,
                    global_length=int(cfg.rollout.global_length),
                    max_length_sample=cfg.rollout.max_length_sample,
                )
                eval_answers = eval_rollout.answers
                eval_rollout_generate_s += time.perf_counter() - t_eval_rollout0

                t_eval_flush0 = time.perf_counter()
                eval_rollout_module.flush_cache()
                eval_rollout_flush_s += time.perf_counter() - t_eval_flush0

                t_eval_reward0 = time.perf_counter()
                eval_rewards_out = reward_module.compute(inputs=eval_repeated_items, answers=eval_answers)
                eval_rewards_per_func, eval_rewards_np = eval_rewards_out.rewards_per_func, eval_rewards_out.rewards
                eval_reward_s += time.perf_counter() - t_eval_reward0

                eval_rewards_global = np.asarray(process_allgather(eval_rewards_np)).reshape(-1)
                eval_rewards_all.append(eval_rewards_global)

                eval_per_func_global = np.asarray(process_allgather(eval_rewards_per_func))
                # shape [process_count, num_funcs, B_local] -> [num_funcs, process_count * B_local]
                eval_per_func_flat = eval_per_func_global.transpose(1, 0, 2).reshape(len(reward_funcs), -1)
                eval_rewards_per_func_all.append(eval_per_func_flat)

            t_eval_release0 = time.perf_counter()
            eval_rollout_module.release_weights()
            eval_rollout_release_s = time.perf_counter() - t_eval_release0

            eval_step_s = time.perf_counter() - eval_step0
            eval_rollout_s = eval_rollout_sync_s + eval_rollout_generate_s + eval_rollout_flush_s + eval_rollout_release_s
            eval_rewards_concat = np.concatenate(eval_rewards_all, axis=0) if eval_rewards_all else np.asarray([], dtype=np.float32)
            eval_reward_stats = _stats_1d(eval_rewards_concat)

            eval_logs["eval/reward/total/mean"] = float(eval_reward_stats["mean"])
            eval_logs["eval/reward/total/std"] = float(eval_reward_stats["std"])
            eval_logs["eval/reward/total/min"] = float(eval_reward_stats["min"])
            eval_logs["eval/reward/total/max"] = float(eval_reward_stats["max"])

            if eval_rewards_per_func_all:
                eval_per_func_concat = np.concatenate(eval_rewards_per_func_all, axis=1)
                eval_per_func_means = eval_per_func_concat.mean(axis=1)
                for name, mean_value in zip(reward_func_names, eval_per_func_means):
                    eval_logs[f"eval/reward/func/{name}/mean"] = float(mean_value)

            eval_logs["time/eval/rollout_s"] = float(eval_rollout_s)
            eval_logs["time/eval/rollout_sync_s"] = float(eval_rollout_sync_s)
            eval_logs["time/eval/rollout_generate_s"] = float(eval_rollout_generate_s)
            eval_logs["time/eval/rollout_flush_s"] = float(eval_rollout_flush_s)
            eval_logs["time/eval/rollout_release_s"] = float(eval_rollout_release_s)
            eval_logs["time/eval/reward_s"] = float(eval_reward_s)
            eval_logs["time/eval/step_s"] = float(eval_step_s)

            if wandb is not None and jax.process_index() == 0:
                wandb.log(eval_logs, step=step)

        # --- Full eval sweep (optional; no updates) ---
        #
        # Config-driven full-sweep eval:
        # - `eval_full_every_epochs>0`: run full eval at the end of every N epochs
        # - `EVAL_FULL_SWEEP=1`: backwards-compatible flag (runs at end of training only)
        if full_sweep_enabled and eval_qas:
            epoch_end = steps_per_epoch > 0 and ((step + 1) % steps_per_epoch == 0)
            eval_every_epochs = int(getattr(cfg, "eval_full_every_epochs", 0) or 0)
            run_on_epoch = bool(
                eval_every_epochs > 0
                and epoch_end
                and (((step + 1) // steps_per_epoch) % eval_every_epochs == 0)
            )
            run_on_end = bool(step == int(cfg.steps) - 1)
            if run_on_epoch or run_on_end:
                _run_full_eval_sweep(step=step)

    # Full eval sweep is executed inside the training loop when enabled.
