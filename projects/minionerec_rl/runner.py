from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from projects.sid_sft.jax.evaluator import evaluate_sid_next_item_jax
from projects.sid_sft.jax.params import resize_lm_vocab
from projects.sid_sft.tokens import maybe_extend_tokenizer
from projects.minionerec_rl.datasets import (
    MiniOneRecMixedRlDataset,
    MiniOneRecNextItemRlDataset,
    MiniOneRecSeqTitle2SidRlDataset,
    MiniOneRecTitle2SidRlDataset,
)
from projects.minionerec_rl.grpo_module import MiniOneRecGrpoModule
from projects.minionerec_rl.reward import build_rank_penalties, compute_ranking_rewards
from plugins.sample.constraints.sid_trie import build_sid_trie_from_index
from plugins.training.core.checkpoint.msgpack import load_checkpoint, save_checkpoint
from plugins.training.core.io.hf_config import ensure_rope_theta
from plugins.training.core.logging.wandb import maybe_init_wandb
from plugins.training.core.optim.optimizer import OptimizerConfig, build_tx
from plugins.training.core.tokenizer import prepare_tokenizer
from plugins.training.rl.advantage.grpo import compute_grpo_advantages_by_group_id


@dataclass(frozen=True)
class MiniOneRecRlDataConfig:
    category: str
    train_file: str
    eval_file: str
    test_file: str
    info_file: str
    sid_index_path: str
    item_meta_path: str | None = None
    # Upstream RL mixes multiple prompt types; keep next-item as default, and
    # optionally enable additional tasks for better alignment.
    enable_title2sid: bool = False
    enable_seqtitle2sid: bool = False
    sample_title2sid: int = -1
    sample_seqtitle2sid: int = 10_000
    max_len: int = 512
    sample_train: int = -1
    sample_eval: int = -1
    sample_test: int = -1


@dataclass(frozen=True)
class MiniOneRecRlJaxConfig:
    mesh_shape: str = "auto"
    param_dtype: str = "bfloat16"
    compute_dtype: str = "bfloat16"
    max_cache_length: int = 512


@dataclass(frozen=True)
class MiniOneRecRlRolloutConfig:
    prompt_batch_size: int = 32  # prompts per update (global)
    # Optional: split constrained decoding (beam search) into prompt micro-batches
    # of this size (0 => no split).
    prompt_micro_batch_size: int = 0
    num_generations: int = 16
    prompt_pad_len: int = 256  # fixed prompt padding (no length bucketing)
    global_length: int = 512  # fixed training padding


@dataclass(frozen=True)
class MiniOneRecRlTrainConfig:
    num_train_epochs: float = 2.0
    max_steps: int = -1
    # Align with upstream TRL/Accelerate semantics: number of prompt-batches to
    # accumulate before each optimizer update (effective batch multiplier).
    gradient_accumulation_steps: int = 1
    # Micro-batch splitting within a single prompt-batch to keep TPU compile/memory
    # bounded (does NOT change the effective batch size).
    grad_accum_steps: int = 1
    ppo_steps: int = 1
    beta: float = 1e-3
    sync_ref_model: bool = False
    sync_ref_model_every_steps: int = 0
    sync_ref_model_mixup_alpha: float = 1.0
    logging_steps: int = 10
    # Full constrained-decoding eval cadence during training (0 => once per epoch).
    eval_steps: int = 0
    save_last: bool = True
    # Optional: save the best checkpoint according to constrained-decoding eval
    # metrics computed on `eval.split`.
    save_best: bool = False
    save_best_metric: str = "ndcg@10"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass(frozen=True)
class MiniOneRecRlEvalConfig:
    enabled: bool = True
    # Which split to use for constrained-decoding evaluation:
    # - "test": use `data.test_file` (default; preserves historical behavior)
    # - "eval": use `data.eval_file` (validation)
    split: str = "test"
    # If enabled and `train.save_best=true`, `train_eval` will load and evaluate
    # the best checkpoint saved during training (instead of the last-step params).
    use_best_checkpoint: bool = False
    every_steps: int = 0
    batch_size: int = 2
    num_beams: int = 50
    topk: tuple[int, ...] = (1, 3, 5, 10, 20, 50)
    save_predictions_json: bool = True


@dataclass(frozen=True)
class MiniOneRecRlWandbConfig:
    project: str = "minionerec-sid-rl"
    mode: str = "online"
    name: str | None = None


@dataclass(frozen=True)
class MiniOneRecRlConfig:
    config_path: str
    base_model: str
    output_dir: str
    seed: int
    device: str
    data: MiniOneRecRlDataConfig
    jax: MiniOneRecRlJaxConfig = field(default_factory=MiniOneRecRlJaxConfig)
    rollout: MiniOneRecRlRolloutConfig = field(default_factory=MiniOneRecRlRolloutConfig)
    train: MiniOneRecRlTrainConfig = field(default_factory=MiniOneRecRlTrainConfig)
    eval: MiniOneRecRlEvalConfig = field(default_factory=MiniOneRecRlEvalConfig)
    wandb: MiniOneRecRlWandbConfig = field(default_factory=MiniOneRecRlWandbConfig)
    resume_from_checkpoint: str | None = None


def _set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)


def _decode_sid_triplet(tokenizer: Any, triplet: np.ndarray) -> str:
    toks = tokenizer.convert_ids_to_tokens([int(x) for x in triplet.tolist()])
    return "".join(str(t) for t in toks)


def _newline_token_id(tokenizer: Any) -> int:
    # Robust way to obtain the token for "\n" in non-initial context.
    base = list(tokenizer.encode("a", add_special_tokens=False))
    with_nl = list(tokenizer.encode("a\n", add_special_tokens=False))
    lcp = 0
    for x, y in zip(base, with_nl, strict=False):
        if int(x) != int(y):
            break
        lcp += 1
    suffix = [int(x) for x in with_nl[lcp:]]
    if len(suffix) != 1:
        raise ValueError(f"Expected single newline token id, got {suffix}")
    return int(suffix[0])


def _run_minionerec_rl_jax(cfg: MiniOneRecRlConfig, *, run_mode_norm: str) -> dict[str, Any]:
    import copy
    import math

    import flax
    from flax.training import train_state
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from MLLM_JAX.language.llama.llama import LlamaJaxConfig, convert_torch_to_flax_llama
    from MLLM_JAX.language.qwen2.modular_qwen2 import Qwen2ForCausalLM
    from MLLM_JAX.utils import get_partition_rules_llama, match_partition_rules

    from projects.sid_sft.jax.beam_search import constrained_beam_search_sid3_prefill
    from plugins.sample.optimizations import patch_qwen2_attention_decode_fast
    from plugins.training.core.mesh.mesh import create_mesh
    from plugins.training.core.step.train_step import training_step

    # Performance: avoid slow float32 matmuls in decode attention path on TPU.
    patch_qwen2_attention_decode_fast()

    def parse_dtype(name: str) -> Any:
        n = str(name or "float32").strip().lower()
        if n in {"float32", "f32"}:
            return jnp.float32
        if n in {"bfloat16", "bf16"}:
            return jnp.bfloat16
        if n in {"float16", "f16"}:
            return jnp.float16
        raise ValueError(f"Unsupported dtype: {name!r}")

    mesh = create_mesh(str(cfg.jax.mesh_shape))
    compute_dtype = parse_dtype(cfg.jax.compute_dtype)
    param_dtype = parse_dtype(cfg.jax.param_dtype)

    wandb = maybe_init_wandb(
        cfg=cfg,
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        mode=cfg.wandb.mode,
        process_index=jax.process_index(),
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    tokenizer, _pad_token_id = prepare_tokenizer(tokenizer, padding_side="right")
    extension = maybe_extend_tokenizer(tokenizer=tokenizer, sid_index_path=cfg.data.sid_index_path)

    trie = build_sid_trie_from_index(
        tokenizer=tokenizer,
        sid_index_path=cfg.data.sid_index_path,
        eos_token_id=int(getattr(tokenizer, "eos_token_id")),
    )
    newline_id = _newline_token_id(tokenizer)

    tokenizer_vocab_size = int(len(tokenizer))
    fsdp = int(mesh.shape.get("fsdp", 1))
    tp = int(mesh.shape.get("tp", 1))
    pad_multiple = max(1, fsdp * tp)
    padded_vocab_size = tokenizer_vocab_size
    if pad_multiple > 1:
        r = tokenizer_vocab_size % pad_multiple
        if r != 0:
            padded_vocab_size = tokenizer_vocab_size + (pad_multiple - r)
            if jax.process_index() == 0:
                print(f"[rl] pad_vocab_size {tokenizer_vocab_size} -> {padded_vocab_size} (multiple={pad_multiple})")

    base_config = AutoConfig.from_pretrained(cfg.base_model, trust_remote_code=True)
    ensure_rope_theta(base_config)
    base_config.vocab_size = int(padded_vocab_size)

    attention_mesh = mesh if jax.devices()[0].platform == "tpu" else None
    jax_config = LlamaJaxConfig(mesh=attention_mesh, dtype=compute_dtype, param_dtype=param_dtype)
    model = Qwen2ForCausalLM(base_config, jax_config)
    ref_model = Qwen2ForCausalLM(base_config, jax_config) if float(cfg.train.beta) != 0.0 else None

    rng = jax.random.PRNGKey(int(cfg.seed))

    # Load params:
    # - If `resume_from_checkpoint` is provided, initialize from the msgpack payload
    #   (supports both SFT checkpoints and RL checkpoints).
    # - Otherwise load HF PyTorch weights and convert to Flax/JAX params.
    if cfg.resume_from_checkpoint:
        payload = load_checkpoint(str(cfg.resume_from_checkpoint))
        params = payload.get("params")
        if params is None:
            raise ValueError("Checkpoint payload missing 'params'")
    else:
        torch_model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        params = convert_torch_to_flax_llama(torch_model.state_dict())
        params = jax.tree_util.tree_map(lambda x: np.asarray(x.detach().cpu().to(torch.float32).numpy()) if hasattr(x, "detach") else np.asarray(x), params)

    params, vocab_resize = resize_lm_vocab(params=params, new_vocab_size=int(padded_vocab_size), rng=rng)
    params = jax.tree_util.tree_map(lambda x: np.asarray(x, dtype=np.dtype(param_dtype)), params)
    shapes = jax.eval_shape(lambda x: x, params)
    partitions = match_partition_rules(get_partition_rules_llama(), shapes)
    shardings = jax.tree_util.tree_map(lambda spec: NamedSharding(mesh, spec), partitions)
    params = jax.tree_util.tree_map(lambda x, sh: jax.device_put(jnp.asarray(x, dtype=param_dtype), sh), params, shardings)

    train_stats: dict[str, Any] | None = None
    state = None
    best_metric_name = str(getattr(cfg.train, "save_best_metric", "ndcg@10") or "ndcg@10")
    best_metric_value: float | None = None
    best_metric_step: int | None = None
    best_checkpoint_path: str | None = None

    if run_mode_norm in {"train", "train_eval"}:
        os.makedirs(cfg.output_dir, exist_ok=True)

        prompt_batch = int(cfg.rollout.prompt_batch_size)
        k = int(cfg.rollout.num_generations)
        if prompt_batch <= 0 or k <= 0:
            raise ValueError("rollout.prompt_batch_size and rollout.num_generations must be > 0")

        update_accum_steps = int(getattr(cfg.train, "gradient_accumulation_steps", 1) or 1)
        if update_accum_steps <= 0:
            raise ValueError("train.gradient_accumulation_steps must be > 0")

        micro_splits = int(cfg.train.grad_accum_steps)
        if micro_splits <= 0:
            raise ValueError("train.grad_accum_steps must be > 0")

        # RL dataset: align with upstream by optionally mixing multiple prompt types.
        # Default keeps next-item-only to preserve historical behavior.
        train_datasets: list[Any] = []
        train_datasets.append(
            MiniOneRecNextItemRlDataset(
                csv_path=cfg.data.train_file,
                sample=cfg.data.sample_train,
                seed=cfg.seed,
            )
        )
        if bool(cfg.data.enable_title2sid):
            if not cfg.data.item_meta_path:
                raise ValueError("data.item_meta_path is required when data.enable_title2sid=true")
            train_datasets.append(
                MiniOneRecTitle2SidRlDataset(
                    item_meta_path=str(cfg.data.item_meta_path),
                    sid_index_path=str(cfg.data.sid_index_path),
                    sample=int(cfg.data.sample_title2sid),
                    seed=int(cfg.seed),
                    include_description=True,
                )
            )
        if bool(cfg.data.enable_seqtitle2sid):
            train_datasets.append(
                MiniOneRecSeqTitle2SidRlDataset(
                    csv_path=cfg.data.train_file,
                    sample=int(cfg.data.sample_seqtitle2sid),
                    seed=int(cfg.seed),
                )
            )

        train_dataset = train_datasets[0] if len(train_datasets) == 1 else MiniOneRecMixedRlDataset(train_datasets)
        if len(train_dataset) <= 0:
            raise ValueError("Empty train_dataset")

        # Steps: derive from epochs over prompt batches (paper uses 2 epochs).
        prompts_per_update = int(prompt_batch) * int(update_accum_steps)
        steps_per_epoch = int(math.ceil(len(train_dataset) / max(1, prompts_per_update)))
        max_steps = int(cfg.train.max_steps)
        if max_steps <= 0:
            max_steps = int(math.ceil(float(cfg.train.num_train_epochs) * steps_per_epoch))

        tx = build_tx(training_steps=max_steps, cfg=cfg.train.optimizer)

        train_module = flax.linen.remat(MiniOneRecGrpoModule, policy=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims)(
            model=model,
            ref_model=ref_model,
            beta=float(cfg.train.beta),
        )

        class _TrainState(train_state.TrainState):
            micro_step: int = 0
            micro_in_mini: int = 1
            grad_accum: Any | None = None
            ref_params: Any | None = None

        def init_fn(p):
            micro_in_mini = int(micro_splits) * int(update_accum_steps)
            grad_accum = None
            if int(micro_in_mini) > 1:
                grad_accum = jax.tree_util.tree_map(jnp.zeros_like, p)
            return _TrainState.create(
                apply_fn=train_module.apply,
                params=p,
                tx=tx,
                ref_params=copy.deepcopy(p) if float(cfg.train.beta) != 0.0 else None,
                micro_step=0,
                micro_in_mini=int(micro_in_mini),
                grad_accum=grad_accum,
            )

        state_shapes = jax.eval_shape(init_fn, params)
        state_partitions = match_partition_rules(get_partition_rules_llama(), state_shapes)
        state_shardings = jax.tree_util.tree_map(lambda spec: NamedSharding(mesh, spec), state_partitions)
        state = jax.jit(init_fn, out_shardings=state_shardings)(params)

        data_sharding_2d = NamedSharding(mesh, P(("dp", "fsdp"), None))
        data_sharding_1d = NamedSharding(mesh, P(("dp", "fsdp"),))
        train_step_fn = jax.jit(training_step, out_shardings=(state_shardings, None))

        rank_penalties = build_rank_penalties(k)
        indices = list(range(len(train_dataset)))
        rng_py = random.Random(int(cfg.seed))
        rng_py.shuffle(indices)
        cursor = 0

        best_metric_name = str(getattr(cfg.train, "save_best_metric", "ndcg@10") or "ndcg@10")
        best_metric_value: float | None = None
        best_metric_step: int | None = None
        best_checkpoint_path: str | None = None
        eval_every_steps = 0

        def _get_best_metric_value(metrics: Any) -> float:
            key = str(best_metric_name or "").strip().lower()
            if key.startswith("eval/"):
                key = key[len("eval/") :]
            if key.startswith("ndcg@"):
                kk = int(key.split("@", 1)[1])
                if int(kk) not in metrics.ndcg:
                    raise KeyError(f"{best_metric_name!r} requested but eval.topk does not include {int(kk)}")
                return float(metrics.ndcg[kk])
            if key.startswith("hr@"):
                kk = int(key.split("@", 1)[1])
                if int(kk) not in metrics.hr:
                    raise KeyError(f"{best_metric_name!r} requested but eval.topk does not include {int(kk)}")
                return float(metrics.hr[kk])
            raise ValueError(
                f"Unsupported train.save_best_metric={best_metric_name!r}; expected 'ndcg@K' or 'hr@K' (e.g. 'ndcg@10')."
            )

        # Optional full constrained-decoding eval during training (at least once per epoch).
        eval_wrapper = None
        if run_mode_norm == "train_eval" and cfg.eval.enabled:
            if int(jax.process_count()) != 1:
                print("[eval] periodic eval requires single-process JAX; skipping.")
            else:
                eval_every_steps = int(getattr(cfg.train, "eval_steps", 0) or 0)
                if eval_every_steps < 0:
                    eval_every_steps = 0
                elif eval_every_steps == 0:
                    eval_every_steps = int(steps_per_epoch)

                split = str(getattr(cfg.eval, "split", "test") or "test").strip().lower()
                if split in {"eval", "valid", "val"}:
                    eval_csv_path = str(cfg.data.eval_file)
                    eval_sample = int(cfg.data.sample_eval)
                elif split in {"test"}:
                    eval_csv_path = str(cfg.data.test_file)
                    eval_sample = int(cfg.data.sample_test)
                else:
                    raise ValueError(f"Unsupported eval.split={getattr(cfg.eval, 'split', None)!r} (expected 'eval'|'test')")

                _eval_dataset = MiniOneRecNextItemRlDataset(
                    csv_path=eval_csv_path,
                    sample=eval_sample,
                    seed=cfg.seed,
                )

                class _EvalWrapper:
                    def __len__(self):
                        return len(_eval_dataset)

                    def __getitem__(self, idx: int):
                        ids = list(tokenizer.encode(_eval_dataset[idx].prompt, add_special_tokens=False))
                        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

                    def get_targets(self):
                        return [_eval_dataset[i].target_sid for i in range(len(_eval_dataset))]

                eval_wrapper = _EvalWrapper()

        def full_eval_cb(step: int, st: Any) -> None:
            nonlocal best_metric_value, best_metric_step, best_checkpoint_path
            if eval_wrapper is None:
                return
            if int(jax.process_index()) != 0:
                return

            print(f"[eval] step={int(step)}/{int(max_steps)}")
            output_predictions_json = None
            if int(step) == int(max_steps) and bool(cfg.eval.save_predictions_json):
                output_predictions_json = os.path.join(cfg.output_dir, "eval_predictions.json")

            _preds, metrics = evaluate_sid_next_item_jax(
                model=model,
                params=st.params,
                tokenizer=tokenizer,
                eval_dataset=eval_wrapper,
                sid_index_path=cfg.data.sid_index_path,
                info_file=cfg.data.info_file,
                batch_size=int(cfg.eval.batch_size),
                num_beams=int(cfg.eval.num_beams),
                max_cache_length=int(cfg.jax.max_cache_length),
                topk=list(cfg.eval.topk),
                output_predictions_json=output_predictions_json,
            )

            if bool(getattr(cfg.train, "save_best", False)):
                value = _get_best_metric_value(metrics)
                improved = best_metric_value is None or value > float(best_metric_value)
                if improved:
                    best_metric_value = float(value)
                    best_metric_step = int(step)
                    best_checkpoint_path = save_checkpoint(output_dir=cfg.output_dir, state=st, name="rl_best")
                    print(
                        f"[eval] new best {best_metric_name}={best_metric_value:.6f} at step={int(step)} -> {best_checkpoint_path}"
                    )

            if wandb is not None:
                log = {}
                for kk, vv in metrics.hr.items():
                    log[f"eval/hr@{kk}"] = vv
                for kk, vv in metrics.ndcg.items():
                    log[f"eval/ndcg@{kk}"] = vv
                log["eval/invalid_prediction_count"] = metrics.invalid_prediction_count
                if best_metric_value is not None:
                    log["eval/best_metric_value"] = float(best_metric_value)
                if best_metric_step is not None:
                    log["eval/best_metric_step"] = int(best_metric_step)
                wandb.log(log, step=int(step))

        prompt_pad_len = int(cfg.rollout.prompt_pad_len)
        global_len = int(cfg.rollout.global_length)
        if prompt_pad_len <= 0 or global_len <= 0:
            raise ValueError("rollout.prompt_pad_len and rollout.global_length must be > 0")

        if global_len < prompt_pad_len + 8:
            raise ValueError("rollout.global_length too small for prompt_pad_len + completion")

        prompt_micro = int(getattr(cfg.rollout, "prompt_micro_batch_size", 0) or 0)
        if prompt_micro < 0:
            raise ValueError("rollout.prompt_micro_batch_size must be >= 0")
        if prompt_micro > 0 and prompt_micro > int(prompt_batch):
            raise ValueError("rollout.prompt_micro_batch_size must be <= rollout.prompt_batch_size")

        def _beam_search(params_in: Any, prompt_input_ids: jax.Array, prompt_true_lens: jax.Array):
            out = constrained_beam_search_sid3_prefill(
                model=model,
                params=params_in,
                prompt_input_ids=prompt_input_ids,
                trie=trie,
                num_beams=int(k),
                max_cache_length=int(cfg.jax.max_cache_length),
                suffix_token_ids=[int(newline_id)],
                prompt_true_len=prompt_true_lens,
            )
            return out.token_ids

        beam_search_jit = jax.jit(_beam_search)

        last_loss = float("nan")
        for step in range(1, max_steps + 1):
            micro_metrics_sum: dict[str, float] = {}
            reward_mean_sum = 0.0
            reward_rule_mean_sum = 0.0
            reward_ndcg_mean_sum = 0.0
            reward_std_sum = 0.0
            cate_diversity_sum = 0.0
            token_diversity_sum = 0.0
            pass_at_1_sum = 0.0
            pass_at_k_sum = 0.0

            pad_token_id = int(tokenizer.pad_token_id)
            eos_token_id = int(tokenizer.eos_token_id)

            # Accumulate multiple prompt-batches before each optimizer update (TRL-style
            # gradient_accumulation_steps semantics).
            for _accum in range(int(update_accum_steps)):
                # Sample prompt batch (cycle).
                if cursor + prompt_batch > len(indices):
                    rng_py.shuffle(indices)
                    cursor = 0
                batch_idx = indices[cursor : cursor + prompt_batch]
                cursor += prompt_batch

                prompts = []
                targets = []
                for i in batch_idx:
                    ex = train_dataset[i]
                    prompts.append(ex.prompt)
                    targets.append(ex.target_sid)

                prompt_ids_list = [list(tokenizer.encode(p, add_special_tokens=False))[-prompt_pad_len:] for p in prompts]
                true_lens = np.asarray([len(x) for x in prompt_ids_list], dtype=np.int32)
                prompt_np = np.full((prompt_batch, prompt_pad_len), pad_token_id, dtype=np.int32)
                for i, ids in enumerate(prompt_ids_list):
                    prompt_np[i, : len(ids)] = np.asarray(ids, dtype=np.int32)

                prompt_arr = jnp.asarray(prompt_np, dtype=jnp.int32)
                true_len_arr = jnp.asarray(true_lens, dtype=jnp.int32)

                # Constrained generation: top-K SIDs.
                if prompt_micro <= 0 or prompt_micro >= int(prompt_batch):
                    tok = np.asarray(beam_search_jit(state.params, prompt_arr, true_len_arr))  # [B, K, 3]
                else:
                    if int(prompt_batch) % int(prompt_micro) != 0:
                        raise ValueError("rollout.prompt_batch_size must be divisible by rollout.prompt_micro_batch_size")
                    tok = np.empty((int(prompt_batch), int(k), 3), dtype=np.int32)
                    for start in range(0, int(prompt_batch), int(prompt_micro)):
                        sl = slice(start, start + int(prompt_micro))
                        tok[sl] = np.asarray(beam_search_jit(state.params, prompt_arr[sl], true_len_arr[sl]))

                preds_grouped: list[list[str]] = []
                for i in range(prompt_batch):
                    preds_grouped.append([_decode_sid_triplet(tokenizer, tok[i, j]) for j in range(k)])

                rewards, correct = compute_ranking_rewards(predictions=preds_grouped, targets=targets, rank_penalties=rank_penalties)
                group_ids = np.repeat(np.arange(prompt_batch, dtype=np.int32), k)
                advantages = compute_grpo_advantages_by_group_id(rewards=rewards, group_ids=group_ids)
                rewards_rule = correct
                rewards_ndcg = rewards - correct

                # Build padded training sequences.
                completion_ids = np.zeros((prompt_batch, k, 5), dtype=np.int32)
                completion_ids[:, :, :3] = tok
                completion_ids[:, :, 3] = int(newline_id)
                completion_ids[:, :, 4] = eos_token_id
                completion_flat = completion_ids.reshape(prompt_batch * k, 5)

                input_ids = np.full((prompt_batch * k, global_len), pad_token_id, dtype=np.int32)
                attention_mask = np.zeros((prompt_batch * k, global_len), dtype=np.int32)
                labels = np.zeros((prompt_batch * k, global_len), dtype=np.int32)

                for i in range(prompt_batch):
                    p_ids = np.asarray(prompt_ids_list[i], dtype=np.int32)
                    p_len = int(p_ids.shape[0])
                    for j in range(k):
                        row = i * k + j
                        comp = completion_ids[i, j]
                        seq_len = int(p_len + comp.shape[0])
                        if seq_len > global_len:
                            # Keep the tail (align with MiniOneRec truncation style).
                            overflow = seq_len - global_len
                            p_start = min(int(overflow), p_len)
                            p_ids_trunc = p_ids[p_start:]
                            comp_trunc = comp[: max(0, global_len - int(p_ids_trunc.shape[0]))]
                            seq = np.concatenate([p_ids_trunc, comp_trunc], axis=0)
                            p_len_eff = int(p_ids_trunc.shape[0])
                            seq_len = int(seq.shape[0])
                            input_ids[row, :seq_len] = seq
                            attention_mask[row, :seq_len] = 1
                            labels[row, p_len_eff:seq_len] = 1
                        else:
                            input_ids[row, :p_len] = p_ids
                            input_ids[row, p_len:seq_len] = comp
                            attention_mask[row, :seq_len] = 1
                            labels[row, p_len:seq_len] = 1

                batch = {
                    "input_ids": jax.device_put(jnp.asarray(input_ids, dtype=jnp.int32), data_sharding_2d),
                    "attention_mask": jax.device_put(jnp.asarray(attention_mask, dtype=jnp.int32), data_sharding_2d),
                    "labels": jax.device_put(jnp.asarray(labels, dtype=jnp.int32), data_sharding_2d),
                    "advantages": jax.device_put(jnp.asarray(advantages, dtype=jnp.float32), data_sharding_1d),
                }

                # Micro-batch splitting within a single prompt-batch.
                micro = int(prompt_batch * k) // int(micro_splits)
                if micro * int(micro_splits) != int(prompt_batch * k):
                    raise ValueError("prompt_batch_size*num_generations must be divisible by train.grad_accum_steps")

                for micro_idx in range(int(micro_splits)):
                    sl = slice(micro_idx * micro, (micro_idx + 1) * micro)
                    micro_batch = {kk: vv[sl] for kk, vv in batch.items()}
                    state, metrics = train_step_fn(state, micro_batch)
                    for kk in ("loss", "kl", "completion_length"):
                        if kk not in metrics:
                            continue
                        micro_metrics_sum[kk] = micro_metrics_sum.get(kk, 0.0) + float(np.asarray(metrics[kk]))

                # Stats (accumulated across prompt-batches, averaged later).
                reward_mean_sum += float(np.mean(rewards)) if rewards.size else float("nan")
                reward_rule_mean_sum += float(np.mean(rewards_rule)) if rewards_rule.size else float("nan")
                reward_ndcg_mean_sum += float(np.mean(rewards_ndcg)) if rewards_ndcg.size else float("nan")
                reward_std_sum += (
                    float(np.mean(np.std(rewards.reshape(prompt_batch, k), axis=1))) if rewards.size else float("nan")
                )
                cate_diversity_sum += (
                    float(np.mean([len(set(group)) / float(k) for group in preds_grouped])) if preds_grouped else float("nan")
                )
                tok_flat = completion_flat.reshape(-1)
                tok_flat = tok_flat[tok_flat != pad_token_id]
                token_diversity_sum += float(len(np.unique(tok_flat)) / len(tok_flat)) if tok_flat.size else float("nan")
                pass_at_1_sum += float(np.mean(correct.reshape(prompt_batch, k)[:, 0])) if correct.size else float("nan")
                pass_at_k_sum += float(np.mean(np.max(correct.reshape(prompt_batch, k), axis=1))) if correct.size else float("nan")

            denom_micro = float(int(micro_splits) * int(update_accum_steps))
            for kk in micro_metrics_sum:
                micro_metrics_sum[kk] /= max(denom_micro, 1.0)
            last_loss = float(micro_metrics_sum.get("loss", float("nan")))

            if (
                bool(cfg.train.sync_ref_model)
                and getattr(state, "ref_params", None) is not None
                and int(cfg.train.sync_ref_model_every_steps) > 0
                and step % int(cfg.train.sync_ref_model_every_steps) == 0
            ):
                alpha = float(cfg.train.sync_ref_model_mixup_alpha)
                if alpha >= 1.0:
                    state = state.replace(ref_params=state.params)
                elif alpha > 0.0:
                    state = state.replace(
                        ref_params=jax.tree_util.tree_map(
                            lambda r, p: (1.0 - alpha) * r + alpha * p,
                            state.ref_params,
                            state.params,
                        )
                    )

            # Full constrained-decoding eval (optional).
            if eval_every_steps > 0 and (step % int(eval_every_steps) == 0 or step == max_steps):
                full_eval_cb(int(step), state)

            if step % int(cfg.train.logging_steps) == 0 or step == 1 or step == max_steps:
                denom_accum = float(int(update_accum_steps))
                reward_mean = reward_mean_sum / denom_accum
                reward_rule_mean = reward_rule_mean_sum / denom_accum
                reward_ndcg_mean = reward_ndcg_mean_sum / denom_accum
                reward_std = reward_std_sum / denom_accum
                cate_diversity = cate_diversity_sum / denom_accum
                token_diversity = token_diversity_sum / denom_accum
                pass_at_1 = pass_at_1_sum / denom_accum
                pass_at_k = pass_at_k_sum / denom_accum
                if jax.process_index() == 0:
                    kl = micro_metrics_sum.get("kl", float("nan"))
                    completion_length = micro_metrics_sum.get("completion_length", float("nan"))
                    print(
                        f"[rl] step={step}/{max_steps} loss={last_loss:.6f} kl={kl:.6f} completion_len={completion_length:.2f} "
                        f"reward_mean={reward_mean:.6f} rule_mean={reward_rule_mean:.6f} ndcg_mean={reward_ndcg_mean:.6f} "
                        f"pass@1={pass_at_1:.4f} pass@K={pass_at_k:.4f}"
                    )
                    if wandb is not None:
                        prompts_per_step = int(prompt_batch) * int(update_accum_steps)
                        completions_per_step = int(prompt_batch) * int(k) * int(update_accum_steps)
                        wandb.log(
                            {
                                "train/loss": last_loss,
                                "train/kl": kl,
                                "train/completion_length": completion_length,
                                "train/prompt_batch_size": int(prompt_batch),
                                "train/prompt_micro_batch_size": int(prompt_micro),
                                "train/num_generations": int(k),
                                "train/gradient_accumulation_steps": int(update_accum_steps),
                                "train/prompts_per_step": int(prompts_per_step),
                                "train/completions_per_step": completions_per_step,
                                "train/grad_accum_steps": int(micro_splits),
                                "train/reward_mean": reward_mean,
                                "train/reward_rule_mean": reward_rule_mean,
                                "train/reward_ndcg_mean": reward_ndcg_mean,
                                "reward": reward_mean,
                                "reward_std": reward_std,
                                "rewards/rule_reward": reward_rule_mean,
                                "rewards/ndcg_rule_reward": reward_ndcg_mean,
                                "categorical_diversity": cate_diversity,
                                "token_diversity": token_diversity,
                                "train/pass@1": pass_at_1,
                                "train/pass@K": pass_at_k,
                            },
                            step=step,
                        )

            if cfg.eval.enabled and int(cfg.eval.every_steps) > 0 and step % int(cfg.eval.every_steps) == 0:
                eval_dataset = MiniOneRecNextItemRlDataset(csv_path=cfg.data.eval_file, sample=cfg.data.sample_eval, seed=cfg.seed)
                prompts_eval = [eval_dataset[i].prompt for i in range(min(len(eval_dataset), 256))]
                targets_eval = [eval_dataset[i].target_sid for i in range(min(len(eval_dataset), 256))]
                # Lightweight sanity eval: reuse ranking reward on a small slice.
                # Full HR/NDCG eval is done at the end via `evaluate_sid_next_item_jax`.
                prompt_ids_eval = [list(tokenizer.encode(p, add_special_tokens=False))[-prompt_pad_len:] for p in prompts_eval]
                true_lens_eval = np.asarray([len(x) for x in prompt_ids_eval], dtype=np.int32)
                prompt_np_eval = np.full((len(prompts_eval), prompt_pad_len), pad_token_id, dtype=np.int32)
                for i, ids in enumerate(prompt_ids_eval):
                    prompt_np_eval[i, : len(ids)] = np.asarray(ids, dtype=np.int32)
                tok_eval = np.asarray(beam_search_jit(state.params, jnp.asarray(prompt_np_eval, dtype=jnp.int32), jnp.asarray(true_lens_eval, dtype=jnp.int32)))
                preds_eval = [[_decode_sid_triplet(tokenizer, tok_eval[i, j]) for j in range(k)] for i in range(len(prompts_eval))]
                rewards_eval, _correct_eval = compute_ranking_rewards(predictions=preds_eval, targets=targets_eval, rank_penalties=rank_penalties)
                if wandb is not None and jax.process_index() == 0:
                    wandb.log({"eval_small/reward_mean": float(np.mean(rewards_eval))}, step=step)

        train_stats = {"steps": int(max_steps), "final_loss": float(last_loss)}

        tokenizer.save_pretrained(cfg.output_dir)
        if bool(cfg.train.save_last) and jax.process_index() == 0:
            save_checkpoint(output_dir=cfg.output_dir, state=state, name="rl_last")

    # Eval (full HR/NDCG) on configured split.
    eval_metrics = None
    if run_mode_norm in {"eval", "train_eval"} and cfg.eval.enabled:
        split = str(getattr(cfg.eval, "split", "test") or "test").strip().lower()
        if split in {"eval", "valid", "val"}:
            csv_path = cfg.data.eval_file
            sample = cfg.data.sample_eval
        elif split in {"test"}:
            csv_path = cfg.data.test_file
            sample = cfg.data.sample_test
        else:
            raise ValueError(f"Unsupported eval.split={getattr(cfg.eval, 'split', None)!r} (expected 'eval'|'test')")

        eval_dataset = MiniOneRecNextItemRlDataset(
            csv_path=str(csv_path),
            sample=int(sample),
            seed=cfg.seed,
        )

        # Adapt the RL dataset into the evaluator contract.
        class _EvalWrapper:
            def __len__(self):
                return len(eval_dataset)

            def __getitem__(self, idx: int):
                # Evaluator expects {"input_ids": ...}
                ids = list(tokenizer.encode(eval_dataset[idx].prompt, add_special_tokens=False))
                return {"input_ids": ids, "attention_mask": [1] * len(ids)}

            def get_targets(self):
                return [eval_dataset[i].target_sid for i in range(len(eval_dataset))]

        output_predictions_json = os.path.join(cfg.output_dir, "eval_predictions.json") if cfg.eval.save_predictions_json else None
        eval_params = state.params if state is not None else params
        eval_step = int(getattr(state, "step", 0) or 0)
        if (
            run_mode_norm == "train_eval"
            and bool(getattr(cfg.eval, "use_best_checkpoint", False))
            and best_checkpoint_path is not None
            and os.path.exists(best_checkpoint_path)
        ):
            payload = load_checkpoint(best_checkpoint_path)
            ckpt_params = payload.get("params")
            if ckpt_params is None:
                raise ValueError("Best checkpoint payload missing 'params'")
            ckpt_params = jax.tree_util.tree_map(lambda x: np.asarray(x, dtype=np.dtype(param_dtype)), ckpt_params)
            eval_params = jax.tree_util.tree_map(
                lambda x, sh: jax.device_put(jnp.asarray(x, dtype=param_dtype), sh),
                ckpt_params,
                shardings,
            )
            eval_step = int(best_metric_step or eval_step)

        _preds, eval_metrics = evaluate_sid_next_item_jax(
            model=model,
            params=eval_params,
            tokenizer=tokenizer,
            eval_dataset=_EvalWrapper(),
            sid_index_path=cfg.data.sid_index_path,
            info_file=cfg.data.info_file,
            batch_size=int(cfg.eval.batch_size),
            num_beams=int(cfg.eval.num_beams),
            max_cache_length=int(cfg.jax.max_cache_length),
            topk=list(cfg.eval.topk),
            output_predictions_json=output_predictions_json,
        )
        if wandb is not None and jax.process_index() == 0:
            log = {}
            for kk, vv in eval_metrics.hr.items():
                log[f"eval/hr@{kk}"] = vv
            for kk, vv in eval_metrics.ndcg.items():
                log[f"eval/ndcg@{kk}"] = vv
            log["eval/invalid_prediction_count"] = eval_metrics.invalid_prediction_count
            wandb.log(log, step=eval_step)

    if wandb is not None:
        wandb.finish()

    return {
        "config": asdict(cfg),
        "token_extension": asdict(extension),
        "vocab_resize": asdict(vocab_resize),
        "train": train_stats,
        "eval": asdict(eval_metrics) if eval_metrics else None,
        "best_checkpoint": {
            "metric": str(best_metric_name),
            "value": best_metric_value,
            "step": best_metric_step,
            "path": best_checkpoint_path,
        }
        if bool(getattr(cfg.train, "save_best", False))
        else None,
    }


def run_minionerec_rl(cfg: MiniOneRecRlConfig, *, run_mode: str) -> dict[str, Any]:
    _set_seed(cfg.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    run_mode_norm = str(run_mode).strip().lower()
    if run_mode_norm not in {"train", "eval", "train_eval"}:
        raise ValueError("run_mode must be one of: train|eval|train_eval")

    backend = str(cfg.device or "tpu").strip().lower()
    if backend not in {"tpu", "jax"}:
        raise ValueError("MiniOneRec RL runner currently supports only JAX/TPU backend")

    return _run_minionerec_rl_jax(cfg, run_mode_norm=run_mode_norm)


__all__ = ["MiniOneRecRlConfig", "run_minionerec_rl"]
