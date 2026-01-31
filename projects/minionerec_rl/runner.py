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
from projects.minionerec_rl.datasets import MiniOneRecNextItemRlDataset
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
    num_generations: int = 16
    prompt_pad_len: int = 256  # fixed prompt padding (no length bucketing)
    global_length: int = 512  # fixed training padding


@dataclass(frozen=True)
class MiniOneRecRlTrainConfig:
    num_train_epochs: float = 2.0
    max_steps: int = -1
    grad_accum_steps: int = 1
    ppo_steps: int = 1
    beta: float = 1e-3
    logging_steps: int = 10
    save_last: bool = True
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass(frozen=True)
class MiniOneRecRlEvalConfig:
    enabled: bool = True
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
    from plugins.training.core.mesh.mesh import create_mesh
    from plugins.training.core.step.train_step import training_step

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

    if run_mode_norm in {"train", "train_eval"}:
        os.makedirs(cfg.output_dir, exist_ok=True)

        prompt_batch = int(cfg.rollout.prompt_batch_size)
        k = int(cfg.rollout.num_generations)
        if prompt_batch <= 0 or k <= 0:
            raise ValueError("rollout.prompt_batch_size and rollout.num_generations must be > 0")

        # RL dataset: next-item only (SID space).
        train_dataset = MiniOneRecNextItemRlDataset(
            csv_path=cfg.data.train_file,
            sample=cfg.data.sample_train,
            seed=cfg.seed,
        )
        if len(train_dataset) <= 0:
            raise ValueError("Empty train_dataset")

        # Steps: derive from epochs over prompt batches (paper uses 2 epochs).
        max_steps = int(cfg.train.max_steps)
        if max_steps <= 0:
            steps_per_epoch = int(math.ceil(len(train_dataset) / max(1, prompt_batch)))
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
            grad_accum = None
            if int(cfg.train.grad_accum_steps) > 1:
                grad_accum = jax.tree_util.tree_map(jnp.zeros_like, p)
            return _TrainState.create(
                apply_fn=train_module.apply,
                params=p,
                tx=tx,
                ref_params=copy.deepcopy(p) if float(cfg.train.beta) != 0.0 else None,
                micro_step=0,
                micro_in_mini=int(cfg.train.grad_accum_steps),
                grad_accum=grad_accum,
            )

        state_shapes = jax.eval_shape(init_fn, params)
        state_partitions = match_partition_rules(get_partition_rules_llama(), state_shapes)
        state_shardings = jax.tree_util.tree_map(lambda spec: NamedSharding(mesh, spec), state_partitions)
        state = jax.jit(init_fn, out_shardings=state_shardings)(params)

        data_sharding_2d = NamedSharding(mesh, P(("dp", "fsdp"), None))
        data_sharding_1d = NamedSharding(mesh, P(("dp", "fsdp"),))
        train_step_fn = jax.jit(training_step, donate_argnums=(0,), out_shardings=(state_shardings, None))

        rank_penalties = build_rank_penalties(k)
        indices = list(range(len(train_dataset)))
        rng_py = random.Random(int(cfg.seed))
        rng_py.shuffle(indices)
        cursor = 0

        prompt_pad_len = int(cfg.rollout.prompt_pad_len)
        global_len = int(cfg.rollout.global_length)
        if prompt_pad_len <= 0 or global_len <= 0:
            raise ValueError("rollout.prompt_pad_len and rollout.global_length must be > 0")

        if global_len < prompt_pad_len + 8:
            raise ValueError("rollout.global_length too small for prompt_pad_len + completion")

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
            pad_token_id = int(tokenizer.pad_token_id)
            prompt_np = np.full((prompt_batch, prompt_pad_len), pad_token_id, dtype=np.int32)
            for i, ids in enumerate(prompt_ids_list):
                prompt_np[i, : len(ids)] = np.asarray(ids, dtype=np.int32)

            prompt_arr = jnp.asarray(prompt_np, dtype=jnp.int32)
            true_len_arr = jnp.asarray(true_lens, dtype=jnp.int32)

            # Constrained generation: top-K SIDs.
            tok = np.asarray(beam_search_jit(state.params, prompt_arr, true_len_arr))  # [B, K, 3]

            preds_grouped: list[list[str]] = []
            for i in range(prompt_batch):
                preds_grouped.append([_decode_sid_triplet(tokenizer, tok[i, j]) for j in range(k)])

            rewards, correct = compute_ranking_rewards(predictions=preds_grouped, targets=targets, rank_penalties=rank_penalties)
            group_ids = np.repeat(np.arange(prompt_batch, dtype=np.int32), k)
            advantages = compute_grpo_advantages_by_group_id(rewards=rewards, group_ids=group_ids)

            # Build padded training sequences.
            completion_ids = np.zeros((prompt_batch, k, 5), dtype=np.int32)
            completion_ids[:, :, :3] = tok
            completion_ids[:, :, 3] = int(newline_id)
            completion_ids[:, :, 4] = int(tokenizer.eos_token_id)
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

            # Gradient accumulation over the full (prompt_batch*k) batch.
            micro = int(prompt_batch * k) // int(cfg.train.grad_accum_steps)
            if micro * int(cfg.train.grad_accum_steps) != int(prompt_batch * k):
                raise ValueError("prompt_batch_size*num_generations must be divisible by grad_accum_steps")

            for micro_idx in range(int(cfg.train.grad_accum_steps)):
                sl = slice(micro_idx * micro, (micro_idx + 1) * micro)
                micro_batch = {kk: vv[sl] for kk, vv in batch.items()}
                state, metrics = train_step_fn(state, micro_batch)
                last_loss = float(np.asarray(metrics["loss"]))

            if step % int(cfg.train.logging_steps) == 0 or step == 1 or step == max_steps:
                reward_mean = float(np.mean(rewards)) if rewards.size else float("nan")
                pass_at_1 = float(np.mean(correct.reshape(prompt_batch, k)[:, 0])) if correct.size else float("nan")
                pass_at_k = float(np.mean(np.max(correct.reshape(prompt_batch, k), axis=1))) if correct.size else float("nan")
                if jax.process_index() == 0:
                    print(f"[rl] step={step}/{max_steps} loss={last_loss:.6f} reward_mean={reward_mean:.6f} pass@1={pass_at_1:.4f} pass@K={pass_at_k:.4f}")
                    if wandb is not None:
                        wandb.log(
                            {
                                "train/loss": last_loss,
                                "train/reward_mean": reward_mean,
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

    # Eval (full test HR/NDCG) on final params.
    eval_metrics = None
    if run_mode_norm in {"eval", "train_eval"} and cfg.eval.enabled:
        eval_dataset = MiniOneRecNextItemRlDataset(
            csv_path=cfg.data.test_file,
            sample=cfg.data.sample_test,
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
        _preds, eval_metrics = evaluate_sid_next_item_jax(
            model=model,
            params=state.params if state is not None else params,
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
            wandb.log(log)

    if wandb is not None:
        wandb.finish()

    return {
        "config": asdict(cfg),
        "token_extension": asdict(extension),
        "vocab_resize": asdict(vocab_resize),
        "train": train_stats,
        "eval": asdict(eval_metrics) if eval_metrics else None,
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
