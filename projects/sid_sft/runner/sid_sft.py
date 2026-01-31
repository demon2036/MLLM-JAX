from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from projects.sid_sft.datasets.concat_dataset import ConcatDataset
from projects.sid_sft.datasets.eval_sid_next_item import SidNextItemEvalDataset
from projects.sid_sft.datasets.fusion_seq_rec import FusionSeqRecSftDataset
from projects.sid_sft.datasets.sid_item_alignment import SidItemAlignmentDataset
from projects.sid_sft.datasets.sid_next_item import SidNextItemSftDataset
from projects.sid_sft.jax.evaluator import SidNextItemJaxEvaluator, evaluate_sid_next_item_jax
from projects.sid_sft.tokens import maybe_extend_tokenizer
from plugins.training.core.logging.wandb import maybe_init_wandb
from plugins.training.core.tokenizer import prepare_tokenizer


def _set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)


@dataclass(frozen=True)
class SidSftDataConfig:
    category: str
    train_file: str
    eval_file: str
    test_file: str
    info_file: str
    sid_index_path: str
    item_meta_path: str
    max_len: int = 512
    sample_train: int = -1
    sample_eval: int = -1
    sample_test: int = -1


@dataclass(frozen=True)
class SidSftTasksConfig:
    sid_next_item: bool = True
    sid_item_alignment: bool = True
    fusion_seq_rec: bool = True


@dataclass(frozen=True)
class SidSftMuonConfig:
    aux_learning_rate: float = 3e-4
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    eps: float = 1e-7
    max_dim: int = 10_000


@dataclass(frozen=True)
class SidSftTrainConfig:
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    global_batch_size: int = 0
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    optimizer: str = "adamw"
    muon: SidSftMuonConfig = field(default_factory=SidSftMuonConfig)
    weight_decay: float = 0.0
    num_train_epochs: float = 1.0
    max_steps: int = -1
    warmup_steps: int = 0
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 1
    save_last: bool = True
    # Optional: save the best checkpoint according to constrained-decoding eval
    # metrics (computed on `cfg.eval.split`).
    #
    # When enabled, the best checkpoint is written as `sft_state_best.msgpack`
    # in `output_dir`.
    save_best: bool = False
    # Metric name used for "best" selection, e.g.:
    # - "ndcg@10" (default)
    # - "hr@10"
    # - "eval/ndcg@10" (prefix is allowed)
    save_best_metric: str = "ndcg@10"
    group_by_length: bool = False
    freeze_LLM: bool = False
    train_from_scratch: bool = False
    resume_from_checkpoint: str | None = None
    early_stopping_patience: int = 3
    bf16: bool = False
    fp16: bool = False


@dataclass(frozen=True)
class SidSftEvalConfig:
    enabled: bool = True
    # Which split to use for constrained-decoding evaluation:
    # - "test": use `data.test_file` (default; preserves historical behavior)
    # - "eval": use `data.eval_file` (validation)
    split: str = "test"
    # If enabled and `train.save_best=true`, `train_eval` will load and evaluate
    # the best checkpoint saved during training (instead of the last-step params).
    #
    # When `train.save_best=false` (or the best checkpoint does not exist), this
    # flag is ignored and evaluation uses last-step params.
    use_best_checkpoint: bool = False
    batch_size: int = 4
    num_beams: int = 50
    max_new_tokens: int = 64
    length_penalty: float = 0.0
    topk: tuple[int, ...] = (1, 3, 5, 10, 20, 50)
    constrained: bool = True
    save_predictions_json: bool = True
    # Constrained decoding prefill strategy:
    # - "bucket": sampler-style prefill buckets (default; can compile per bucket)
    # - "fixed": single fixed prefill length across the whole dataset (compile once)
    prefill_mode: str = "bucket"
    # Only used when prefill_mode="fixed" (or for explicit control). If None,
    # the evaluator derives a safe fixed prefill length from the dataset.
    fixed_prefill_len: int | None = None


@dataclass(frozen=True)
class SidSftWandbConfig:
    project: str = "minionerec-sid-sft"
    mode: str = "online"
    name: str | None = None


@dataclass(frozen=True)
class SidSftJaxConfig:
    mesh_shape: str = "1,-1,1"
    param_dtype: str = "float32"
    compute_dtype: str = "bfloat16"
    max_cache_length: int = 2048


@dataclass(frozen=True)
class SidSftConfig:
    config_path: str
    backend: str
    base_model: str
    output_dir: str
    seed: int
    device: str
    data: SidSftDataConfig
    jax: SidSftJaxConfig = field(default_factory=SidSftJaxConfig)
    tasks: SidSftTasksConfig = field(default_factory=SidSftTasksConfig)
    train: SidSftTrainConfig = field(default_factory=SidSftTrainConfig)
    eval: SidSftEvalConfig = field(default_factory=SidSftEvalConfig)
    wandb: SidSftWandbConfig = field(default_factory=SidSftWandbConfig)


def _build_train_dataset(cfg: SidSftConfig, tokenizer: Any) -> Any:
    train_datasets = []
    if cfg.tasks.sid_next_item:
        train_datasets.append(
            SidNextItemSftDataset(
                csv_path=cfg.data.train_file,
                tokenizer=tokenizer,
                max_len=cfg.data.max_len,
                sample=cfg.data.sample_train,
                seed=cfg.seed,
                include_labels=True,
                pretokenize=True,
            )
        )
    if cfg.tasks.sid_item_alignment:
        train_datasets.append(
            SidItemAlignmentDataset(
                item_meta_path=cfg.data.item_meta_path,
                sid_index_path=cfg.data.sid_index_path,
                tokenizer=tokenizer,
                max_len=cfg.data.max_len,
                sample=cfg.data.sample_train,
                seed=cfg.seed,
                include_labels=True,
                pretokenize=True,
            )
        )
    if cfg.tasks.fusion_seq_rec:
        train_datasets.append(
            FusionSeqRecSftDataset(
                csv_path=cfg.data.train_file,
                item_meta_path=cfg.data.item_meta_path,
                sid_index_path=cfg.data.sid_index_path,
                tokenizer=tokenizer,
                max_len=cfg.data.max_len,
                sample=cfg.data.sample_train,
                seed=cfg.seed,
                include_labels=True,
                pretokenize=True,
            )
        )
    if not train_datasets:
        raise ValueError("No training tasks enabled under cfg.tasks.*")
    return train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)


def _build_eval_dataset(cfg: SidSftConfig, tokenizer: Any) -> SidNextItemEvalDataset:
    split = str(getattr(cfg.eval, "split", "test") or "test").strip().lower()
    if split in {"eval", "valid", "val"}:
        csv_path = cfg.data.eval_file
        sample = cfg.data.sample_eval
    elif split in {"test"}:
        csv_path = cfg.data.test_file
        sample = cfg.data.sample_test
    else:
        raise ValueError(f"Unsupported eval.split={cfg.eval.split!r} (expected 'eval'|'test')")
    return SidNextItemEvalDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        max_len=cfg.data.max_len,
        sample=sample,
        seed=cfg.seed,
        pretokenize=True,
    )


def _run_sid_sft_jax(cfg: SidSftConfig, *, run_mode_norm: str) -> dict[str, Any]:
    import math

    import flax
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding

    from MLLM_JAX.language.llama.llama import LlamaJaxConfig, convert_torch_to_flax_llama
    from MLLM_JAX.language.qwen2.modular_qwen2 import Qwen2ForCausalLM
    from MLLM_JAX.utils import get_partition_rules_llama, match_partition_rules

    from projects.sid_sft.jax.checkpoint import load_checkpoint, save_checkpoint
    from projects.sid_sft.jax.params import resize_lm_vocab
    from projects.sid_sft.jax.train import create_mesh_from_config, run_sft_train

    best_metric_name = str(cfg.train.save_best_metric or "ndcg@10")
    best_metric_value: float | None = None
    best_metric_step: int | None = None
    best_checkpoint_path: str | None = None

    def parse_dtype(name: str) -> Any:
        n = str(name or "float32").strip().lower()
        if n in {"float32", "f32"}:
            return jnp.float32
        if n in {"bfloat16", "bf16"}:
            return jnp.bfloat16
        if n in {"float16", "f16"}:
            return jnp.float16
        raise ValueError(f"Unsupported dtype: {name!r}")

    mesh = create_mesh_from_config(cfg.jax.mesh_shape)
    compute_dtype = parse_dtype(cfg.jax.compute_dtype)
    param_dtype = parse_dtype(cfg.jax.param_dtype)

    wandb = maybe_init_wandb(
        cfg=cfg,
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        mode=cfg.wandb.mode,
        process_index=jax.process_index(),
    )

    # Tokenizer + SID token extension (tokenizer-only; params resized below).
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    tokenizer, pad_token_id = prepare_tokenizer(tokenizer, padding_side="right")

    extension = maybe_extend_tokenizer(tokenizer=tokenizer, sid_index_path=cfg.data.sid_index_path)

    tokenizer_vocab_size = int(len(tokenizer))
    # When using FSDP/TP sharding, some params shard over the vocab axis. Pad the
    # model vocab size so the embedding table shards evenly.
    fsdp = int(mesh.shape.get("fsdp", 1))
    tp = int(mesh.shape.get("tp", 1))
    pad_multiple = max(1, fsdp * tp)
    padded_vocab_size = tokenizer_vocab_size
    if pad_multiple > 1:
        r = tokenizer_vocab_size % pad_multiple
        if r != 0:
            padded_vocab_size = tokenizer_vocab_size + (pad_multiple - r)
            if jax.process_index() == 0:
                print(f"[sft] pad_vocab_size {tokenizer_vocab_size} -> {padded_vocab_size} (multiple={pad_multiple})")

    # Model config must reflect the resized vocab (tokenizer + optional padding).
    base_config = AutoConfig.from_pretrained(cfg.base_model, trust_remote_code=True)
    from plugins.training.core.io.hf_config import ensure_rope_theta

    ensure_rope_theta(base_config)
    base_config.vocab_size = int(padded_vocab_size)

    # Avoid TPU-only fused attention kernels on CPU/GPU backends.
    attention_mesh = mesh if jax.devices()[0].platform == "tpu" else None
    jax_config = LlamaJaxConfig(mesh=attention_mesh, dtype=compute_dtype, param_dtype=param_dtype)
    model = Qwen2ForCausalLM(base_config, jax_config)

    rng = jax.random.PRNGKey(int(cfg.seed))

    # Param init / load:
    # - For eval-only runs with `resume_from_checkpoint`, skip loading base model weights.
    # - Otherwise, load from base model or init from scratch.
    loaded_from_checkpoint = False
    if run_mode_norm == "eval" and cfg.train.resume_from_checkpoint:
        payload = load_checkpoint(str(cfg.train.resume_from_checkpoint))
        ckpt_params = payload.get("params")
        if ckpt_params is None:
            raise ValueError("Checkpoint payload missing 'params'")
        params = ckpt_params
        loaded_from_checkpoint = True
    elif cfg.train.train_from_scratch:
        dummy_len = 8
        dummy = jnp.zeros((1, dummy_len), dtype=jnp.int32)
        dummy_mask = jnp.ones((1, dummy_len), dtype=jnp.int32)
        dummy_pos = jnp.arange(dummy_len, dtype=jnp.int32)[None, :]
        variables = model.init(rng, input_ids=dummy, attention_mask=dummy_mask, position_ids=dummy_pos, cache=None)
        params = flax.core.unfreeze(variables["params"])
    else:
        torch_model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        state_dict = torch_model.state_dict()
        params = convert_torch_to_flax_llama(state_dict)

        def _to_numpy(x: Any) -> np.ndarray:
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu()
                # PyTorch cannot export bfloat16 tensors to NumPy directly.
                if x.dtype == torch.bfloat16:
                    x = x.to(torch.float32)
                return x.numpy()
            return np.asarray(x)

        params = jax.tree_util.tree_map(_to_numpy, params)
        del torch_model

    # Resize embeddings/lm_head for new SID tokens (+ optional padding for sharding divisibility).
    params, vocab_resize = resize_lm_vocab(params=params, new_vocab_size=int(padded_vocab_size), rng=rng)

    # Place params with sharding + dtype.
    params = jax.tree_util.tree_map(lambda x: np.asarray(x, dtype=np.dtype(param_dtype)), params)
    shapes = jax.eval_shape(lambda x: x, params)
    partitions = match_partition_rules(get_partition_rules_llama(), shapes)
    shardings = jax.tree_util.tree_map(lambda spec: NamedSharding(mesh, spec), partitions)
    params = jax.tree_util.tree_map(lambda x, sh: jax.device_put(jnp.asarray(x, dtype=param_dtype), sh), params, shardings)

    # Train (optional).
    state = None
    train_stats = None
    eval_metrics = None
    if run_mode_norm in {"train", "train_eval"}:
        train_dataset = _build_train_dataset(cfg, tokenizer)

        replicas = int(mesh.shape.get("dp", 1)) * int(mesh.shape.get("fsdp", 1))
        micro_per_replica = int(cfg.train.per_device_train_batch_size)
        if replicas <= 0 or micro_per_replica <= 0:
            raise ValueError(f"Invalid replicas={replicas} or micro_batch_size_per_replica={micro_per_replica}")

        global_batch_size = int(getattr(cfg.train, "global_batch_size", 0) or 0)
        grad_accum_steps = int(cfg.train.gradient_accumulation_steps)
        if global_batch_size > 0:
            micro_global = micro_per_replica * replicas
            ga = int(global_batch_size) // int(micro_global) if micro_global > 0 else 1
            if ga < 1:
                ga = 1
            if jax.process_index() == 0:
                effective = int(micro_global) * int(ga)
                if effective != int(global_batch_size):
                    print(
                        f"[sft] global_batch_size={global_batch_size} not divisible by micro_global={micro_global}; "
                        f"using grad_accum_steps={ga} (effective_bs={effective})"
                    )
                else:
                    print(f"[sft] auto grad_accum_steps={ga} (effective_bs={effective})")
            grad_accum_steps = ga

        max_steps = int(cfg.train.max_steps)
        micro = micro_per_replica * replicas
        effective = micro * int(grad_accum_steps)
        steps_per_epoch = int(math.ceil(len(train_dataset) / max(1, int(effective))))

        if max_steps <= 0:
            max_steps = int(math.ceil(float(cfg.train.num_train_epochs) * steps_per_epoch))

        save_steps = int(cfg.train.save_steps)
        save_total_limit = int(cfg.train.save_total_limit)
        saved_steps: list[int] = []

        def checkpoint_cb(step: int, st: Any) -> None:
            if jax.process_index() != 0:
                return
            save_checkpoint(output_dir=cfg.output_dir, state=st, name=f"step{int(step)}")
            if save_total_limit > 0:
                saved_steps.append(int(step))
                while len(saved_steps) > int(save_total_limit):
                    old = saved_steps.pop(0)
                    try:
                        os.remove(os.path.join(cfg.output_dir, f"sft_state_step{int(old)}.msgpack"))
                    except FileNotFoundError:
                        pass

        evaluator = None
        eval_every_steps = 0

        def _get_best_metric_value(metrics: Any) -> float:
            key = str(best_metric_name or "").strip().lower()
            if key.startswith("eval/"):
                key = key[len("eval/") :]
            if key.startswith("ndcg@"):
                k = int(key.split("@", 1)[1])
                if int(k) not in metrics.ndcg:
                    raise KeyError(f"{best_metric_name!r} requested but eval.topk does not include {int(k)}")
                return float(metrics.ndcg[k])
            if key.startswith("hr@"):
                k = int(key.split("@", 1)[1])
                if int(k) not in metrics.hr:
                    raise KeyError(f"{best_metric_name!r} requested but eval.topk does not include {int(k)}")
                return float(metrics.hr[k])
            raise ValueError(
                f"Unsupported train.save_best_metric={best_metric_name!r}; expected 'ndcg@K' or 'hr@K' (e.g. 'ndcg@10')."
            )

        # For `train_eval`, run constrained-decoding eval periodically so W&B
        # shows metrics during training (at least once per epoch).
        if run_mode_norm == "train_eval" and cfg.eval.enabled:
            if int(jax.process_count()) != 1:
                print("[eval] periodic eval requires single-process JAX; skipping epoch eval.")
            else:
                eval_dataset = _build_eval_dataset(cfg, tokenizer)
                evaluator = SidNextItemJaxEvaluator(
                    model=model,
                    tokenizer=tokenizer,
                    eval_dataset=eval_dataset,
                    sid_index_path=cfg.data.sid_index_path,
                    info_file=cfg.data.info_file,
                    batch_size=cfg.eval.batch_size,
                    num_beams=cfg.eval.num_beams,
                    max_cache_length=cfg.jax.max_cache_length,
                    topk=list(cfg.eval.topk),
                    show_progress=False,
                    prefill_mode=cfg.eval.prefill_mode,
                    fixed_prefill_len=cfg.eval.fixed_prefill_len,
                )
                eval_every_steps = int(cfg.train.eval_steps)
                if eval_every_steps < 0:
                    eval_every_steps = 0
                elif eval_every_steps == 0:
                    eval_every_steps = int(steps_per_epoch)

        def eval_cb(step: int, st: Any) -> None:
            nonlocal eval_metrics
            nonlocal best_metric_value, best_metric_step, best_checkpoint_path
            if evaluator is None:
                return
            if int(jax.process_index()) != 0:
                return

            epoch = float(step) / max(1.0, float(steps_per_epoch))
            print(f"[eval] step={int(step)}/{int(max_steps)} epoch={epoch:.2f}")

            output_predictions_json = None
            if int(step) == int(max_steps) and bool(cfg.eval.save_predictions_json):
                output_predictions_json = os.path.join(cfg.output_dir, "eval_predictions.json")

            _preds, metrics = evaluator.evaluate(
                params=st.params,
                output_predictions_json=output_predictions_json,
            )
            eval_metrics = metrics

            if bool(cfg.train.save_best):
                value = _get_best_metric_value(metrics)
                improved = best_metric_value is None or value > float(best_metric_value)
                if improved:
                    best_metric_value = float(value)
                    best_metric_step = int(step)
                    best_checkpoint_path = save_checkpoint(output_dir=cfg.output_dir, state=st, name="best")
                    print(
                        f"[eval] new best {best_metric_name}={best_metric_value:.6f} at step={int(step)} "
                        f"-> {best_checkpoint_path}"
                    )

            if wandb is not None:
                log = {"eval/epoch": epoch}
                for k, v in metrics.hr.items():
                    log[f"eval/hr@{k}"] = v
                for k, v in metrics.ndcg.items():
                    log[f"eval/ndcg@{k}"] = v
                log["eval/invalid_prediction_count"] = metrics.invalid_prediction_count
                if best_metric_value is not None:
                    log["eval/best_metric_value"] = float(best_metric_value)
                if best_metric_step is not None:
                    log["eval/best_metric_step"] = int(best_metric_step)
                wandb.log(log, step=int(step))

        state, train_stats = run_sft_train(
            mesh=mesh,
            model=model,
            params=params,
            train_dataset=train_dataset,
            pad_token_id=int(pad_token_id),
            pad_to_length=int(cfg.data.max_len) if int(cfg.data.max_len) > 0 else None,
            optimizer_name=cfg.train.optimizer,
            learning_rate=float(cfg.train.learning_rate),
            weight_decay=float(cfg.train.weight_decay),
            muon_aux_learning_rate=float(cfg.train.muon.aux_learning_rate),
            muon_momentum=float(cfg.train.muon.momentum),
            muon_nesterov=bool(cfg.train.muon.nesterov),
            muon_ns_steps=int(cfg.train.muon.ns_steps),
            muon_eps=float(cfg.train.muon.eps),
            muon_max_dim=int(cfg.train.muon.max_dim),
            grad_accum_steps=int(grad_accum_steps),
            micro_batch_size_per_replica=int(cfg.train.per_device_train_batch_size),
            max_steps=int(max_steps),
            seed=int(cfg.seed),
            logging_steps=int(cfg.train.logging_steps),
            warmup_steps=int(cfg.train.warmup_steps),
            log_cb=(
                (
                    lambda step, loss, effective_bs, step_time_sec: wandb.log(
                        {
                            "train/loss": loss,
                            "train/effective_batch_size": effective_bs,
                            "train/step_time_sec": float(step_time_sec),
                            "train/samples_per_sec": float(effective_bs) / max(float(step_time_sec), 1e-9),
                        },
                        step=step,
                    )
                )
                if wandb is not None
                else None
            ),
            eval_every_steps=int(eval_every_steps),
            eval_cb=(eval_cb if eval_every_steps > 0 else None),
            checkpoint_every_steps=int(save_steps),
            checkpoint_cb=(checkpoint_cb if save_steps > 0 else None),
        )

        os.makedirs(cfg.output_dir, exist_ok=True)
        tokenizer.save_pretrained(cfg.output_dir)
        if bool(cfg.train.save_last) and jax.process_index() == 0:
            save_checkpoint(output_dir=cfg.output_dir, state=state, name="last")

    # Eval params: prefer trained state, else use placed params.
    eval_params = state.params if state is not None else params
    # For non-eval-only runs, allow eval to use a checkpoint if no trained state is available.
    if state is None and cfg.train.resume_from_checkpoint and not loaded_from_checkpoint:
        payload = load_checkpoint(str(cfg.train.resume_from_checkpoint))
        ckpt_params = payload.get("params")
        if ckpt_params is not None:
            ckpt_params = jax.tree_util.tree_map(lambda x: np.asarray(x, dtype=np.dtype(param_dtype)), ckpt_params)
            eval_params = jax.tree_util.tree_map(lambda x, sh: jax.device_put(jnp.asarray(x, dtype=param_dtype), sh), ckpt_params, shardings)
    if (
        run_mode_norm == "train_eval"
        and cfg.eval.enabled
        and bool(getattr(cfg.eval, "use_best_checkpoint", False))
        and best_checkpoint_path is not None
        and os.path.exists(best_checkpoint_path)
    ):
        payload = load_checkpoint(best_checkpoint_path)
        ckpt_params = payload.get("params")
        if ckpt_params is not None:
            ckpt_params = jax.tree_util.tree_map(lambda x: np.asarray(x, dtype=np.dtype(param_dtype)), ckpt_params)
            eval_params = jax.tree_util.tree_map(lambda x, sh: jax.device_put(jnp.asarray(x, dtype=param_dtype), sh), ckpt_params, shardings)

    if run_mode_norm == "train_eval" and cfg.eval.enabled and eval_metrics is None:
        if not cfg.eval.constrained:
            raise NotImplementedError("JAX evaluator currently supports only constrained=true (SID trie).")
        eval_dataset = _build_eval_dataset(cfg, tokenizer)
        output_predictions_json = os.path.join(cfg.output_dir, "eval_predictions.json") if cfg.eval.save_predictions_json else None
        _preds, eval_metrics = evaluate_sid_next_item_jax(
            model=model,
            params=eval_params,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            sid_index_path=cfg.data.sid_index_path,
            info_file=cfg.data.info_file,
            batch_size=cfg.eval.batch_size,
            num_beams=cfg.eval.num_beams,
            max_cache_length=cfg.jax.max_cache_length,
            topk=list(cfg.eval.topk),
            output_predictions_json=output_predictions_json,
            prefill_mode=cfg.eval.prefill_mode,
            fixed_prefill_len=cfg.eval.fixed_prefill_len,
        )

        if wandb is not None:
            log = {}
            for k, v in eval_metrics.hr.items():
                log[f"eval/hr@{k}"] = v
            for k, v in eval_metrics.ndcg.items():
                log[f"eval/ndcg@{k}"] = v
            log["eval/invalid_prediction_count"] = eval_metrics.invalid_prediction_count
            wandb.log(log, step=int(getattr(state, "step", 0) or 0))

    if run_mode_norm == "eval" and cfg.eval.enabled:
        if not cfg.eval.constrained:
            raise NotImplementedError("JAX evaluator currently supports only constrained=true (SID trie).")
        eval_dataset = _build_eval_dataset(cfg, tokenizer)
        output_predictions_json = os.path.join(cfg.output_dir, "eval_predictions.json") if cfg.eval.save_predictions_json else None
        _preds, eval_metrics = evaluate_sid_next_item_jax(
            model=model,
            params=eval_params,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            sid_index_path=cfg.data.sid_index_path,
            info_file=cfg.data.info_file,
            batch_size=cfg.eval.batch_size,
            num_beams=cfg.eval.num_beams,
            max_cache_length=cfg.jax.max_cache_length,
            topk=list(cfg.eval.topk),
            output_predictions_json=output_predictions_json,
            prefill_mode=cfg.eval.prefill_mode,
            fixed_prefill_len=cfg.eval.fixed_prefill_len,
        )

        if wandb is not None:
            log = {}
            for k, v in eval_metrics.hr.items():
                log[f"eval/hr@{k}"] = v
            for k, v in eval_metrics.ndcg.items():
                log[f"eval/ndcg@{k}"] = v
            log["eval/invalid_prediction_count"] = eval_metrics.invalid_prediction_count
            wandb.log(log, step=0)

    if wandb is not None:
        wandb.finish()

    return {
        "config": asdict(cfg),
        "token_extension": asdict(extension),
        "vocab_resize": asdict(vocab_resize),
        "train": asdict(train_stats) if train_stats else None,
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


def run_sid_sft(cfg: SidSftConfig, *, run_mode: str) -> dict[str, Any]:
    _set_seed(cfg.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    run_mode_norm = str(run_mode).strip().lower()
    if run_mode_norm not in {"train", "eval", "train_eval"}:
        raise ValueError("run_mode must be one of: train, eval, train_eval")

    backend = str(cfg.backend or "jax").strip().lower()
    if backend in {"jax", "tpu"}:
        return _run_sid_sft_jax(cfg, run_mode_norm=run_mode_norm)
    raise ValueError(f"Unknown backend={cfg.backend!r} (expected jax|tpu)")
