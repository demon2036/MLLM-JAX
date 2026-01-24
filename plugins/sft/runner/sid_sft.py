from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from plugins.sft.datasets.eval_sid_next_item import SidNextItemEvalDataset
from plugins.sft.datasets.fusion_seq_rec import FusionSeqRecSftDataset
from plugins.sft.datasets.sid_item_alignment import SidItemAlignmentDataset
from plugins.sft.datasets.sid_next_item import SidNextItemSftDataset
from plugins.sft.evaluator import evaluate_sid_next_item
from plugins.sft.jax.evaluator import evaluate_sid_next_item_jax
from plugins.sft.tokens import freeze_llm_only_train_new_embeddings, maybe_extend_tokenizer, maybe_extend_tokenizer_and_model
from plugins.sft.trainer import SftTrainResult, run_trainer
from plugins.sft.wandb_utils import maybe_init_wandb


def _set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
class SidSftTrainConfig:
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    optimizer: str = "adamw"
    weight_decay: float = 0.0
    num_train_epochs: float = 1.0
    max_steps: int = -1
    warmup_steps: int = 0
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 1
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
    batch_size: int = 4
    num_beams: int = 50
    max_new_tokens: int = 64
    length_penalty: float = 0.0
    topk: tuple[int, ...] = (1, 3, 5, 10, 20, 50)
    constrained: bool = True
    save_predictions_json: bool = True


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
    return SidNextItemEvalDataset(
        csv_path=cfg.data.test_file,
        tokenizer=tokenizer,
        max_len=cfg.data.max_len,
        sample=cfg.data.sample_test,
        seed=cfg.seed,
        pretokenize=True,
    )


def _run_sid_sft_hf(cfg: SidSftConfig, *, run_mode_norm: str) -> dict[str, Any]:
    # --- Load model/tokenizer ---
    if cfg.train.train_from_scratch:
        base_config = AutoConfig.from_pretrained(cfg.base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(base_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg.base_model, trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    extension = maybe_extend_tokenizer_and_model(tokenizer=tokenizer, model=model, sid_index_path=cfg.data.sid_index_path)
    if cfg.train.freeze_LLM:
        freeze_llm_only_train_new_embeddings(model=model, original_vocab_size=extension.original_vocab_size)

    wandb = maybe_init_wandb(cfg=cfg, project=cfg.wandb.project, name=cfg.wandb.name, mode=cfg.wandb.mode)
    wandb_enabled = wandb is not None

    train_result: SftTrainResult | None = None
    if run_mode_norm in {"train", "train_eval"}:
        train_dataset = _build_train_dataset(cfg, tokenizer)
        eval_dataset = SidNextItemSftDataset(
            csv_path=cfg.data.eval_file,
            tokenizer=tokenizer,
            max_len=cfg.data.max_len,
            sample=cfg.data.sample_eval,
            seed=cfg.seed,
            include_labels=True,
            pretokenize=True,
        )
        train_result = run_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=cfg.output_dir,
            run_name=cfg.wandb.name,
            seed=cfg.seed,
            train_cfg=asdict(cfg.train),
            wandb_enabled=wandb_enabled,
            device=cfg.device,
        )

    eval_metrics = None
    if run_mode_norm in {"eval", "train_eval"} and cfg.eval.enabled:
        eval_dataset = _build_eval_dataset(cfg, tokenizer)

        output_predictions_json = None
        if cfg.eval.save_predictions_json:
            output_predictions_json = os.path.join(cfg.output_dir, "eval_predictions.json")

        _preds, eval_metrics = evaluate_sid_next_item(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            info_file=cfg.data.info_file,
            batch_size=cfg.eval.batch_size,
            num_beams=cfg.eval.num_beams,
            max_new_tokens=cfg.eval.max_new_tokens,
            length_penalty=cfg.eval.length_penalty,
            topk=list(cfg.eval.topk),
            constrained=cfg.eval.constrained,
            output_predictions_json=output_predictions_json,
            device=cfg.device,
        )

        if wandb is not None:
            log = {}
            for k, v in eval_metrics.hr.items():
                log[f"eval/hr@{k}"] = v
            for k, v in eval_metrics.ndcg.items():
                log[f"eval/ndcg@{k}"] = v
            log["eval/invalid_prediction_count"] = eval_metrics.invalid_prediction_count
            wandb.log(log)

    if wandb is not None:
        wandb.finish()

    return {
        "config": asdict(cfg),
        "token_extension": asdict(extension),
        "train": asdict(train_result) if train_result else None,
        "eval": asdict(eval_metrics) if eval_metrics else None,
    }


def _run_sid_sft_jax(cfg: SidSftConfig, *, run_mode_norm: str) -> dict[str, Any]:
    import math

    import flax
    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch
    from jax.sharding import NamedSharding

    from MLLM_JAX.language.llama.llama import LlamaJaxConfig, convert_torch_to_flax_llama
    from MLLM_JAX.language.qwen2.modular_qwen2 import Qwen2ForCausalLM
    from MLLM_JAX.utils import get_partition_rules_llama, match_partition_rules

    from plugins.sft.jax.checkpoint import load_checkpoint, save_checkpoint
    from plugins.sft.jax.params import resize_lm_vocab
    from plugins.sft.jax.train import create_mesh_from_config, run_sft_train

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

    wandb = maybe_init_wandb(cfg=cfg, project=cfg.wandb.project, name=cfg.wandb.name, mode=cfg.wandb.mode) if jax.process_index() == 0 else None

    # Tokenizer + SID token extension (tokenizer-only; params resized below).
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    extension = maybe_extend_tokenizer(tokenizer=tokenizer, sid_index_path=cfg.data.sid_index_path)

    # Model config must reflect the resized tokenizer vocab.
    base_config = AutoConfig.from_pretrained(cfg.base_model, trust_remote_code=True)
    base_config.vocab_size = int(len(tokenizer))

    # Avoid TPU-only fused attention kernels on CPU/GPU backends.
    attention_mesh = mesh if jax.devices()[0].platform == "tpu" else None
    jax_config = LlamaJaxConfig(mesh=attention_mesh, dtype=compute_dtype, param_dtype=param_dtype)
    model = Qwen2ForCausalLM(base_config, jax_config)

    rng = jax.random.PRNGKey(int(cfg.seed))

    if cfg.train.train_from_scratch:
        dummy_len = 8
        dummy = jnp.zeros((1, dummy_len), dtype=jnp.int32)
        dummy_mask = jnp.ones((1, dummy_len), dtype=jnp.int32)
        dummy_pos = jnp.arange(dummy_len, dtype=jnp.int32)[None, :]
        variables = model.init(rng, input_ids=dummy, attention_mask=dummy_mask, position_ids=dummy_pos, cache=None)
        params = flax.core.unfreeze(variables["params"])
    else:
        torch_model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        state_dict = torch_model.state_dict()
        params = convert_torch_to_flax_llama(state_dict)
        params = jax.tree_util.tree_map(lambda x: np.array(x), params)

    # Resize embeddings/lm_head for new SID tokens.
    params, vocab_resize = resize_lm_vocab(params=params, new_vocab_size=int(len(tokenizer)), rng=rng)

    # Place params with sharding + dtype.
    params = jax.tree_util.tree_map(lambda x: np.asarray(x, dtype=np.dtype(param_dtype)), params)
    shapes = jax.eval_shape(lambda x: x, params)
    partitions = match_partition_rules(get_partition_rules_llama(), shapes)
    shardings = jax.tree_util.tree_map(lambda spec: NamedSharding(mesh, spec), partitions)
    params = jax.tree_util.tree_map(lambda x, sh: jax.device_put(jnp.asarray(x, dtype=param_dtype), sh), params, shardings)

    # Train (optional).
    state = None
    train_stats = None
    if run_mode_norm in {"train", "train_eval"}:
        train_dataset = _build_train_dataset(cfg, tokenizer)

        max_steps = int(cfg.train.max_steps)
        if max_steps <= 0:
            replicas = int(mesh.shape.get("dp", 1)) * int(mesh.shape.get("fsdp", 1))
            micro = int(cfg.train.per_device_train_batch_size) * replicas
            effective = micro * int(cfg.train.gradient_accumulation_steps)
            steps_per_epoch = int(math.ceil(len(train_dataset) / max(1, effective)))
            max_steps = int(math.ceil(float(cfg.train.num_train_epochs) * steps_per_epoch))

        state, train_stats = run_sft_train(
            mesh=mesh,
            model=model,
            params=params,
            train_dataset=train_dataset,
            pad_token_id=int(tokenizer.pad_token_id),
            optimizer_name=cfg.train.optimizer,
            learning_rate=float(cfg.train.learning_rate),
            weight_decay=float(cfg.train.weight_decay),
            grad_accum_steps=int(cfg.train.gradient_accumulation_steps),
            micro_batch_size_per_replica=int(cfg.train.per_device_train_batch_size),
            max_steps=int(max_steps),
            seed=int(cfg.seed),
            logging_steps=int(cfg.train.logging_steps),
            log_cb=(
                (lambda step, loss, effective_bs: wandb.log({"train/loss": loss, "train/effective_batch_size": effective_bs}, step=step))
                if wandb is not None
                else None
            ),
        )

        os.makedirs(cfg.output_dir, exist_ok=True)
        tokenizer.save_pretrained(cfg.output_dir)
        if jax.process_index() == 0:
            save_checkpoint(output_dir=cfg.output_dir, state=state, name="last")

    # Eval params: prefer trained state, else optionally load checkpoint, else use base params.
    eval_params = state.params if state is not None else params
    if state is None and cfg.train.resume_from_checkpoint:
        payload = load_checkpoint(str(cfg.train.resume_from_checkpoint))
        eval_params = payload.get("params", eval_params)

    eval_metrics = None
    if run_mode_norm in {"eval", "train_eval"} and cfg.eval.enabled:
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
        )

        if wandb is not None:
            log = {}
            for k, v in eval_metrics.hr.items():
                log[f"eval/hr@{k}"] = v
            for k, v in eval_metrics.ndcg.items():
                log[f"eval/ndcg@{k}"] = v
            log["eval/invalid_prediction_count"] = eval_metrics.invalid_prediction_count
            wandb.log(log)

    if wandb is not None:
        wandb.finish()

    return {
        "config": asdict(cfg),
        "token_extension": asdict(extension),
        "vocab_resize": asdict(vocab_resize),
        "train": asdict(train_stats) if train_stats else None,
        "eval": asdict(eval_metrics) if eval_metrics else None,
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
    if backend in {"hf", "torch", "pytorch"}:
        return _run_sid_sft_hf(cfg, run_mode_norm=run_mode_norm)
    raise ValueError(f"Unknown backend={cfg.backend!r} (expected jax|hf)")
