from __future__ import annotations

import importlib
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from MLLM_JAX.language.llama.llama import LlamaJaxConfig, convert_torch_to_flax_llama
from MLLM_JAX.language.qwen2.modular_qwen2 import Qwen2ForCausalLM
from MLLM_JAX.utils import get_partition_rules_llama, match_partition_rules

from plugins.training.core.checkpoint.msgpack import save_checkpoint
from plugins.training.core.io.hf_config import ensure_rope_theta
from plugins.training.core.logging.wandb import maybe_init_wandb
from plugins.training.core.tokenizer import prepare_tokenizer
from plugins.training.sft.jax.train import create_mesh_from_config, run_sft_train
from projects.openonerec_train.datasets import OpenOneRecSftDataset, aggregate_loader_pairs
from projects.sid_sft.jax.params import resize_lm_vocab


@dataclass(frozen=True)
class OpenOneRecTrainDataConfig:
    benchmark_data_dir: str = "workdir/OpenOneRec/benchmarks/data"
    task_types: tuple[str, ...] = ("video", "ad", "product", "label_cond", "interactive")
    split: str = "test"
    sample_size: int | str | None = 32
    max_len: int = 512


@dataclass(frozen=True)
class OpenOneRecTrainJaxConfig:
    mesh_shape: str = "1,-1,1"
    param_dtype: str = "float32"
    compute_dtype: str = "bfloat16"
    max_cache_length: int = 2048


@dataclass(frozen=True)
class OpenOneRecTrainTrainConfig:
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_steps: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    logging_steps: int = 10
    warmup_steps: int = 0
    shuffle: bool = True
    dataloader_drop_last: bool = True
    padding_side: str = "right"
    save_last: bool = True


@dataclass(frozen=True)
class OpenOneRecTrainWandbConfig:
    project: str = "openonerec-train"
    mode: str = "online"
    name: str | None = None


@dataclass(frozen=True)
class OpenOneRecTrainConfig:
    config_path: str
    openonerec_root: str
    base_model: str
    output_dir: str
    seed: int
    data: OpenOneRecTrainDataConfig = field(default_factory=OpenOneRecTrainDataConfig)
    jax: OpenOneRecTrainJaxConfig = field(default_factory=OpenOneRecTrainJaxConfig)
    train: OpenOneRecTrainTrainConfig = field(default_factory=OpenOneRecTrainTrainConfig)
    wandb: OpenOneRecTrainWandbConfig = field(default_factory=OpenOneRecTrainWandbConfig)


def _set_seed(seed: int) -> None:
    seed_i = int(seed)
    random.seed(seed_i)
    np.random.seed(seed_i)


def _resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _import_openonerec_registry(openonerec_root: str):
    benchmarks_dir = _resolve_path(str(Path(openonerec_root) / "benchmarks"))
    if not benchmarks_dir.exists():
        raise FileNotFoundError(f"OpenOneRec benchmarks directory not found: {benchmarks_dir}")

    bench_str = str(benchmarks_dir)
    if bench_str not in sys.path:
        sys.path.insert(0, bench_str)

    importlib.invalidate_caches()
    from benchmark.tasks.v1_0.registry import get_loader

    return get_loader


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.to(torch.float32)
        return value.numpy()
    return np.asarray(value)


def _parse_dtype(name: str):
    import jax.numpy as jnp

    norm = str(name or "float32").strip().lower()
    if norm in {"float32", "f32"}:
        return jnp.float32
    if norm in {"bfloat16", "bf16"}:
        return jnp.bfloat16
    if norm in {"float16", "f16"}:
        return jnp.float16
    raise ValueError(f"Unsupported dtype: {name!r}")


def _normalize_sample_size(value: int | str | None) -> int | str | None:
    if value is None:
        return None
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"", "none", "null"}:
            return None
        if norm == "full":
            return "full"
        return int(value)
    return int(value)


def run_openonerec_train(cfg: OpenOneRecTrainConfig, *, run_mode: str = "train") -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding

    _set_seed(cfg.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    run_mode_norm = str(run_mode).strip().lower()
    if run_mode_norm != "train":
        raise ValueError("run_mode must be 'train' for projects/openonerec_train")

    output_dir = _resolve_path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    tokenizer, pad_token_id = prepare_tokenizer(tokenizer, padding_side=str(cfg.train.padding_side))

    get_loader = _import_openonerec_registry(cfg.openonerec_root)

    sample_size = _normalize_sample_size(cfg.data.sample_size)
    task_samples: dict[str, dict[str, dict[str, Any]]] = {}
    task_sample_counts: dict[str, int] = {}

    for task_name in cfg.data.task_types:
        loader = get_loader(
            task_name=str(task_name),
            data_dir=str(cfg.data.benchmark_data_dir),
            tokenizer=tokenizer,
            enable_thinking=False,
        )
        samples = loader.load_data(split=str(cfg.data.split), sample_size=sample_size)
        task_samples[str(task_name)] = samples
        task_sample_counts[str(task_name)] = int(len(samples))

    pairs = aggregate_loader_pairs(task_types=cfg.data.task_types, task_samples=task_samples)
    train_dataset = OpenOneRecSftDataset(
        pairs=pairs,
        tokenizer=tokenizer,
        max_len=int(cfg.data.max_len),
        include_labels=True,
        pretokenize=True,
    )

    if len(train_dataset) <= 0:
        raise ValueError("OpenOneRec training dataset is empty after loader-to-SFT conversion")

    mesh = create_mesh_from_config(cfg.jax.mesh_shape)
    param_dtype = _parse_dtype(cfg.jax.param_dtype)
    compute_dtype = _parse_dtype(cfg.jax.compute_dtype)

    wandb = maybe_init_wandb(
        cfg=cfg,
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        mode=cfg.wandb.mode,
        process_index=jax.process_index(),
    )

    tokenizer_vocab_size = int(len(tokenizer))
    fsdp = int(mesh.shape.get("fsdp", 1))
    tp = int(mesh.shape.get("tp", 1))
    pad_multiple = max(1, fsdp * tp)
    padded_vocab_size = int(tokenizer_vocab_size)
    if pad_multiple > 1:
        remainder = int(tokenizer_vocab_size) % int(pad_multiple)
        if remainder != 0:
            padded_vocab_size = int(tokenizer_vocab_size) + (int(pad_multiple) - int(remainder))
            if int(jax.process_index()) == 0:
                print(
                    f"[openonerec-train] pad_vocab_size {tokenizer_vocab_size} -> {padded_vocab_size} "
                    f"(multiple={pad_multiple})"
                )

    base_config = AutoConfig.from_pretrained(cfg.base_model, trust_remote_code=True)
    ensure_rope_theta(base_config)
    base_config.vocab_size = int(padded_vocab_size)

    attention_mesh = mesh if jax.devices()[0].platform == "tpu" else None
    jax_config = LlamaJaxConfig(mesh=attention_mesh, dtype=compute_dtype, param_dtype=param_dtype)
    model = Qwen2ForCausalLM(base_config, jax_config)

    torch_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    state_dict = torch_model.state_dict()
    params = convert_torch_to_flax_llama(state_dict)
    params = jax.tree_util.tree_map(_to_numpy, params)
    del torch_model

    rng = jax.random.PRNGKey(int(cfg.seed))
    params, vocab_resize = resize_lm_vocab(params=params, new_vocab_size=int(padded_vocab_size), rng=rng)

    params = jax.tree_util.tree_map(lambda x: np.asarray(x, dtype=np.dtype(param_dtype)), params)
    shapes = jax.eval_shape(lambda x: x, params)
    partitions = match_partition_rules(get_partition_rules_llama(), shapes)
    shardings = jax.tree_util.tree_map(lambda spec: NamedSharding(mesh, spec), partitions)
    params = jax.tree_util.tree_map(lambda x, sh: jax.device_put(jnp.asarray(x, dtype=param_dtype), sh), params, shardings)

    max_steps = int(cfg.train.max_steps)
    if max_steps <= 0:
        raise ValueError("train.max_steps must be > 0 for openonerec train runner")

    state, train_stats = run_sft_train(
        mesh=mesh,
        model=model,
        params=params,
        train_dataset=train_dataset,
        pad_token_id=int(pad_token_id),
        pad_to_length=int(cfg.data.max_len) if int(cfg.data.max_len) > 0 else None,
        optimizer_name=str(cfg.train.optimizer),
        learning_rate=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
        grad_accum_steps=int(cfg.train.gradient_accumulation_steps),
        micro_batch_size_per_replica=int(cfg.train.per_device_train_batch_size),
        max_steps=int(max_steps),
        seed=int(cfg.seed),
        shuffle=bool(cfg.train.shuffle),
        dataloader_drop_last=bool(cfg.train.dataloader_drop_last),
        padding_side=str(cfg.train.padding_side),
        logging_steps=int(cfg.train.logging_steps),
        warmup_steps=int(cfg.train.warmup_steps),
        log_cb=(
            (
                lambda step, loss, effective_bs, step_time_sec: wandb.log(
                    {
                        "train/loss": float(loss),
                        "train/effective_batch_size": int(effective_bs),
                        "train/step_time_sec": float(step_time_sec),
                        "train/samples_per_sec": float(effective_bs) / max(float(step_time_sec), 1e-9),
                    },
                    step=int(step),
                )
            )
            if wandb is not None
            else None
        ),
    )

    tokenizer.save_pretrained(str(output_dir))

    checkpoint_path: str | None = None
    if bool(cfg.train.save_last) and int(jax.process_index()) == 0:
        checkpoint_path = save_checkpoint(output_dir=str(output_dir), state=state, name="last")

    if wandb is not None:
        wandb.finish()

    return {
        "config": asdict(cfg),
        "dataset_size": int(len(train_dataset)),
        "task_sample_counts": task_sample_counts,
        "vocab_resize": asdict(vocab_resize),
        "train": asdict(train_stats),
        "checkpoint_path": checkpoint_path,
    }


__all__ = [
    "OpenOneRecTrainConfig",
    "OpenOneRecTrainDataConfig",
    "OpenOneRecTrainJaxConfig",
    "OpenOneRecTrainTrainConfig",
    "OpenOneRecTrainWandbConfig",
    "run_openonerec_train",
]
