from __future__ import annotations

import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

from MLLM_JAX.language.llama.llama import LlamaForCausalLM, LlamaJaxConfig, convert_torch_to_flax_llama
from MLLM_JAX.language.qwen2.modular_qwen2 import Qwen2ForCausalLM
from MLLM_JAX.utils import get_partition_rules_llama, match_partition_rules

from plugins.minionerec_v2.constraints import build_sid_trie_from_index, build_valid_sids_from_info
from plugins.training.core.io.hf_config import ensure_rope_theta
from plugins.training.core.io.hf_safetensors import load_hf_safetensors_state_dict
from plugins.training.core.logging.wandb import maybe_init_wandb
from plugins.training.core.mesh.mesh import create_mesh
from plugins.training.core.tokenizer import prepare_tokenizer
from projects.minionerec_jax_v2.config import MiniOneRecJaxV2Config
from projects.minionerec_jax_v2.datasets import OfficialEvalSidDataset
from projects.minionerec_jax_v2.evaluator import SidNextItemJaxEvaluator, evaluate_sid_next_item_jax


def _set_seed(seed: int) -> None:
    seed_i = int(seed)
    random.seed(seed_i)
    np.random.seed(seed_i)


def _parse_dtype(name: str) -> Any:
    import jax.numpy as jnp

    norm = str(name or "float32").strip().lower()
    if norm in {"float32", "f32"}:
        return jnp.float32
    if norm in {"bfloat16", "bf16"}:
        return jnp.bfloat16
    if norm in {"float16", "f16"}:
        return jnp.float16
    raise ValueError(f"Unsupported dtype: {name!r}")


def _to_numpy(x: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(x, torch.Tensor):
            value = x.detach().cpu()
            if value.dtype == torch.bfloat16:
                value = value.to(torch.float32)
            return value.numpy()
    except Exception:
        pass
    return np.asarray(x)


def _resolve_model_dir(cfg: MiniOneRecJaxV2Config) -> str:
    if cfg.checkpoint.local_root:
        local_root = str(cfg.checkpoint.local_root)
        if cfg.checkpoint.subdir:
            return os.path.join(local_root, str(cfg.checkpoint.subdir))
        name = str(cfg.dataset.dataset_name).strip().lower()
        candidates = [
            "Industrial_ckpt" if name == "industrial" else "Office_ckpt",
            "industrial_ckpt" if name == "industrial" else "office_ckpt",
            "industrial" if name == "industrial" else "office",
            "Industrial" if name == "industrial" else "Office",
        ]
        for candidate in candidates:
            path = os.path.join(local_root, candidate)
            if os.path.isdir(path):
                return path
        return local_root

    repo_snapshot = snapshot_download(repo_id=cfg.checkpoint.repo_id, revision=cfg.checkpoint.revision)
    if cfg.checkpoint.subdir:
        return os.path.join(repo_snapshot, str(cfg.checkpoint.subdir))

    name = str(cfg.dataset.dataset_name).strip().lower()
    candidates = [
        "Industrial_ckpt" if name == "industrial" else "Office_ckpt",
        "industrial_ckpt" if name == "industrial" else "office_ckpt",
        "industrial" if name == "industrial" else "office",
    ]
    for candidate in candidates:
        path = os.path.join(repo_snapshot, candidate)
        if os.path.isdir(path):
            return path
    return repo_snapshot


def _resolve_dataset_paths(cfg: MiniOneRecJaxV2Config) -> tuple[str, str, str]:
    dataset_name = str(cfg.dataset.dataset_name).strip().lower()
    if dataset_name not in {"industrial", "office"}:
        raise ValueError(f"Unsupported dataset_name={dataset_name!r}")

    source_root = str(cfg.dataset.source_root)
    full_name = "Industrial_and_Scientific" if dataset_name == "industrial" else "Office_Products"
    suffix = f"{full_name}_5_2016-10-2018-11"

    test_file = cfg.dataset.test_file or os.path.join(source_root, "test", f"{suffix}.csv")
    info_file = cfg.dataset.info_file or os.path.join(source_root, "info", f"{suffix}.txt")
    sid_index_path = cfg.dataset.sid_index_path or os.path.join(source_root, "index", f"{full_name}.index.json")

    return str(test_file), str(info_file), str(sid_index_path)


def _build_model_and_params(
    *,
    cfg: MiniOneRecJaxV2Config,
    model_path: str,
    mesh: Any,
    param_dtype: Any,
    compute_dtype: Any,
) -> tuple[Any, Any, Any]:
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=bool(cfg.checkpoint.trust_remote_code))
    ensure_rope_theta(hf_config)

    attention_mesh = mesh if jax.devices()[0].platform == "tpu" else None
    jax_cfg = LlamaJaxConfig(mesh=attention_mesh, dtype=compute_dtype, param_dtype=param_dtype)

    model_type = str(getattr(hf_config, "model_type", "") or "")
    if model_type == "qwen2":
        model = Qwen2ForCausalLM(hf_config, jax_cfg)
    elif model_type == "llama":
        model = LlamaForCausalLM(hf_config, jax_config=jax_cfg)
    else:
        raise ValueError(f"Unsupported model_type={model_type!r} for MiniOneRec eval (expected qwen2|llama)")

    state_dict = load_hf_safetensors_state_dict(model_path)
    if "lm_head.weight" not in state_dict and "model.embed_tokens.weight" in state_dict:
        state_dict = dict(state_dict)
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

    params = convert_torch_to_flax_llama(state_dict)
    params = jax.tree_util.tree_map(_to_numpy, params)
    params = jax.tree_util.tree_map(lambda x: np.asarray(x, dtype=np.dtype(param_dtype)), params)

    shapes = jax.eval_shape(lambda x: x, params)
    partitions = match_partition_rules(get_partition_rules_llama(), shapes)
    shardings = jax.tree_util.tree_map(lambda spec: NamedSharding(mesh, spec), partitions)
    params = jax.tree_util.tree_map(lambda x, sh: jax.device_put(jnp.asarray(x, dtype=param_dtype), sh), params, shardings)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=bool(cfg.checkpoint.trust_remote_code))
    tokenizer, _ = prepare_tokenizer(tokenizer, padding_side="left")
    return model, params, tokenizer


def run_official_eval(cfg: MiniOneRecJaxV2Config, *, run_tag: str | None = None) -> dict[str, Any]:
    import jax

    _set_seed(cfg.runtime.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if str(cfg.runtime.run_mode).strip().lower() != "eval":
        raise ValueError(f"MiniOneRec v2 currently supports eval-only, got runtime.run_mode={cfg.runtime.run_mode!r}")

    mesh = create_mesh(str(cfg.jax.mesh_shape or "auto"))
    param_dtype = _parse_dtype(cfg.jax.param_dtype)
    compute_dtype = _parse_dtype(cfg.jax.compute_dtype)

    wandb = maybe_init_wandb(
        cfg=cfg,
        project=cfg.wandb.project,
        name=(cfg.wandb.name if run_tag is None else f"{cfg.wandb.name or 'minionerec-jax-v2'}-{run_tag}"),
        mode=cfg.wandb.mode,
        process_index=jax.process_index(),
    )

    model_path = _resolve_model_dir(cfg)
    test_file, info_file, sid_index_path = _resolve_dataset_paths(cfg)

    for path in [test_file, info_file, sid_index_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required eval file not found: {path}")

    model, params, tokenizer = _build_model_and_params(
        cfg=cfg,
        model_path=model_path,
        mesh=mesh,
        param_dtype=param_dtype,
        compute_dtype=compute_dtype,
    )

    eval_dataset = OfficialEvalSidDataset(
        csv_path=test_file,
        tokenizer=tokenizer,
        max_len=int(cfg.dataset.max_len),
        sample=int(cfg.dataset.sample_test),
        seed=int(cfg.runtime.seed),
        dedup=bool(cfg.dataset.dedup),
        pretokenize=True,
        truncate_to_max_len=bool(cfg.dataset.truncate_to_max_len),
    )

    valid_sids = set(build_valid_sids_from_info(info_file))
    trie = build_sid_trie_from_index(tokenizer=tokenizer, sid_index_path=sid_index_path)

    output_dir = str(cfg.runtime.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_predictions_json = (
        os.path.join(output_dir, str(cfg.eval.output_predictions_name)) if bool(cfg.eval.save_predictions_json) else None
    )

    evaluator = SidNextItemJaxEvaluator(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        trie=trie,
        valid_sids=valid_sids,
        batch_size=int(cfg.decode.batch_size),
        num_beams=int(cfg.decode.num_beams),
        max_cache_length=int(cfg.jax.max_cache_length),
        topk=[int(k) for k in cfg.eval.topk],
        show_progress=bool(cfg.eval.show_progress),
        prefill_mode=str(cfg.decode.prefill_mode),
        fixed_prefill_len=(None if cfg.decode.fixed_prefill_len is None else int(cfg.decode.fixed_prefill_len)),
        do_sample=bool(cfg.decode.do_sample),
        temperature=float(cfg.decode.temperature),
        seed=int(cfg.runtime.seed),
    )

    _preds, metrics = evaluator.evaluate(params=params, output_predictions_json=output_predictions_json)

    if wandb is not None:
        log = {}
        for k, v in metrics.hr.items():
            log[f"eval/hr@{k}"] = v
        for k, v in metrics.ndcg.items():
            log[f"eval/ndcg@{k}"] = v
        log["eval/invalid_prediction_count"] = metrics.invalid_prediction_count
        log["eval/n_samples"] = metrics.n_samples
        log["eval/n_beams"] = metrics.n_beams
        wandb.log(log, step=0)
        wandb.finish()

    result = {
        "config": asdict(cfg),
        "checkpoint": {
            "model_path": model_path,
            "repo_id": cfg.checkpoint.repo_id,
            "revision": cfg.checkpoint.revision,
            "subdir": cfg.checkpoint.subdir,
        },
        "dataset": {
            "test_file": test_file,
            "info_file": info_file,
            "sid_index_path": sid_index_path,
            "size": int(len(eval_dataset)),
            "dataset_name": cfg.dataset.dataset_name,
        },
        "output": {
            "output_dir": output_dir,
            "predictions_json": output_predictions_json,
            "metrics_json": (
                None
                if output_predictions_json is None
                else str(Path(output_predictions_json).with_suffix(".metrics.json"))
            ),
            "run_tag": run_tag,
        },
        "eval": asdict(metrics),
    }
    return result


def run_official_eval_once(
    *,
    model: Any,
    params: Any,
    tokenizer: Any,
    eval_dataset: Any,
    sid_index_path: str,
    info_file: str,
    output_predictions_json: str | None,
    batch_size: int,
    num_beams: int,
    max_cache_length: int,
    topk: list[int],
    prefill_mode: str | None = "bucket",
    fixed_prefill_len: int | None = None,
    do_sample: bool = False,
    temperature: float = 1.0,
    seed: int = 42,
    show_progress: bool = False,
) -> tuple[list[list[str]], Any]:
    valid_sids = set(build_valid_sids_from_info(str(info_file)))
    trie = build_sid_trie_from_index(tokenizer=tokenizer, sid_index_path=str(sid_index_path))
    return evaluate_sid_next_item_jax(
        model=model,
        params=params,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        trie=trie,
        valid_sids=valid_sids,
        batch_size=int(batch_size),
        num_beams=int(num_beams),
        max_cache_length=int(max_cache_length),
        topk=[int(k) for k in topk],
        output_predictions_json=output_predictions_json,
        prefill_mode=prefill_mode,
        fixed_prefill_len=fixed_prefill_len,
        do_sample=bool(do_sample),
        temperature=float(temperature),
        seed=int(seed),
        show_progress=show_progress,
    )


__all__ = ["run_official_eval", "run_official_eval_once"]
