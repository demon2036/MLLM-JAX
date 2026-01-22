import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _best_effort_extract_text(result: object) -> str | None:
    if isinstance(result, dict):
        for key in ("text", "output_text", "generated_text"):
            val = result.get(key)
            if isinstance(val, str):
                return val
        choices = result.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        return content
                text = first.get("text")
                if isinstance(text, str):
                    return text
    return None


def _get_process_rss_bytes() -> int | None:
    """Best-effort process RSS (bytes). Works on Linux; returns None elsewhere."""

    try:
        status_path = Path("/proc/self/status")
        if status_path.exists():
            for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except Exception:
        pass
    return None


def _get_jax_device_memory_stats() -> list[dict[str, int]]:
    """Best-effort per-device memory stats (numeric fields only)."""

    try:
        import jax

        stats: list[dict[str, int]] = []
        for device in jax.devices():
            try:
                raw = device.memory_stats() or {}
            except Exception:
                raw = {}
            numeric: dict[str, int] = {}
            for k, v in raw.items():
                if isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)):
                    numeric[str(k)] = int(v)
            stats.append(numeric)
        return stats
    except Exception:
        return []


def _summarize_device_memory(device_stats: list[dict[str, int]]) -> dict[str, int]:
    """Summarize common fields across devices (sum/min/max)."""

    def _vals(key: str) -> list[int]:
        out: list[int] = []
        for d in device_stats:
            v = d.get(key)
            if isinstance(v, int):
                out.append(v)
        return out

    summary: dict[str, int] = {}
    for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit", "largest_free_block_bytes"):
        vals = _vals(key)
        if not vals:
            continue
        summary[f"{key}_sum"] = int(sum(vals))
        summary[f"{key}_min"] = int(min(vals))
        summary[f"{key}_max"] = int(max(vals))
    return summary


def _collect_memory_snapshot() -> dict[str, object]:
    device_stats = _get_jax_device_memory_stats()
    return {
        "process_rss_bytes": _get_process_rss_bytes(),
        "jax_device_memory_summary": _summarize_device_memory(device_stats),
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    workdir = repo_root / "workdir"
    download_dir = workdir / "hf_download"
    hf_cache_dir = workdir / "hf_models"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--prompt", default="你是谁？")
    parser.add_argument(
        "--tp-size",
        type=int,
        default=0,
        help="Tensor parallel size. 0 means auto (= jax.device_count()).",
    )
    parser.add_argument(
        "--mllm-param-dtype",
        default="bfloat16",
        help="Device dtype for injected params: bfloat16/float16/float32.",
    )
    parser.add_argument(
        "--device-indexes",
        default="",
        help=(
            "Comma-separated device indices to use. "
            "If empty and tp_size < device_count, defaults to 0..tp_size-1 to avoid DP replication."
        ),
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HOME", str(hf_cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache_dir))

    workdir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)
    hf_cache_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(
        json.dumps(
            {
                "phase": "start",
                "ts": datetime.now(timezone.utc).isoformat(),
                "model_id": str(args.model_id),
                "prompt": str(args.prompt),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    import jax

    device_count = int(jax.device_count())
    tp_size = int(args.tp_size)
    if tp_size <= 0:
        tp_size = device_count

    device_indexes: list[int] | None
    if str(args.device_indexes).strip() != "":
        device_indexes = [int(x) for x in str(args.device_indexes).split(",") if x.strip() != ""]
    elif tp_size < device_count:
        device_indexes = list(range(tp_size))
    else:
        device_indexes = None

    mem = _collect_memory_snapshot()
    print(
        json.dumps(
            {
                "phase": "jax_ready",
                "device_count": device_count,
                "tp_size": tp_size,
                "device_indexes": device_indexes,
                "elapsed_s": time.time() - t0,
                "memory": mem,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    # Build a mesh + model_config first, so we can place MLLM params on-device before Engine exists.
    from sgl_jax.srt.configs.model_config import ModelConfig
    from sgl_jax.srt.utils.mesh_utils import create_device_mesh

    mesh = create_device_mesh(
        ici_parallelism=[-1, tp_size],
        dcn_parallelism=[1, 1],
        device_indexes=device_indexes,
    )
    model_config = ModelConfig(
        model_path=str(args.model_id),
        trust_remote_code=True,
        dtype="bfloat16",
    )
    model_config.configure_for_tensor_parallel(tp_size)

    mem = _collect_memory_snapshot()
    print(
        json.dumps(
            {
                "phase": "mesh_and_model_config_ready",
                "mesh_shape": getattr(mesh, "shape", None),
                "num_hidden_layers": int(getattr(model_config, "num_hidden_layers", -1)),
                "num_attention_heads": int(getattr(model_config, "num_attention_heads", -1)),
                "num_key_value_heads": int(getattr(model_config, "num_key_value_heads", -1)),
                "original_num_kv_heads": int(model_config.get_total_num_kv_heads()),
                "elapsed_s": time.time() - t0,
                "memory": mem,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    # Load params via MLLM-JAX conversion (torch -> flax params tree) and materialize
    # the sglang-jax param dict ONCE (this is the "shared params" we want Engine to reuse).
    from MLLM_JAX.language.llama.llama import convert_torch_to_flax_llama
    from plugins.sglang_jax_inference.qwen2_mllm_param_adapter import (
        build_sglang_qwen2_param_dict_from_mllm_params,
    )

    try:
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM

        torch_model = AutoModelForCausalLM.from_pretrained(
            str(args.model_id),
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        state_dict = torch_model.state_dict()
        mllm_params = convert_torch_to_flax_llama(state_dict)
        mllm_params = jax.tree_util.tree_map(lambda x: np.array(x), mllm_params)
    finally:
        try:
            del state_dict
        except Exception:
            pass
        try:
            del torch_model
        except Exception:
            pass
        try:
            gc.collect()
        except Exception:
            pass
        try:
            if "torch" in sys.modules:
                torch.cuda.empty_cache()  # no-op on CPU, safe.
        except Exception:
            pass

    mem = _collect_memory_snapshot()
    print(
        json.dumps(
            {
                "phase": "mllm_params_cpu_ready",
                "elapsed_s": time.time() - t0,
                "memory": mem,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    with jax.set_mesh(mesh):
        param_dict = build_sglang_qwen2_param_dict_from_mllm_params(
            mllm_params=mllm_params,
            model_config=model_config,
            mesh=mesh,
            param_dtype=str(args.mllm_param_dtype),
        )

    # Drop CPU-side param tree; keep only the sharded JAX arrays (param_dict) alive.
    del mllm_params
    gc.collect()

    mem = _collect_memory_snapshot()
    print(
        json.dumps(
            {
                "phase": "mllm_params_device_ready",
                "num_params": int(len(param_dict)),
                "elapsed_s": time.time() - t0,
                "memory": mem,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    # Next step: patch Engine to reuse the same mesh + hot-swap params (implemented in later steps).
    raise SystemExit(0)


if __name__ == "__main__":
    main()

