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
        "jax_device_memory_stats": device_stats,
        "jax_device_memory_summary": _summarize_device_memory(device_stats),
    }


def _flatten_memory_for_wandb(mem: dict[str, object]) -> dict[str, int]:
    out: dict[str, int] = {}

    rss = mem.get("process_rss_bytes")
    if isinstance(rss, int):
        out["mem/process/rss_bytes"] = rss

    summary = mem.get("jax_device_memory_summary")
    if isinstance(summary, dict):
        for k, v in summary.items():
            if isinstance(v, int):
                out[f"mem/jax/{k}"] = v

    device_stats = mem.get("jax_device_memory_stats")
    if isinstance(device_stats, list):
        for i, d in enumerate(device_stats):
            if not isinstance(d, dict):
                continue
            for k, v in d.items():
                if isinstance(v, int):
                    out[f"mem/jax/device{i}/{k}"] = v

    return out


def _wandb_log_phase(
    *,
    wandb,
    step: int,
    phase: str,
    elapsed_s: float,
    mem: dict[str, object],
    extra: dict[str, object] | None = None,
) -> None:
    if wandb is None:
        return
    metrics: dict[str, object] = {
        "phase/idx": int(step),
        "phase/name": str(phase),
        "time/elapsed_s": float(elapsed_s),
    }
    metrics.update(_flatten_memory_for_wandb(mem))
    if extra:
        metrics.update(extra)
    try:
        wandb.log(metrics, step=int(step))
    except Exception:
        return


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    workdir = repo_root / "workdir"
    download_dir = workdir / "hf_download"
    hf_cache_dir = workdir / "hf_models"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--prompt", default="你是谁")
    parser.add_argument(
        "--tp-size",
        type=int,
        default=0,
        help="Tensor parallel size. 0 means auto (= jax.device_count()).",
    )
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument(
        "--mllm-param-dtype",
        default="bfloat16",
        help="Device dtype for injected params: bfloat16/float16/float32.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="启用 Weights & Biases 记录（仅 jax.process_index()==0）。",
    )
    parser.add_argument(
        "--wandb-project",
        default="sglang-jax-qwen25-3b-mllm-param-swap-memory",
        help="W&B project（不存在会自动创建）。",
    )
    parser.add_argument(
        "--wandb-name",
        default="",
        help="W&B run name（默认自动生成）。",
    )
    parser.add_argument(
        "--wandb-entity",
        default="",
        help="W&B entity（可选；默认使用当前账号）。",
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HOME", str(hf_cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache_dir))

    workdir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)
    hf_cache_dir.mkdir(parents=True, exist_ok=True)

    from sgl_jax.version import __version__ as sglang_jax_version
    from sgl_jax.srt.entrypoints.engine import Engine

    try:
        import jax

        device_count = int(jax.device_count())
        process_index = int(jax.process_index())
    except Exception:
        device_count = None
        process_index = 0

    tp_size = int(args.tp_size)
    dp_size = int(args.dp_size)
    if tp_size <= 0:
        tp_size = int(device_count) if device_count is not None else 4
    if dp_size <= 0:
        raise ValueError(f"--dp-size must be > 0, got {dp_size}")
    if device_count is not None and tp_size * dp_size != device_count:
        raise ValueError(
            f"tp_size*dp_size must match jax.device_count() "
            f"(tp_size={tp_size}, dp_size={dp_size}, device_count={device_count})"
        )

    t0 = time.time()
    wandb_step = 0

    wandb = None
    wandb_service = None
    if args.wandb:
        try:
            if process_index == 0 and os.environ.get("WANDB_MODE") != "disabled":
                import wandb as _wandb

                run_name = str(args.wandb_name).strip()
                if run_name == "":
                    run_name = (
                        "qwen25_3b_mllm_param_swap_"
                        + datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
                    )
                init_kwargs = {
                    "project": str(args.wandb_project),
                    "name": run_name,
                    "config": {
                        "model_id": str(args.model_id),
                        "tp_size": tp_size,
                        "dp_size": dp_size,
                        "dtype": str(args.mllm_param_dtype),
                        "load_format": "dummy",
                    },
                    "dir": str(workdir / "wandb"),
                }
                entity = str(args.wandb_entity).strip()
                if entity != "":
                    init_kwargs["entity"] = entity

                _wandb.init(**init_kwargs)
                wandb = _wandb
                try:
                    run = getattr(wandb, "run", None)
                    backend = getattr(run, "_backend", None)
                    wandb_service = getattr(backend, "_service", None)
                except Exception:
                    wandb_service = None

                mem = _collect_memory_snapshot()
                elapsed_s = time.time() - t0
                print(
                    json.dumps(
                        {
                            "phase": "wandb_init",
                            "project": str(args.wandb_project),
                            "name": run_name,
                            "url": getattr(wandb.run, "url", None),
                            "id": getattr(wandb.run, "id", None),
                            "elapsed_s": elapsed_s,
                            "memory": mem,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                _wandb_log_phase(
                    wandb=wandb,
                    step=wandb_step,
                    phase="wandb_init",
                    elapsed_s=elapsed_s,
                    mem=mem,
                )
                wandb_step += 1
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "phase": "wandb_init_error",
                        "error": str(exc),
                        "hint": (
                            "检查 WANDB_API_KEY 是否已设置（建议通过 .env 同步到 TPU），"
                            "或设置 WANDB_MODE=disabled 关闭。"
                        ),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    mem = _collect_memory_snapshot()
    elapsed_s = time.time() - t0
    print(
        json.dumps(
            {
                "phase": "jax_ready",
                "sgl_jax_version": sglang_jax_version,
                "elapsed_s": elapsed_s,
                "memory": mem,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    _wandb_log_phase(
        wandb=wandb,
        step=wandb_step,
        phase="jax_ready",
        elapsed_s=elapsed_s,
        mem=mem,
        extra={"sgl_jax_version": sglang_jax_version},
    )
    wandb_step += 1

    engine = Engine(
        model_path=str(args.model_id),
        tokenizer_path=str(args.model_id),
        trust_remote_code=True,
        device="tpu",
        tp_size=tp_size,
        dp_size=dp_size,
        enable_single_process=True,
        load_format="dummy",
        dtype="bfloat16",
        mem_fraction_static=0.8,
        max_prefill_tokens=1024,
        max_total_tokens=4096,
        max_running_requests=16,
        disable_precompile=True,
        skip_server_warmup=True,
        log_level="info",
        download_dir=str(download_dir),
    )

    try:
        model_runner = engine.scheduler_info["scheduler"].tp_worker.worker.model_runner
        mem = _collect_memory_snapshot()
        elapsed_s = time.time() - t0
        print(
            json.dumps(
                {
                    "phase": "engine_ready_dummy",
                    "model_id": str(args.model_id),
                    "device": "tpu",
                    "tp_size": tp_size,
                    "dp_size": dp_size,
                    "dtype": "bfloat16",
                    "load_format": "dummy",
                    "download_dir": str(download_dir),
                    "hf_cache_dir": str(hf_cache_dir),
                    "sgl_jax_version": sglang_jax_version,
                    "num_model_state_leaves": len(model_runner.model_state_leaves),
                    "elapsed_s": elapsed_s,
                    "memory": mem,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        _wandb_log_phase(
            wandb=wandb,
            step=wandb_step,
            phase="engine_ready_dummy",
            elapsed_s=elapsed_s,
            mem=mem,
            extra={"num_model_state_leaves": int(len(model_runner.model_state_leaves))},
        )
        wandb_step += 1

        # Load params via MLLM-JAX conversion (torch -> flax params tree).
        from MLLM_JAX.language.llama.llama import convert_torch_to_flax_llama

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
        elapsed_s = time.time() - t0
        print(
            json.dumps(
                {
                    "phase": "mllm_params_cpu_ready",
                    "model_id": str(args.model_id),
                    "elapsed_s": elapsed_s,
                    "memory": mem,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        _wandb_log_phase(
            wandb=wandb,
            step=wandb_step,
            phase="mllm_params_cpu_ready",
            elapsed_s=elapsed_s,
            mem=mem,
        )
        wandb_step += 1

        from plugins.sglang_jax_inference.qwen2_mllm_param_adapter import (
            build_sglang_qwen2_param_dict_from_mllm_params,
        )

        param_dict = build_sglang_qwen2_param_dict_from_mllm_params(
            mllm_params=mllm_params,
            model_config=model_runner.model_config,
            mesh=model_runner.mesh,
            param_dtype=str(args.mllm_param_dtype),
        )
        mem = _collect_memory_snapshot()
        elapsed_s = time.time() - t0
        print(
            json.dumps(
                {
                    "phase": "sglang_param_dict_ready",
                    "num_params": int(len(param_dict)),
                    "elapsed_s": elapsed_s,
                    "memory": mem,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        _wandb_log_phase(
            wandb=wandb,
            step=wandb_step,
            phase="sglang_param_dict_ready",
            elapsed_s=elapsed_s,
            mem=mem,
            extra={"num_params": int(len(param_dict))},
        )
        wandb_step += 1

        from plugins.sglang_jax_inference.engine_weight_swap import (
            swap_engine_weights_from_param_dict,
        )

        num_leaves = swap_engine_weights_from_param_dict(engine=engine, param_dict=param_dict)
        mem = _collect_memory_snapshot()
        elapsed_s = time.time() - t0
        print(
            json.dumps(
                {
                    "phase": "weights_swapped_from_mllm",
                    "num_model_state_leaves": int(num_leaves),
                    "elapsed_s": elapsed_s,
                    "memory": mem,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        _wandb_log_phase(
            wandb=wandb,
            step=wandb_step,
            phase="weights_swapped_from_mllm",
            elapsed_s=elapsed_s,
            mem=mem,
            extra={"num_model_state_leaves": int(num_leaves)},
        )
        wandb_step += 1

        # Drop CPU-side param tree; keep only Engine leaves.
        del mllm_params
        del param_dict
        gc.collect()

        mem = _collect_memory_snapshot()
        elapsed_s = time.time() - t0
        print(
            json.dumps(
                {
                    "phase": "after_cleanup",
                    "elapsed_s": elapsed_s,
                    "memory": mem,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        _wandb_log_phase(
            wandb=wandb,
            step=wandb_step,
            phase="after_cleanup",
            elapsed_s=elapsed_s,
            mem=mem,
        )
        wandb_step += 1

        sampling_params = {"temperature": 0.0, "max_new_tokens": 64}
        result = engine.generate(prompt=str(args.prompt), sampling_params=sampling_params)
        mem = _collect_memory_snapshot()
        elapsed_s = time.time() - t0
        print(
            json.dumps(
                {
                    "phase": "generate_result",
                    "prompt": str(args.prompt),
                    "sampling_params": sampling_params,
                    "text": _best_effort_extract_text(result),
                    "raw": result,
                    "elapsed_s": elapsed_s,
                    "memory": mem,
                },
                ensure_ascii=False,
                default=str,
            ),
            flush=True,
        )
        _wandb_log_phase(
            wandb=wandb,
            step=wandb_step,
            phase="generate_result",
            elapsed_s=elapsed_s,
            mem=mem,
        )
        wandb_step += 1
    finally:
        if wandb is not None:
            try:
                wandb.finish()
            except Exception:
                pass
            try:
                if wandb_service is not None and hasattr(wandb_service, "teardown"):
                    wandb_service.teardown(0)
            except Exception:
                pass
        engine.shutdown()


if __name__ == "__main__":
    main()

