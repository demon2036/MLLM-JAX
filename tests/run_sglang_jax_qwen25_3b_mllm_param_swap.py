import argparse
import gc
import hashlib
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


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _get_nnx_state_leaf(params_state: object, path: str):
    """Traverse an NNX state tree using a dot-separated path."""

    keys = path.split(".")
    current_level = params_state

    for key in keys:
        if key.isdigit():
            current_level = current_level[int(key)]
            continue
        if hasattr(current_level, "__contains__") and key in current_level:
            current_level = current_level[key]
        elif hasattr(current_level, key):
            current_level = getattr(current_level, key)
        else:
            raise ValueError(f"{path} is not a valid param path")

    return current_level


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


def _truncate_text_middle(text: str, *, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0:
        return "", True
    if len(text) <= max_chars:
        return text, False
    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars
    truncated = (
        text[:head_chars]
        + "\n\n...[TRUNCATED]...\n\n"
        + text[-tail_chars:]
    )
    return truncated, True


def _print_text_block(*, title: str, text: str | None, max_chars: int) -> None:
    print(f"\n===== {title} =====", flush=True)
    if text is None:
        print("<None>", flush=True)
        print(f"===== END {title} =====\n", flush=True)
        return

    truncated, was_truncated = _truncate_text_middle(text, max_chars=int(max_chars))
    print(truncated, flush=True)
    if was_truncated:
        print(
            f"\n...[TRUNCATED: total_chars={len(text)} max_chars={int(max_chars)}]...\n",
            flush=True,
        )
    print(f"===== END {title} =====\n", flush=True)


def _wandb_log_text(
    *,
    wandb,
    step: int,
    key: str,
    text: str | None,
    max_chars: int = 20000,
    as_html: bool = True,
) -> None:
    if wandb is None or text is None:
        return

    truncated, was_truncated = _truncate_text_middle(text, max_chars=max_chars)

    value: object = truncated
    if as_html:
        try:
            html_cls = getattr(wandb, "Html", None)
            if html_cls is not None:
                import html as _html

                value = html_cls(f"<pre>{_html.escape(truncated)}</pre>")
        except Exception:
            value = truncated

    payload: dict[str, object] = {
        key: value,
        f"{key}_chars": int(len(text)),
        f"{key}_truncated": int(was_truncated),
    }
    try:
        wandb.log(payload, step=int(step))
    except Exception:
        return


def _wandb_log_text_table(
    *,
    wandb,
    step: int,
    key: str,
    rows: list[tuple[str, str | None]],
    max_chars: int = 20000,
) -> None:
    if wandb is None:
        return

    table_cls = getattr(wandb, "Table", None)
    if table_cls is None:
        return

    columns = ["kind", "sha256", "chars", "truncated", "text"]
    data: list[list[object]] = []
    for kind, text in rows:
        if text is None:
            data.append([str(kind), "", 0, 0, ""])
            continue
        truncated, was_truncated = _truncate_text_middle(text, max_chars=int(max_chars))
        data.append(
            [
                str(kind),
                _sha256_text(text),
                int(len(text)),
                int(was_truncated),
                truncated,
            ]
        )

    try:
        table = table_cls(data=data, columns=columns)
        wandb.log({str(key): table}, step=int(step))
    except Exception:
        return


def _wandb_save_text_files(
    *,
    wandb,
    workdir: Path,
    prompt: str,
    prompt_sha256: str,
    out_text: str | None,
    out_text_sha256: str | None,
    out_text2: str | None,
    out_text2_sha256: str | None,
) -> None:
    if wandb is None:
        return
    run = getattr(wandb, "run", None)
    if run is None:
        return

    run_id = str(getattr(run, "id", "") or "unknown")
    sample_dir = Path(workdir) / "wandb_samples" / run_id
    try:
        sample_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    def _safe_write(name: str, text: str | None) -> Path | None:
        path = sample_dir / name
        try:
            path.write_text(text if text is not None else "<None>", encoding="utf-8")
            return path
        except Exception:
            return None

    paths: list[Path] = []
    paths.append(_safe_write(f"prompt_sha256_{prompt_sha256}.txt", prompt))
    if out_text_sha256 is not None:
        paths.append(_safe_write(f"output_1_sha256_{out_text_sha256}.txt", out_text))
    else:
        paths.append(_safe_write("output_1.txt", out_text))
    if out_text2_sha256 is not None:
        paths.append(_safe_write(f"output_2_sha256_{out_text2_sha256}.txt", out_text2))
    else:
        paths.append(_safe_write("output_2.txt", out_text2))

    for p in paths:
        if p is None:
            continue
        try:
            wandb.save(str(p), base_path=str(workdir))
        except Exception:
            continue


def _wandb_summary_set(*, wandb, key: str, value: object) -> None:
    if wandb is None:
        return
    run = getattr(wandb, "run", None)
    if run is None:
        return
    try:
        run.summary[str(key)] = value
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
        "--prompt-file",
        default="",
        help="Load prompt from a UTF-8 text file (overrides --prompt).",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=0,
        help="Tensor parallel size. 0 means auto (= jax.device_count()).",
    )
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=4096,
        help="Engine KV cache capacity cap (tokens). Larger => larger preallocated KV cache.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Generation cap for the first generate() (tokens).",
    )
    parser.add_argument(
        "--max-new-tokens-after-rebuild",
        type=int,
        default=128,
        help="Generation cap for the post-rebuild generate() (tokens).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 is greedy).",
    )
    parser.add_argument(
        "--check-same-output",
        action="store_true",
        help="After KV rebuild: compare output texts between the 1st and 2nd generate() and log hashes.",
    )
    parser.add_argument(
        "--assert-same-output",
        action="store_true",
        help="Same as --check-same-output, but exit non-zero if texts differ.",
    )
    parser.add_argument(
        "--print-text-max-chars",
        type=int,
        default=20000,
        help="Max chars to print for prompt/outputs (<=0 disables printing).",
    )
    parser.add_argument(
        "--wandb-text-max-chars",
        type=int,
        default=20000,
        help="Max chars to store per text field in W&B media/table.",
    )
    parser.add_argument(
        "--keep-param-dict",
        action="store_true",
        help=(
            "Keep the injected param_dict alive through generate() "
            "to verify Engine shares the same jax.Arrays (no copy)."
        ),
    )
    parser.add_argument(
        "--verify-param-sharing",
        action="store_true",
        help=(
            "Verify Engine Param.value is the exact same jax.Array object as "
            "param_dict[path] for a small sample."
        ),
    )
    parser.add_argument(
        "--mllm-param-dtype",
        default="bfloat16",
        help="Device dtype for injected params: bfloat16/float16/float32.",
    )
    parser.add_argument(
        "--kv-drop-rebuild",
        action="store_true",
        help=(
            "After the first generate(): flush cache, drop KV cache buffers, allocate a second "
            "param_dict (to simulate two param sets), rebuild KV cache buffers, and generate() again."
        ),
    )
    parser.add_argument(
        "--kv-drop-clear-jax-caches",
        action="store_true",
        help=(
            "After dropping KV cache buffers, call jax.clear_caches(). "
            "This may help free memory but can trigger re-compilation later."
        ),
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
    if args.assert_same_output:
        args.check_same_output = True

    prompt = str(args.prompt)
    prompt_file = str(args.prompt_file).strip()
    if prompt_file != "":
        prompt_path = Path(prompt_file)
        if not prompt_path.is_absolute():
            prompt_path = repo_root / prompt_path
        prompt = prompt_path.read_text(encoding="utf-8")

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
                        "max_total_tokens": int(args.max_total_tokens),
                        "max_new_tokens": int(args.max_new_tokens),
                        "max_new_tokens_after_rebuild": int(args.max_new_tokens_after_rebuild),
                        "temperature": float(args.temperature),
                        "kv_drop_rebuild": bool(args.kv_drop_rebuild),
                        "kv_drop_clear_jax_caches": bool(args.kv_drop_clear_jax_caches),
                        "prompt_file": prompt_file,
                        "prompt_chars": int(len(prompt)),
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
                _wandb_summary_set(wandb=wandb, key="wandb/url", value=getattr(wandb.run, "url", None))
                _wandb_summary_set(wandb=wandb, key="wandb/id", value=getattr(wandb.run, "id", None))
                _wandb_summary_set(wandb=wandb, key="prompt/chars", value=int(len(prompt)))
                if prompt_file != "":
                    _wandb_summary_set(wandb=wandb, key="prompt/file", value=prompt_file)
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
        max_total_tokens=int(args.max_total_tokens),
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

        if args.verify_param_sharing:
            from flax import nnx

            params_state = nnx.state(model_runner.model)
            sample_paths = [
                "model.embed_tokens.embedding",
                "model.norm.scale",
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.mlp.gate_proj.weight",
            ]
            results: dict[str, object] = {}
            for path in sample_paths:
                try:
                    leaf = _get_nnx_state_leaf(params_state, str(path))
                    if not hasattr(leaf, "value"):
                        results[str(path)] = {"is_param": False}
                        continue
                    expected = param_dict.get(str(path))
                    engine_value = getattr(leaf, "value", None)
                    results[str(path)] = {
                        "found_in_param_dict": expected is not None,
                        "same_object": bool(expected is engine_value) if expected is not None else False,
                    }
                except Exception as exc:
                    results[str(path)] = {"error": str(exc)}

            mem = _collect_memory_snapshot()
            elapsed_s = time.time() - t0
            print(
                json.dumps(
                    {
                        "phase": "param_sharing_check",
                        "elapsed_s": elapsed_s,
                        "memory": mem,
                        "results": results,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            _wandb_log_phase(
                wandb=wandb,
                step=wandb_step,
                phase="param_sharing_check",
                elapsed_s=elapsed_s,
                mem=mem,
            )
            wandb_step += 1

        if not args.kv_drop_rebuild:
            # Drop CPU-side param tree; keep on-device arrays (Engine + optional param_dict).
            del mllm_params
            gc.collect()

        pre_generate_phase = "after_cleanup"
        if not args.keep_param_dict:
            del param_dict
            gc.collect()
        else:
            pre_generate_phase = "before_generate_param_dict_kept"

        mem = _collect_memory_snapshot()
        elapsed_s = time.time() - t0
        print(
            json.dumps(
                {
                    "phase": pre_generate_phase,
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
            phase=pre_generate_phase,
            elapsed_s=elapsed_s,
            mem=mem,
        )
        wandb_step += 1

        sampling_params = {
            "temperature": float(args.temperature),
            "max_new_tokens": int(args.max_new_tokens),
        }
        result = engine.generate(prompt=prompt, sampling_params=sampling_params)
        out_text = _best_effort_extract_text(result)
        prompt_sha256 = _sha256_text(prompt)
        out_text_sha256 = _sha256_text(out_text) if out_text is not None else None
        mem = _collect_memory_snapshot()
        elapsed_s = time.time() - t0
        print(
            json.dumps(
                {
                    "phase": "generate_result",
                    "prompt": prompt,
                    "prompt_sha256": prompt_sha256,
                    "sampling_params": sampling_params,
                    "text": out_text,
                    "text_sha256": out_text_sha256,
                    "raw": result,
                    "elapsed_s": elapsed_s,
                    "memory": mem,
                },
                ensure_ascii=False,
                default=str,
            ),
            flush=True,
        )
        if int(args.print_text_max_chars) > 0:
            _print_text_block(
                title=f"SAMPLE PROMPT sha256={prompt_sha256} chars={len(prompt)}",
                text=prompt,
                max_chars=int(args.print_text_max_chars),
            )
            _print_text_block(
                title=f"SAMPLE OUTPUT #1 sha256={out_text_sha256} chars={len(out_text) if out_text is not None else 0}",
                text=out_text,
                max_chars=int(args.print_text_max_chars),
            )
        _wandb_log_phase(
            wandb=wandb,
            step=wandb_step,
            phase="generate_result",
            elapsed_s=elapsed_s,
            mem=mem,
        )
        _wandb_log_text(
            wandb=wandb,
            step=wandb_step,
            key="sample/prompt",
            text=prompt,
            max_chars=int(args.wandb_text_max_chars),
        )
        _wandb_log_text(
            wandb=wandb,
            step=wandb_step,
            key="sample/output_text",
            text=out_text,
            max_chars=int(args.wandb_text_max_chars),
        )
        _wandb_log_text(
            wandb=wandb,
            step=wandb_step,
            key="sample_prompt",
            text=prompt,
            max_chars=int(args.wandb_text_max_chars),
        )
        _wandb_log_text(
            wandb=wandb,
            step=wandb_step,
            key="sample_output_text_1",
            text=out_text,
            max_chars=int(args.wandb_text_max_chars),
        )
        _wandb_summary_set(wandb=wandb, key="sample/prompt_sha256", value=prompt_sha256)
        if out_text_sha256 is not None:
            _wandb_summary_set(wandb=wandb, key="sample/output_text_sha256", value=out_text_sha256)
        if out_text is not None:
            summary_text, _ = _truncate_text_middle(out_text, max_chars=int(args.wandb_text_max_chars))
            _wandb_summary_set(wandb=wandb, key="sample/output_text", value=summary_text)
            _wandb_summary_set(wandb=wandb, key="sample/output_text_chars", value=int(len(out_text)))
        wandb_step += 1

        if args.keep_param_dict:
            # Drop the Python-side dict after generate; Engine keeps references.
            del param_dict
            gc.collect()

            mem = _collect_memory_snapshot()
            elapsed_s = time.time() - t0
            print(
                json.dumps(
                    {
                        "phase": "after_drop_param_dict",
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
                phase="after_drop_param_dict",
                elapsed_s=elapsed_s,
                mem=mem,
            )
            wandb_step += 1

        if args.kv_drop_rebuild:
            # 1) Ensure engine has no in-flight requests and allocator is reset.
            flush_out = engine.flush_cache()
            mem = _collect_memory_snapshot()
            elapsed_s = time.time() - t0
            print(
                json.dumps(
                    {
                        "phase": "flush_cache_before_kv_drop",
                        "flush_cache_output": getattr(flush_out, "__dict__", str(flush_out)),
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
                phase="flush_cache_before_kv_drop",
                elapsed_s=elapsed_s,
                mem=mem,
            )
            wandb_step += 1

            # 2) Drop KV cache buffers (HBM) while keeping Engine alive.
            from plugins.sglang_jax_inference.kv_cache_lifecycle import drop_engine_kv_cache

            drop_info = drop_engine_kv_cache(
                engine=engine,
                flush_cache=False,
                clear_jax_caches=bool(args.kv_drop_clear_jax_caches),
            )
            gc.collect()
            mem = _collect_memory_snapshot()
            elapsed_s = time.time() - t0
            print(
                json.dumps(
                    {
                        "phase": "kv_cache_dropped",
                        "drop_info": drop_info,
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
                phase="kv_cache_dropped",
                elapsed_s=elapsed_s,
                mem=mem,
                extra={
                    "kv/fused_bytes_total": int(drop_info.get("fused_bytes_total", 0)),
                    "kv/fused_bytes_per_layer": int(drop_info.get("fused_bytes_per_layer", 0)),
                    "kv/deleted_buffers": int(drop_info.get("deleted_buffers", 0)),
                },
            )
            wandb_step += 1

            # 3) Allocate a *second* param_dict to simulate two param sets alive.
            param_dict2 = build_sglang_qwen2_param_dict_from_mllm_params(
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
                        "phase": "sglang_param_dict2_ready",
                        "num_params": int(len(param_dict2)),
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
                phase="sglang_param_dict2_ready",
                elapsed_s=elapsed_s,
                mem=mem,
                extra={"num_params": int(len(param_dict2))},
            )
            wandb_step += 1

            # CPU-side tree no longer needed after we have two on-device sets.
            del mllm_params
            gc.collect()

            mem = _collect_memory_snapshot()
            elapsed_s = time.time() - t0
            print(
                json.dumps(
                    {
                        "phase": "after_drop_mllm_params_cpu",
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
                phase="after_drop_mllm_params_cpu",
                elapsed_s=elapsed_s,
                mem=mem,
            )
            wandb_step += 1

            # 4) Rebuild KV cache buffers and re-run generate().
            from plugins.sglang_jax_inference.kv_cache_lifecycle import rebuild_engine_kv_cache

            rebuild_info = rebuild_engine_kv_cache(engine=engine)
            mem = _collect_memory_snapshot()
            elapsed_s = time.time() - t0
            print(
                json.dumps(
                    {
                        "phase": "kv_cache_rebuilt",
                        "rebuild_info": rebuild_info,
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
                phase="kv_cache_rebuilt",
                elapsed_s=elapsed_s,
                mem=mem,
                extra={
                    "kv/fused_bytes_total": int(rebuild_info.get("fused_bytes_total", 0)),
                    "kv/fused_bytes_per_layer": int(rebuild_info.get("fused_bytes_per_layer", 0)),
                },
            )
            wandb_step += 1

            sampling_params2 = {
                "temperature": float(args.temperature),
                "max_new_tokens": int(args.max_new_tokens_after_rebuild),
            }
            if args.check_same_output:
                sampling_params2 = dict(sampling_params)
            result2 = engine.generate(prompt=prompt, sampling_params=sampling_params2)
            out_text2 = _best_effort_extract_text(result2)
            out_text2_sha256 = _sha256_text(out_text2) if out_text2 is not None else None
            same_output = bool(
                out_text is not None and out_text2 is not None and out_text == out_text2
            )
            mem = _collect_memory_snapshot()
            elapsed_s = time.time() - t0
            print(
                json.dumps(
                    {
                        "phase": "generate_result_after_kv_rebuild",
                        "prompt": prompt,
                        "prompt_sha256": prompt_sha256,
                        "sampling_params": sampling_params2,
                        "text_sha256_before": out_text_sha256,
                        "text": out_text2,
                        "text_sha256": out_text2_sha256,
                        "same_output": int(same_output),
                        "raw": result2,
                        "elapsed_s": elapsed_s,
                        "memory": mem,
                    },
                    ensure_ascii=False,
                    default=str,
                ),
                flush=True,
            )
            print(
                f"[determinism] same_output={int(same_output)} "
                f"output1_sha256={out_text_sha256} output2_sha256={out_text2_sha256}",
                flush=True,
            )
            if int(args.print_text_max_chars) > 0:
                _print_text_block(
                    title=f"SAMPLE OUTPUT #2 sha256={out_text2_sha256} chars={len(out_text2) if out_text2 is not None else 0}",
                    text=out_text2,
                    max_chars=int(args.print_text_max_chars),
                )
            _wandb_log_phase(
                wandb=wandb,
                step=wandb_step,
                phase="generate_result_after_kv_rebuild",
                elapsed_s=elapsed_s,
                mem=mem,
                extra={
                    "determinism/same_output": int(same_output),
                },
            )
            _wandb_log_text(
                wandb=wandb,
                step=wandb_step,
                key="sample/output_text_after_rebuild",
                text=out_text2,
                max_chars=int(args.wandb_text_max_chars),
            )
            _wandb_log_text(
                wandb=wandb,
                step=wandb_step,
                key="sample_output_text_2",
                text=out_text2,
                max_chars=int(args.wandb_text_max_chars),
            )
            _wandb_log_text_table(
                wandb=wandb,
                step=wandb_step,
                key="sample_texts",
                rows=[
                    ("prompt", prompt),
                    ("output_1", out_text),
                    ("output_2", out_text2),
                ],
                max_chars=int(args.wandb_text_max_chars),
            )
            _wandb_summary_set(wandb=wandb, key="determinism/same_output", value=int(same_output))
            if out_text2_sha256 is not None:
                _wandb_summary_set(
                    wandb=wandb, key="sample/output_text_after_rebuild_sha256", value=out_text2_sha256
                )
            if out_text2 is not None:
                summary_text2, _ = _truncate_text_middle(out_text2, max_chars=int(args.wandb_text_max_chars))
                _wandb_summary_set(
                    wandb=wandb, key="sample/output_text_after_rebuild", value=summary_text2
                )
                _wandb_summary_set(
                    wandb=wandb,
                    key="sample/output_text_after_rebuild_chars",
                    value=int(len(out_text2)),
                )

            _wandb_save_text_files(
                wandb=wandb,
                workdir=workdir,
                prompt=prompt,
                prompt_sha256=prompt_sha256,
                out_text=out_text,
                out_text_sha256=out_text_sha256,
                out_text2=out_text2,
                out_text2_sha256=out_text2_sha256,
            )

            if args.assert_same_output and not same_output:
                raise SystemExit(1)
            wandb_step += 1

            # Cleanup the extra param_dict to return to 1-param steady state.
            del param_dict2
            gc.collect()
            mem = _collect_memory_snapshot()
            elapsed_s = time.time() - t0
            print(
                json.dumps(
                    {
                        "phase": "after_drop_param_dict2",
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
                phase="after_drop_param_dict2",
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
