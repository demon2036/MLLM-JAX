import argparse
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
        # Linux: /proc/self/status contains "VmRSS:\t  12345 kB"
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
        # Avoid breaking inference on logging failures.
        return


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    workdir = repo_root / "workdir"
    download_dir = workdir / "hf_download"
    hf_cache_dir = workdir / "hf_models"
    model_id = "Qwen/Qwen3-4B"

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="你是谁")
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="启用 Weights & Biases 记录（仅 jax.process_index()==0）。",
    )
    parser.add_argument(
        "--wandb-project",
        default="sglang-jax-weight-swap-memory",
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

    workdir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)
    hf_cache_dir.mkdir(parents=True, exist_ok=True)

    from sgl_jax.version import __version__ as sglang_jax_version
    from sgl_jax.srt.entrypoints.engine import Engine

    t0 = time.time()
    wandb_step = 0

    wandb = None
    wandb_service = None
    if args.wandb:
        try:
            import jax

            if jax.process_index() == 0 and os.environ.get("WANDB_MODE") != "disabled":
                import wandb as _wandb

                run_name = str(args.wandb_name).strip()
                if run_name == "":
                    run_name = (
                        "qwen3_4b_param_swap_"
                        + datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
                    )
                init_kwargs = {
                    "project": str(args.wandb_project),
                    "name": run_name,
                    "config": {
                        "model_id": model_id,
                        "tp_size": 4,
                        "dp_size": 1,
                        "dtype": "bfloat16",
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
        model_path=model_id,
        tokenizer_path=model_id,
        trust_remote_code=True,
        device="tpu",
        tp_size=4,
        dp_size=1,
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
                    "model_id": model_id,
                    "device": "tpu",
                    "tp_size": 4,
                    "dp_size": 1,
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

        from plugins.sglang_jax_inference.engine_weight_swap import swap_engine_weights_from_hf

        try:
            snapshot_dir, num_leaves = swap_engine_weights_from_hf(
                engine=engine,
                model_id_or_path=model_id,
                cache_dir=str(hf_cache_dir),
            )
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "phase": "weights_swap_error",
                        "error": str(exc),
                        "hint": (
                            "If this is an HF download issue, verify TPU egress, "
                            "ensure `huggingface_hub[hf_transfer]` is installed, "
                            "and retry with `HF_HUB_ENABLE_HF_TRANSFER=1`."
                        ),
                        "elapsed_s": time.time() - t0,
                        "memory": _collect_memory_snapshot(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            raise
        mem = _collect_memory_snapshot()
        elapsed_s = time.time() - t0
        print(
            json.dumps(
                {
                    "phase": "weights_swapped",
                    "snapshot_dir": snapshot_dir,
                    "num_model_state_leaves": num_leaves,
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
            phase="weights_swapped",
            elapsed_s=elapsed_s,
            mem=mem,
            extra={"num_model_state_leaves": int(num_leaves)},
        )
        wandb_step += 1

        prompt = args.prompt
        sampling_params = {"temperature": 0.0, "max_new_tokens": 64}

        result = engine.generate(prompt=prompt, sampling_params=sampling_params)
        mem = _collect_memory_snapshot()
        elapsed_s = time.time() - t0
        print(
            json.dumps(
                {
                    "phase": "generate_result",
                    "prompt": prompt,
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
