import json
import os
import sys
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


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    from sgl_jax.srt.entrypoints.engine import Engine

    engine = Engine(
        model_path="Qwen/Qwen3-4B",
        tokenizer_path="Qwen/Qwen3-4B",
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
        download_dir="/root/MLLM-JAX/workdir/hf_download",
    )

    try:
        model_runner = engine.scheduler_info["scheduler"].tp_worker.worker.model_runner
        print(
            json.dumps(
                {
                    "phase": "engine_ready_dummy",
                    "num_model_state_leaves": len(model_runner.model_state_leaves),
                },
                ensure_ascii=False,
            )
        )

        from plugins.sglang_jax_inference.engine_weight_swap import swap_engine_weights_from_hf

        snapshot_dir, num_leaves = swap_engine_weights_from_hf(
            engine=engine,
            model_id_or_path="Qwen/Qwen3-4B",
            cache_dir="/root/MLLM-JAX/workdir/hf_models",
        )
        print(
            json.dumps(
                {
                    "phase": "weights_swapped",
                    "snapshot_dir": snapshot_dir,
                    "num_model_state_leaves": num_leaves,
                },
                ensure_ascii=False,
            )
        )

        prompt = "你是谁"
        sampling_params = {"temperature": 0.0, "max_new_tokens": 64}

        result = engine.generate(prompt=prompt, sampling_params=sampling_params)
        print(
            json.dumps(
                {
                    "phase": "generate_result",
                    "prompt": prompt,
                    "sampling_params": sampling_params,
                    "text": _best_effort_extract_text(result),
                    "raw": result,
                },
                ensure_ascii=False,
                default=str,
            )
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
