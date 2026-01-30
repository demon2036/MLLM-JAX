from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from plugins.common.env import load_dotenv_if_present


_OOM_PATTERNS = [
    re.compile(r"RESOURCE_EXHAUSTED", re.IGNORECASE),
    re.compile(r"out of memory", re.IGNORECASE),
    re.compile(r"\boom\b", re.IGNORECASE),
    re.compile(r"HBM", re.IGNORECASE),
]


def _rm_libtpu_lockfile() -> None:
    try:
        os.remove("/tmp/libtpu_lockfile")
    except FileNotFoundError:
        return
    except Exception:
        return


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _stream_subprocess_to_file(cmd: list[str], *, log_path: str, env: dict[str, str] | None = None) -> tuple[int, bool]:
    oom_hit = False
    Path(os.path.dirname(log_path)).mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
            if not oom_hit:
                for pat in _OOM_PATTERNS:
                    if pat.search(line):
                        oom_hit = True
                        break
        rc = proc.wait()
    return int(rc), bool(oom_hit)


def _parse_last_json(output: str) -> dict:
    """Extract the last JSON object from a noisy stdout blob.

    `scripts/grpo_kernel_bench.py` prints one pretty JSON object at the end,
    but W&B and other loggers can add lines before/after. We recover the last
    full JSON object by scanning backwards from the final '}' and matching
    braces (with minimal string/escape handling).
    """
    end = output.rfind("}")
    if end < 0:
        raise ValueError("No JSON object found in output (missing '}')")

    depth = 0
    in_string = False
    escape = False
    start = None

    for i in range(end, -1, -1):
        ch = output[i]
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "}":
            depth += 1
            continue
        if ch == "{":
            depth -= 1
            if depth == 0:
                start = i
                break

    if start is None:
        raise ValueError("Failed to locate JSON object start in output")
    blob = output[start : end + 1]
    return json.loads(blob)


@dataclass(frozen=True)
class BenchResult:
    impl: str
    avg_step_ms: float
    peak_bytes_reserved: int | None
    peak_bytes_in_use: int | None


def _run_kernel_bench(*, impl: str, wandb_mode: str, wandb_name: str | None) -> BenchResult:
    cmd = [
        sys.executable,
        "-u",
        os.path.join(REPO_ROOT, "scripts", "grpo_kernel_bench.py"),
        "--impl",
        str(impl),
        "--mode",
        "off_policy",
        "--batch",
        "1",
        "--time",
        "4096",
        "--vocab",
        "151643",
        "--dtype",
        "bf16",
        "--iters",
        "2",
        "--warmup",
        "1",
        "--old_logp_noise_scale",
        "0.3",
        "--epsilon_low",
        "0.2",
        "--epsilon_high",
        "0.2",
        "--temperature",
        "1.0",
        "--block_size",
        "2048",
        "--time_block",
        "512",
        "--compute_dtype",
        "bf16",
        "--wandb_mode",
        str(wandb_mode),
    ]
    if wandb_name:
        cmd += ["--wandb_name", str(wandb_name)]
    out = subprocess.check_output(cmd, cwd=REPO_ROOT, stderr=subprocess.STDOUT, text=True)
    parsed = _parse_last_json(out)
    mem = parsed.get("mem_after_run") or {}
    return BenchResult(
        impl=str(impl),
        avg_step_ms=float(parsed["avg_step_ms"]),
        peak_bytes_reserved=(int(mem["peak_bytes_reserved"]) if mem.get("peak_bytes_reserved") is not None else None),
        peak_bytes_in_use=(int(mem["peak_bytes_in_use"]) if mem.get("peak_bytes_in_use") is not None else None),
    )


@dataclass(frozen=True)
class TrainingRunResult:
    label: str
    config: str
    exit_code: int
    oom_detected: bool
    log_path: str
    wall_s: float


def _run_training_once(*, label: str, config_path: str, logs_dir: str) -> TrainingRunResult:
    _rm_libtpu_lockfile()
    t0 = time.perf_counter()
    log_path = os.path.join(logs_dir, f"len2048_{label}_{_now_tag()}.log")
    rc, oom_hit = _stream_subprocess_to_file(
        [sys.executable, "-u", os.path.join(REPO_ROOT, "scripts", "run_grpo_gsm8k_training.py"), "--config", config_path],
        log_path=log_path,
        env=os.environ.copy(),
    )
    wall_s = time.perf_counter() - t0
    return TrainingRunResult(
        label=str(label),
        config=str(config_path),
        exit_code=int(rc),
        oom_detected=bool(oom_hit),
        log_path=str(log_path),
        wall_s=float(wall_s),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config_pallas",
        type=str,
        default="plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps2_len2048_pallas_tb512_bf16_mem.yaml",
    )
    p.add_argument(
        "--config_jax",
        type=str,
        default="plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps2_len2048_jax_mem.yaml",
    )
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "disabled"])
    p.add_argument("--require_kernel_faster", action="store_true")
    args = p.parse_args()

    load_dotenv_if_present(repo_root=REPO_ROOT)

    logs_dir = os.path.join(REPO_ROOT, "logs")
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    print("== Phase 1/3: numeric correctness (pytest) ==")
    _rm_libtpu_lockfile()
    subprocess.check_call([sys.executable, "-m", "pytest", "-q", "tests/test_grpo_pallas_kernel.py"], cwd=REPO_ROOT)

    print("== Phase 2/3: speed gate (kernel micro-bench) ==")
    bench_tag = _now_tag()
    bench_jax = _run_kernel_bench(impl="jax", wandb_mode=args.wandb_mode, wandb_name=f"gate_jax_{bench_tag}")
    bench_kernel = _run_kernel_bench(impl="kernel", wandb_mode=args.wandb_mode, wandb_name=f"gate_kernel_{bench_tag}")
    if args.require_kernel_faster and bench_kernel.avg_step_ms > bench_jax.avg_step_ms:
        raise SystemExit(
            f"Speed gate failed: kernel avg_step_ms={bench_kernel.avg_step_ms:.3f} > "
            f"jax avg_step_ms={bench_jax.avg_step_ms:.3f}"
        )
    print(
        "speed_gate:",
        json.dumps(
            {
                "jax_avg_step_ms": bench_jax.avg_step_ms,
                "kernel_avg_step_ms": bench_kernel.avg_step_ms,
            },
            sort_keys=True,
        ),
    )

    print("== Phase 3/3: memory stress (max_length_sample=2048) ==")
    run_pallas = _run_training_once(label="pallas", config_path=str(args.config_pallas), logs_dir=logs_dir)
    run_jax = _run_training_once(label="jax", config_path=str(args.config_jax), logs_dir=logs_dir)

    summary = {
        "bench_jax": asdict(bench_jax),
        "bench_kernel": asdict(bench_kernel),
        "run_pallas": asdict(run_pallas),
        "run_jax": asdict(run_jax),
    }
    print("== Summary ==")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if run_pallas.exit_code != 0:
        raise SystemExit(f"Pallas run failed (exit={run_pallas.exit_code}). Log: {run_pallas.log_path}")
    if run_jax.exit_code == 0:
        raise SystemExit("JAX run unexpectedly succeeded; cannot claim len2048 OOM avoidance.")
    if not run_jax.oom_detected:
        raise SystemExit(
            f"JAX run failed but OOM not detected in logs (exit={run_jax.exit_code}). Log: {run_jax.log_path}"
        )


if __name__ == "__main__":
    main()
