from __future__ import annotations

from pathlib import Path

from projects.minionerec_jax_v2.evaluator import (
    _build_prefill_buckets,
    _load_progress_state,
    _normalize_prefill_mode,
    _save_progress_state,
)


def test_normalize_prefill_mode_accepts_exact_aliases() -> None:
    assert _normalize_prefill_mode("exact") == "exact"
    assert _normalize_prefill_mode("per-length") == "exact"
    assert _normalize_prefill_mode("per_length") == "exact"
    assert _normalize_prefill_mode("length") == "exact"


def test_build_prefill_buckets_exact_groups_by_prompt_length() -> None:
    prompt_lens = [76, 91, 91, 121, 76]
    buckets = _build_prefill_buckets(
        prompt_lens,
        max_cache_length=4096,
        suffix_len=2,
        prefill_mode="exact",
        fixed_prefill_len=None,
    )
    assert buckets == {
        76: [0, 4],
        91: [1, 2],
        121: [3],
    }


def test_progress_state_roundtrip_restores_only_existing_indices(tmp_path: Path) -> None:
    progress_path = tmp_path / "eval.progress.json"
    predictions: list[list[str] | None] = [None, ["sid-1", "sid-2"], None, ["sid-3"]]

    _save_progress_state(
        path=progress_path,
        predictions=predictions,
        num_beams=50,
        topk=[3, 5, 10],
        do_sample=False,
        temperature=1.0,
    )

    restored = _load_progress_state(
        path=progress_path,
        n_samples=4,
        num_beams=50,
        topk=[3, 5, 10],
        do_sample=False,
        temperature=1.0,
    )
    assert restored == predictions


def test_progress_state_mismatched_n_samples_resets(tmp_path: Path) -> None:
    progress_path = tmp_path / "eval.progress.json"
    _save_progress_state(
        path=progress_path,
        predictions=[["sid-a"], None, ["sid-b"]],
        num_beams=50,
        topk=[3, 5, 10],
        do_sample=False,
        temperature=1.0,
    )

    restored = _load_progress_state(path=progress_path, n_samples=2)
    assert restored == [None, None]


def test_progress_state_mismatched_decode_sampling_mode_resets(tmp_path: Path) -> None:
    progress_path = tmp_path / "eval.progress.json"
    _save_progress_state(
        path=progress_path,
        predictions=[["sid-a"], ["sid-b"]],
        num_beams=50,
        topk=[3, 5, 10],
        do_sample=False,
        temperature=1.0,
    )

    restored = _load_progress_state(
        path=progress_path,
        n_samples=2,
        num_beams=50,
        topk=[3, 5, 10],
        do_sample=True,
        temperature=1.0,
    )

    assert restored == [None, None]
