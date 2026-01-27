import pytest

pytest.importorskip("jax")

from plugins.sft.jax.evaluator import _build_prefill_buckets, _normalize_prefill_mode


def test_normalize_prefill_mode_aliases():
    assert _normalize_prefill_mode("bucket") == "bucket"
    assert _normalize_prefill_mode("buckets") == "bucket"
    assert _normalize_prefill_mode("fixed") == "fixed"
    assert _normalize_prefill_mode("single") == "fixed"
    assert _normalize_prefill_mode("one") == "fixed"


def test_fixed_prefill_falls_back_to_exact_len_when_bucket_too_large():
    # max_prompt_len=10, but the next bucket is 128 which would not fit in a tiny max_cache_length.
    buckets = _build_prefill_buckets(
        [5, 10, 7],
        max_cache_length=64,
        suffix_len=2,
        prefill_mode="fixed",
        fixed_prefill_len=None,
    )
    assert list(buckets.keys()) == [10]
    assert buckets[10] == [0, 1, 2]


def test_fixed_prefill_rejects_too_small_explicit_len():
    with pytest.raises(ValueError, match=r"fixed prefill_len too small"):
        _build_prefill_buckets(
            [5, 10, 7],
            max_cache_length=64,
            suffix_len=2,
            prefill_mode="fixed",
            fixed_prefill_len=8,
        )


def test_bucket_prefill_groups_by_ceil_buckets():
    buckets = _build_prefill_buckets(
        [100, 120, 200],
        max_cache_length=512,
        suffix_len=2,
        prefill_mode="bucket",
        fixed_prefill_len=None,
    )
    assert buckets[128] == [0, 1]
    assert buckets[256] == [2]


def test_unknown_prefill_mode_raises():
    with pytest.raises(ValueError, match=r"Unknown prefill_mode"):
        _build_prefill_buckets(
            [5],
            max_cache_length=64,
            suffix_len=2,
            prefill_mode="wat",
            fixed_prefill_len=None,
        )

