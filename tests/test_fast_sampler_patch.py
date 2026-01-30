from __future__ import annotations


def test_patch_sampler_generate_fast_is_idempotent():
    from plugins.training.rl.rollout.optimizations import patch_sampler_generate_fast

    class DummySampler:
        def generate(self, *args, **kwargs):
            return {"ok": True}

    sampler = DummySampler()
    original_func = sampler.generate.__func__

    patch_sampler_generate_fast(sampler)
    assert getattr(sampler, "_fast_generate_patched", False) is True
    saved = sampler._fast_generate_original  # type: ignore[attr-defined]
    assert getattr(saved, "__self__", None) is sampler
    assert getattr(saved, "__func__", None) is original_func
    assert sampler.generate.__func__ is not original_func

    patch_sampler_generate_fast(sampler)
    saved2 = sampler._fast_generate_original  # type: ignore[attr-defined]
    assert getattr(saved2, "__self__", None) is sampler
    assert getattr(saved2, "__func__", None) is original_func


def test_fast_generate_timing_is_guarded_by_env_var():
    from plugins.training.rl.rollout.optimizations import patch_sampler_generate_fast

    class DummySampler:
        def generate(self, *args, **kwargs):
            return {"ok": True}

    sampler = DummySampler()
    patch_sampler_generate_fast(sampler)

    code = sampler.generate.__func__.__code__
    assert "PRINT_SAMPLER_GENERATE_TIMING" in code.co_consts
