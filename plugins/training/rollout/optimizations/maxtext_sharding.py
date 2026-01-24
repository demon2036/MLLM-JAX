from __future__ import annotations

from functools import partial
from typing import Any, Callable


def _dp_only_global_array_builder(*, global_mesh: Any) -> tuple[Callable[..., Any], Callable[[Any], Any]]:
    """Return (with_path_fn, no_path_fn) builders for dp-only sharded arrays.

    dp-only here means: shard the leading (batch) axis across `dp`, and replicate
    across `fsdp`/`tp`. This mirrors MaxText's common "data vs fsdp" separation
    and avoids mixing `fsdp` into the batch axis sharding.

    Important constraint: this builder assumes each process owns exactly one `dp`
    slice (i.e. `mesh.shape['dp'] == jax.process_count()`). This allows forming a
    global array from process-local host arrays without a cross-process all-gather.
    """

    import numpy as np

    import jax
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as PS

    dp_size = int(getattr(global_mesh, "shape", {}).get("dp", 1))
    process_count = int(jax.process_count())
    if dp_size != process_count:
        raise ValueError(
            "dp-only rollout sharding requires mesh dp axis to equal jax.process_count() "
            f"(got dp={dp_size}, process_count={process_count}). "
            "Use `mesh_shape: host_local` (dp=process_count) or an explicit mesh like `4,4,1` on v6e-16."
        )

    local_devices = list(getattr(global_mesh, "local_devices", ())) or list(jax.local_devices())
    if not local_devices:
        raise RuntimeError("global_mesh has no local_devices; cannot place arrays.")

    sharding = NamedSharding(global_mesh, PS("dp"))

    def _form_global_array_dp_only(path, array: Any) -> Any:
        del path
        host_array = np.asarray(array)
        if host_array.ndim == 0:
            # Replicate scalars without inventing a fake batch dimension.
            scalar_sharding = NamedSharding(global_mesh, PS())
            local_device_buffers = jax.device_put([host_array] * len(local_devices), local_devices)
            return jax.make_array_from_single_device_arrays(host_array.shape, scalar_sharding, local_device_buffers)

        global_shape = (process_count * int(host_array.shape[0]),) + tuple(int(x) for x in host_array.shape[1:])
        local_device_buffers = jax.device_put([host_array] * len(local_devices), local_devices)
        return jax.make_array_from_single_device_arrays(global_shape, sharding, local_device_buffers)

    with_path_fn = _form_global_array_dp_only
    no_path_fn = partial(_form_global_array_dp_only, ())
    return with_path_fn, no_path_fn


def _dp_only_sample_fn_builder(*, mesh: Any) -> Any:
    import jax
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as PS

    from MLLM_JAX.sample.sanple_utils import _top_k_sampling_batched

    dp = int(getattr(mesh, "shape", {}).get("dp", 1))
    if dp <= 0:
        raise ValueError(f"Invalid mesh dp axis size: {dp}")

    def warp_sample_fn(rng, logits):
        rngs = jax.random.split(rng, dp)

        def sample_inner(rng_local, logits_local):
            return _top_k_sampling_batched(rng_local[0], logits_local, t=1.0)

        sample_fn = shard_map(
            sample_inner,
            mesh=mesh,
            in_specs=(PS("dp"), PS("dp", "tp")),
            out_specs=PS("dp"),
            check_rep=False,
        )
        return sample_fn(rngs, logits)

    return jax.jit(warp_sample_fn)


def patch_sampler_rollout_dp_only_sharding(sampler: Any) -> None:
    """Patch a Sampler-like object to use dp-only rollout (MaxText-style).

    This only affects how prompt/caches are placed onto the mesh for generation.
    Params remain sharded per the model partition rules.
    """

    if getattr(sampler, "_rollout_dp_only_sharding_patched", False):
        return

    mesh = getattr(sampler, "mesh", None)
    if mesh is None:
        raise AttributeError("sampler.mesh is required for dp-only sharding patch")

    with_path_fn, no_path_fn = _dp_only_global_array_builder(global_mesh=mesh)
    sampler.global_collect_method_with_path = with_path_fn
    sampler.global_collect_method = no_path_fn

    sampler.sample_fn = _dp_only_sample_fn_builder(mesh=mesh)

    sampler._rollout_dp_only_sharding_patched = True


__all__ = ["patch_sampler_rollout_dp_only_sharding"]
