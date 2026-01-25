from __future__ import annotations

from typing import Any, Callable

import numpy as np


def make_form_training_global_array(mesh: Any) -> Callable[[Any, Any], Any]:
    """Return a `tree_map_with_path` callback that builds a sharded global JAX array.

    This mirrors the GRPO runner's inlined `_form_training_global_array` helper:
    - Prefer sharding the batch axis across (dp, fsdp) and replicate across tp.
    - Fall back to sharding across all mesh axes when the dp/fsdp mapping is
      ambiguous (multi-host + tp spanning host boundaries).
    """
    import jax
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as PS

    mesh_devices = np.asarray(mesh.devices)
    device_to_coords: dict[Any, tuple[int, ...]] = {dev: coords for coords, dev in np.ndenumerate(mesh_devices)}
    local_devices = list(mesh.local_devices)
    local_dp_fsdp_coords = sorted({device_to_coords[d][:2] for d in local_devices})
    local_dp_fsdp_count = len(local_dp_fsdp_coords)

    process_count = int(jax.process_count())
    tp_size = int(mesh.shape.get("tp", 1))
    dp_fsdp_total = int(mesh.shape.get("dp", 1)) * int(mesh.shape.get("fsdp", 1))

    use_dp_fsdp_batch_sharding = True
    if process_count != 1 and tp_size > 1 and local_dp_fsdp_count * process_count != dp_fsdp_total:
        use_dp_fsdp_batch_sharding = False

    def _form(path, array: Any):
        array = np.asarray(array)
        if array.ndim == 0:
            raise ValueError(
                f"Expected batched array at {jax.tree_util.keystr(path)}, got scalar shape={array.shape}."
            )

        global_shape = (jax.process_count() * int(array.shape[0]),) + tuple(int(x) for x in array.shape[1:])

        if use_dp_fsdp_batch_sharding:
            if local_dp_fsdp_count <= 0:
                raise RuntimeError("Unable to infer local dp/fsdp shard count from mesh.local_devices")
            if int(array.shape[0]) % local_dp_fsdp_count != 0:
                raise ValueError(
                    f"Unable to shard batch={int(array.shape[0])} across local (dp,fsdp) shards={local_dp_fsdp_count} "
                    f"at {jax.tree_util.keystr(path)}. Consider adjusting rollout/train batch sizes."
                )
            shard_chunks = np.split(array, local_dp_fsdp_count, axis=0) if local_dp_fsdp_count > 1 else [array]
            coord_to_chunk = dict(zip(local_dp_fsdp_coords, shard_chunks, strict=True))
            local_device_arrays = [coord_to_chunk[device_to_coords[d][:2]] for d in local_devices]
            local_device_buffers = jax.device_put(local_device_arrays, local_devices)
            sharding = NamedSharding(mesh, PS(("dp", "fsdp")))
            return jax.make_array_from_single_device_arrays(global_shape, sharding, local_device_buffers)

        if int(array.shape[0]) % len(local_devices) != 0:
            raise ValueError(
                f"Unable to shard batch={int(array.shape[0])} across local devices={len(local_devices)} "
                f"at {jax.tree_util.keystr(path)}. Consider adjusting rollout/train batch sizes."
            )
        local_device_arrays = np.split(array, len(local_devices), axis=0) if len(local_devices) > 1 else [array]
        local_device_buffers = jax.device_put(local_device_arrays, local_devices)
        sharding = NamedSharding(mesh, PS(mesh.axis_names))
        return jax.make_array_from_single_device_arrays(global_shape, sharding, local_device_buffers)

    return _form


def local_from_global(array: Any) -> np.ndarray:
    """Best-effort extraction of the local batch from a sharded JAX array."""
    shards = getattr(array, "addressable_shards", None)
    if not shards:
        return np.asarray(array)
    first_index = shards[0].index
    if all(shard.index == first_index for shard in shards):
        return np.asarray(shards[0].data)
    shards_sorted = sorted(shards, key=lambda s: s.index[0].start)
    return np.concatenate([np.asarray(shard.data) for shard in shards_sorted], axis=0)


__all__ = ["make_form_training_global_array", "local_from_global"]

