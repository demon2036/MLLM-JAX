from __future__ import annotations

from typing import Any, Protocol, Sequence

from plugins.training.api import RolloutResult


class RolloutBackend(Protocol):
    """A rollout engine that turns prompts into (answers, training batch).

    The backend owns whatever state it needs to generate (in-process sampler,
    external engine client, etc). Callers provide the latest `params` so the
    backend can implement optional weight syncing strategies.

    Optional hooks (duck-typed, used by the runner when present):
    - `initialize()`: eager backend initialization (e.g. spin up an engine).
    - `sync_weights(params)`: push the latest policy weights into the backend.
    - `flush_cache()`: release KV/cache memory after rollout.
    - `shutdown()`: teardown backend resources.
    """

    def rollout(
        self,
        *,
        prompts: Sequence[str],
        params: Any,
        system_prompt: str,
        global_length: int,
        max_length_sample: int,
    ) -> RolloutResult: ...
