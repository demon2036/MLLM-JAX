from __future__ import annotations

from typing import Any, Protocol, Sequence

from plugins.training.api import RolloutResult


class RolloutBackend(Protocol):
    """A rollout engine that turns prompts into (answers, training batch).

    The backend owns whatever state it needs to generate (in-process sampler,
    external engine client, etc). Callers provide the latest `params` so the
    backend can implement optional weight syncing strategies.
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

