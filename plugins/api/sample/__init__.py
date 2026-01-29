"""Sampling/generation contracts (backend-agnostic)."""

from plugins.api.sample.backend import GenerationBackend
from plugins.api.sample.generation import GenerationRequest, GenerationResult

__all__ = ["GenerationBackend", "GenerationRequest", "GenerationResult"]

