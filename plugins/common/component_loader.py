from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class ComponentSpec:
    """A normalized component specification.

    Supports 3 user-facing forms:
    - Builtin alias:  "naive"
    - Import path:    "pkg.module:Symbol"
    - Dict form:      {"target": "...", "kwargs": {...}}
    """

    target: str
    kwargs: dict[str, Any]


def _as_dict(x: Any) -> dict[str, Any] | None:
    return x if isinstance(x, dict) else None


def normalize_component_spec(spec: Any, *, default_target: str | None = None) -> ComponentSpec:
    if spec is None:
        if default_target is None:
            raise ValueError("Component spec is None and no default_target was provided.")
        return ComponentSpec(target=str(default_target), kwargs={})

    if isinstance(spec, str):
        return ComponentSpec(target=str(spec).strip(), kwargs={})

    spec_dict = _as_dict(spec)
    if spec_dict is None:
        raise TypeError(f"Unsupported component spec type: {type(spec).__name__} (expected str|dict|None)")

    target = spec_dict.get("target") or spec_dict.get("name") or spec_dict.get("type")
    if target is None:
        raise ValueError("Component spec dict must contain 'target' (or legacy 'name'/'type').")
    kwargs = spec_dict.get("kwargs") or {}
    if not isinstance(kwargs, dict):
        raise TypeError("Component spec 'kwargs' must be a dict.")

    return ComponentSpec(target=str(target).strip(), kwargs=dict(kwargs))


def load_symbol(target: str) -> Any:
    """Load a Python symbol from an import path: 'pkg.module:Symbol'."""
    target = str(target).strip()
    if ":" not in target:
        raise ValueError(f"Expected import-path target 'pkg.module:Symbol', got: {target!r}")
    module_name, symbol_name = target.split(":", 1)
    if not module_name or not symbol_name:
        raise ValueError(f"Invalid import-path target: {target!r}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, symbol_name)
    except AttributeError as e:
        raise AttributeError(f"Symbol {symbol_name!r} not found in module {module_name!r}") from e


def instantiate_component(
    spec: Any,
    *,
    registry: Mapping[str, Callable[..., Any]] | None = None,
    default_target: str | None = None,
    extra_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Instantiate a component from a builtin alias or import-path target.

    Parameters
    ----------
    spec:
        A component spec in any supported form (see `ComponentSpec`).
    registry:
        Optional mapping for builtin aliases. If `spec.target` is in the registry,
        the registry constructor is used.
    default_target:
        Used when `spec` is None.
    extra_kwargs:
        Additional kwargs merged into `spec.kwargs` (spec wins on conflicts).
    """
    normalized = normalize_component_spec(spec, default_target=default_target)
    target = normalized.target

    merged_kwargs: dict[str, Any] = {}
    if extra_kwargs:
        merged_kwargs.update(dict(extra_kwargs))
    merged_kwargs.update(dict(normalized.kwargs))

    if registry is not None:
        ctor = registry.get(target)
        if ctor is not None:
            return ctor(**merged_kwargs)

    if ":" in target:
        obj = load_symbol(target)
        if not callable(obj):
            raise TypeError(f"Loaded target is not callable: {target!r} ({type(obj).__name__})")
        return obj(**merged_kwargs)

    supported = sorted(registry.keys()) if registry is not None else []
    raise ValueError(
        f"Unknown component target={target!r}. "
        f"Expected an import path 'pkg.module:Symbol' or one of: {supported}"
    )


__all__ = ["ComponentSpec", "instantiate_component", "load_symbol", "normalize_component_spec"]

