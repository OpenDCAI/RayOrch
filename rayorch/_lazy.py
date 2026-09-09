"""Small PEP 562 helpers used by RayOrch's optional public surfaces."""

from __future__ import annotations

from importlib import import_module
from collections.abc import Iterable
from typing import Any, MutableMapping


def resolve_export(
    name: str,
    exports: dict[str, tuple[str, str]],
    namespace: MutableMapping[str, Any],
) -> Any:
    """Import one requested attribute and cache it in the caller namespace."""

    try:
        module_name, attribute = exports[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    namespace[name] = value
    return value


def public_dir(
    namespace: MutableMapping[str, Any], exports: Iterable[str]
) -> list[str]:
    """Return deterministic names for modules implementing lazy ``__dir__``."""

    return sorted(set(namespace) | set(exports))


__all__ = ["public_dir", "resolve_export"]
