"""Lazy benchmark namespace; importing it never imports workload dependencies."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .registry import available, get_plugin, register_plugin

__all__ = ["available", "get_plugin", "register_plugin"]


def __getattr__(name: str) -> Any:
    try:
        plugin = get_plugin(name)
    except KeyError as exc:
        raise AttributeError(name) from exc
    module = import_module(plugin.runtime_package)
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(available()))
