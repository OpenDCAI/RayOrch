"""Lazy benchmark namespace; importing it never imports workload dependencies."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .registry import available, get_plugin, register_plugin

__all__ = ["available", "get_plugin", "register_plugin"]


def __getattr__(name: str) -> Any:
    if name != "mineru":
        raise AttributeError(name)
    module = import_module("rayorch.benchmark.mineru")
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | {"mineru"})
