"""Lazy registry for benchmark UDF groups."""

from __future__ import annotations

from importlib import import_module

from .plugin import BenchmarkPlugin


_PLUGIN_MODULES = {"mineru": "rayorch.benchmark.mineru.plugin"}


def available() -> tuple[str, ...]:
    return tuple(sorted(_PLUGIN_MODULES))


def register_plugin(
    name: str,
    module: str,
    *,
    exist_ok: bool = False,
) -> None:
    """Register a lightweight metadata module without importing it."""

    if not name or not module:
        raise ValueError("plugin name and module must be non-empty")
    if name in _PLUGIN_MODULES and not exist_ok:
        raise ValueError(f"benchmark {name!r} is already registered")
    _PLUGIN_MODULES[name] = module


def get_plugin(name: str) -> BenchmarkPlugin:
    try:
        module_name = _PLUGIN_MODULES[name]
    except KeyError as exc:
        choices = ", ".join(available())
        raise KeyError(f"unknown benchmark {name!r}; available: {choices}") from exc
    plugin = import_module(module_name).PLUGIN
    if not isinstance(plugin, BenchmarkPlugin):
        raise TypeError(f"{module_name}.PLUGIN is not a BenchmarkPlugin")
    if plugin.name != name:
        raise ValueError(
            f"{module_name}.PLUGIN declares {plugin.name!r}, expected {name!r}"
        )
    return plugin


__all__ = ["available", "get_plugin", "register_plugin"]
