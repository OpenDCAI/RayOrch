"""Lazy registry for benchmark UDF groups."""

from __future__ import annotations

from importlib import import_module

from .plugin import BenchmarkPlugin


_PLUGIN_MODULES = {"mineru": "rayorch.benchmark.mineru.plugin"}


def available() -> tuple[str, ...]:
    return tuple(sorted(_PLUGIN_MODULES))


def get_plugin(name: str) -> BenchmarkPlugin:
    try:
        module_name = _PLUGIN_MODULES[name]
    except KeyError as exc:
        choices = ", ".join(available())
        raise KeyError(f"unknown benchmark {name!r}; available: {choices}") from exc
    plugin = import_module(module_name).PLUGIN
    if not isinstance(plugin, BenchmarkPlugin):
        raise TypeError(f"{module_name}.PLUGIN is not a BenchmarkPlugin")
    return plugin


__all__ = ["available", "get_plugin"]
