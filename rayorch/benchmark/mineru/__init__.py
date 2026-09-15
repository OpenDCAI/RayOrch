# pyright: reportUnsupportedDunderAll=false
"""MinerU UDF group with lazy runtime imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_MODULES = {
    "job": "rayorch.benchmark.mineru.job",
    "pipeline": "rayorch.benchmark.mineru.pipeline",
    "plugin": "rayorch.benchmark.mineru.plugin",
    "runner": "rayorch.benchmark.mineru.runner",
    "udfs": "rayorch.benchmark.mineru.udfs",
    "validate": "rayorch.benchmark.mineru.validate",
}
__all__ = list(_MODULES)


def __getattr__(name: str) -> Any:
    try:
        module_name = _MODULES[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    module = import_module(module_name)
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
