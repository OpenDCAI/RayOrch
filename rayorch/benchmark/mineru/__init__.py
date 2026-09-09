# pyright: reportUnsupportedDunderAll=false
"""MinerU UDF group with lazy pipeline and adapter imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_MODULES = {
    "adapters": "rayorch.benchmark.mineru.adapters",
    "pipeline": "rayorch.benchmark.mineru.pipeline",
    "plugin": "rayorch.benchmark.mineru.plugin",
    "poison": "rayorch.benchmark.mineru.poison",
    "udfs": "rayorch.benchmark.mineru.udfs",
}
__all__ = ["adapters", "pipeline", "plugin", "poison", "udfs"]


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
