# pyright: reportUnsupportedDunderAll=false
"""RayOrch public API with optional components loaded on first attribute use."""

from __future__ import annotations

from typing import Any

from ._lazy import public_dir, resolve_export
from .version import __version__, version_info


_MODULES = {
    "benchmark": "rayorch.benchmark",
    "multigrain": "rayorch.multigrain",
}

_EXPORTS = {
    "DispatchMode": ("rayorch.dispatch_mode", "DispatchMode"),
    "get_predefined_dispatch_fn": (
        "rayorch.dispatch_mode",
        "get_predefined_dispatch_fn",
    ),
    "Dispatch": ("rayorch.dispatch_mode", "Dispatch"),
    "RayModule": ("rayorch.ray_module", "RayModule"),
    "DagNode": ("rayorch.dag_pipeline", "DagNode"),
    "DagPipeline": ("rayorch.dag_pipeline", "DagPipeline"),
    "DagPipelineExecutor": ("rayorch.dag_pipeline", "DagPipelineExecutor"),
    "PipelineExecutor": ("rayorch.dag_pipeline", "PipelineExecutor"),
    "PipeRef": ("rayorch.dag_pipeline", "PipeRef"),
    "Pipeline": ("rayorch.dag_new_pipeline", "Pipeline"),
    "DagNewPipeline": ("rayorch.dag_new_pipeline", "DagPipeline"),
    "DagNewPipeRef": ("rayorch.dag_new_pipeline", "PipeRef"),
    "Executor": ("rayorch.dag_new_pipeline", "Executor"),
    "SequentialExecutor": ("rayorch.dag_new_pipeline", "SequentialExecutor"),
    "DagExecutor": ("rayorch.dag_new_pipeline", "DagExecutor"),
    "OverlappedPipeline": ("rayorch.overlapped_pipeline", "OverlappedPipeline"),
    "EnvRegistry": ("rayorch.env_registry", "EnvRegistry"),
}

__all__ = [
    "__version__",
    "version_info",
    "DispatchMode",
    "get_predefined_dispatch_fn",
    "Dispatch",
    "RayModule",
    "RayModuleFuture",
    "DagNode",
    "DagPipeline",
    "DagPipelineExecutor",
    "Pipeline",
    "DagNewPipeline",
    "DagNewPipeRef",
    "Executor",
    "SequentialExecutor",
    "DagExecutor",
    "OverlappedPipeline",
    "PipelineExecutor",
    "PipeRef",
]


def __getattr__(name: str) -> Any:
    if name in _MODULES:
        from importlib import import_module

        value = import_module(_MODULES[name])
        globals()[name] = value
        return value
    if name == "RayModuleFuture":
        value = resolve_export("RayModule", _EXPORTS, globals()).RayModuleFuture
        globals()[name] = value
        return value
    return resolve_export(name, _EXPORTS, globals())


def __dir__() -> list[str]:
    return public_dir(globals(), (*_EXPORTS, *_MODULES, "RayModuleFuture"))


def hello() -> str:
    return "Hello from RayOrch!"
