"""Public authoring facade for the experimental multigrain dataflow.

Advanced passive-IR APIs live under :mod:`multigrain.ir`; local execution and
metrics under :mod:`multigrain.execution`; the optional Ray backend is loaded
only when one of its public symbols is first accessed.
"""
from __future__ import annotations

from typing import Any

from .data import ErrorTrace, Grouped, PortBatch, concat, group_by, rebatch, source
from .execution import MultigrainExecutor, NodeMetric, RunMetrics
from .ir import (
    DrainScope,
    IsolationBudget,
    IsolationExhaustedAction,
    Pipeline,
    RecoveryPolicy,
    RetryTiming,
    ShardExhaustedAction,
)
from .primitives import Expand, Filter, Map, Reduce, Relate, Select

_RAY_EXPORTS = frozenset(
    {"FaultSpec", "MultigrainRayExecutor", "lpt_shard_planner"}
)


def __getattr__(name: str) -> Any:
    if name in _RAY_EXPORTS:
        from . import ray as ray_backend

        value = getattr(ray_backend, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ErrorTrace",
    "DrainScope",
    "Expand",
    "Filter",
    "FaultSpec",
    "Grouped",
    "IsolationBudget",
    "IsolationExhaustedAction",
    "Map",
    "MultigrainExecutor",
    "MultigrainRayExecutor",
    "NodeMetric",
    "Pipeline",
    "PortBatch",
    "Reduce",
    "RecoveryPolicy",
    "Relate",
    "RunMetrics",
    "RetryTiming",
    "Select",
    "ShardExhaustedAction",
    "concat",
    "group_by",
    "lpt_shard_planner",
    "rebatch",
    "source",
]
