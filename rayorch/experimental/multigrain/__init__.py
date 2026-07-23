"""Public authoring facade for the experimental multigrain dataflow.

Advanced passive-IR APIs live under :mod:`multigrain.ir`; local execution and
metrics under :mod:`multigrain.execution`; the optional Ray backend is loaded
only when one of its public symbols is first accessed.
"""
from __future__ import annotations

from typing import Any

from .data import (
    ErrorTrace,
    Grouped,
    IdentityDomain,
    PortBatch,
    concat,
    group_by,
    rebatch,
    source,
    via,
)
from .execution import MultigrainExecutor, NodeMetric, RunMetrics, StreamScope
from .ir import (
    IncompleteGroupPolicy,
    IsolationBudget,
    IsolationExhaustedAction,
    RecoveryPolicy,
    RetryTiming,
    ShardExhaustedAction,
    WorkerPoolSpec,
)
from .primitives import Expand, Filter, Map, Reduce, Relate, Select
from .tracing import Pipeline

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
    "Expand",
    "Filter",
    "FaultSpec",
    "Grouped",
    "IdentityDomain",
    "IncompleteGroupPolicy",
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
    "StreamScope",
    "WorkerPoolSpec",
    "concat",
    "group_by",
    "lpt_shard_planner",
    "rebatch",
    "source",
    "via",
]
