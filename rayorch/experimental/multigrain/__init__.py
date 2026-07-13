"""Experimental multi-grain dataflow prototype.

The package-level API intentionally exposes only the user-facing authoring
surface. IR data structures and passes remain available from the ``graph`` and
``passes`` submodules for tests, debugging, and research work.
"""
from .core import ErrorTrace, Grouped, PortBatch, concat, group_by, rebatch, source
from .executor import MultigrainExecutor
from .graph import (
    DrainScope,
    IsolationBudget,
    IsolationExhaustedAction,
    Pipeline,
    RecoveryPolicy,
    RetryTiming,
    ShardExhaustedAction,
)
from .metrics import NodeMetric, RunMetrics
from .ops import Expand, Filter, Map, Reduce, Relate, Select
from .ray_executor import FaultSpec, MultigrainRayExecutor, lpt_shard_planner

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
