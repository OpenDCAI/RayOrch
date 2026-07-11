"""Experimental multi-grain dataflow prototype.

The package-level API intentionally exposes only the user-facing authoring
surface. IR data structures and passes remain available from the ``graph`` and
``passes`` submodules for tests, debugging, and research work.
"""
from .core import ErrorTrace, Grouped, PortBatch, concat, group_by, rebatch, source
from .executor import MultigrainExecutor
from .graph import Pipeline
from .ops import Expand, Filter, Map, Reduce, Relate, Select

__all__ = [
    "ErrorTrace",
    "Expand",
    "Filter",
    "Grouped",
    "Map",
    "MultigrainExecutor",
    "Pipeline",
    "PortBatch",
    "Reduce",
    "Relate",
    "Select",
    "concat",
    "group_by",
    "rebatch",
    "source",
]
