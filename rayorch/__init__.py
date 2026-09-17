"""Stable public API for authoring and executing RayOrch dataflows."""

from . import benchmark, functional as F
from ._execution.executor import Executor
from ._model import ItemOutcome, MISSING
from .api import Pipeline, Port, RayModule, function
from .errors import CompileError, ExecutionError
from .failures import GroupFailure, RecordFailure
from .recovery import RecoveryPolicy
from .result import OutputIssue, RunResult
from .runner import run
from .version import __version__, version_info

__all__ = [
    "CompileError",
    "ExecutionError",
    "Executor",
    "F",
    "GroupFailure",
    "ItemOutcome",
    "MISSING",
    "OutputIssue",
    "Pipeline",
    "Port",
    "RayModule",
    "RecordFailure",
    "RecoveryPolicy",
    "RunResult",
    "__version__",
    "benchmark",
    "function",
    "run",
    "version_info",
]
