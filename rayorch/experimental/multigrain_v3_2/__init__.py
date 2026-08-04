"""Multigrain V3.2：构建在 V3 backend 上的 Port-first 逻辑 domain。"""

from rayorch.experimental.multigrain_v3.contracts import (
    BadRecordError,
    ExecutionError,
    MISSING,
)

from . import functional
from .api import (
    CompiledProgram,
    LogicalCompileError,
    Pipeline,
    Port,
    RayModule,
    optional,
)
from .executor import Executor, RunResult
from .ir import (
    DirectValue,
    DomainId,
    DomainSpec,
    ExpandNode,
    ExpansionId,
    ExpansionSpec,
    GroupValue,
    LogicalDAG,
    LogicalPortId,
    PortSpec,
    ReduceNode,
)

__all__ = [
    "MISSING",
    "BadRecordError",
    "CompiledProgram",
    "DirectValue",
    "DomainId",
    "DomainSpec",
    "ExecutionError",
    "Executor",
    "ExpandNode",
    "ExpansionId",
    "ExpansionSpec",
    "GroupValue",
    "LogicalCompileError",
    "LogicalDAG",
    "LogicalPortId",
    "Pipeline",
    "Port",
    "PortSpec",
    "RayModule",
    "ReduceNode",
    "RunResult",
    "functional",
    "optional",
]
