"""以 Port/Domain 为核心抽象的实验性 Multigrain v3.5。"""

from . import functional
from .api import OptionalPort, Pipeline, Port, RayModule, function
from .model import (
    CallRef,
    CompileError,
    DomainRef,
    EntityRef,
    GrainRef,
    InputMode,
    ItemRef,
    MISSING,
    PortRef,
)
from .logical import LogicalProgram
from .plan import CompiledProgram, RuntimePlan
from .executor import Executor
from .worker import WorkerObservation

# 文档与论文伪代码统一使用 F.expand/F.reduce；它只是 functional 模块别名。
F = functional

__all__ = [
    "CallRef",
    "CompileError",
    "CompiledProgram",
    "DomainRef",
    "EntityRef",
    "F",
    "GrainRef",
    "InputMode",
    "ItemRef",
    "LogicalProgram",
    "MISSING",
    "OptionalPort",
    "Pipeline",
    "Port",
    "PortRef",
    "RuntimePlan",
    "WorkerObservation",
    "Executor",
    "RayModule",
    "function",
    "functional",
]
