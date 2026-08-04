"""以 Port/Domain 为核心抽象的实验性 Multigrain v3.3。"""

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
from .executor import LocalExecutor
from .program import CompiledProgram, Program
from .ray_executor import RayExecutor

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
    "LocalExecutor",
    "MISSING",
    "OptionalPort",
    "Pipeline",
    "Port",
    "PortRef",
    "Program",
    "RayExecutor",
    "RayModule",
    "function",
    "functional",
]
