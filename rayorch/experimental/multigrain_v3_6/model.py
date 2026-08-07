"""Multigrain v3.6 的纯逻辑身份与终态。

本模块刻意不依赖 Ray、编译器或运行时。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class CompileError(ValueError):
    """符号 Program 违反了 v3.6 编译期不变量。"""


class ExecutionError(RuntimeError):
    """一次 Call dispatch 无法按照编译后的执行合同继续。"""


@dataclass(frozen=True, slots=True, order=True)
class CallRef:
    """静态 Program 中一个计算调用点的紧凑身份。"""

    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("CallRef must be non-negative")


@dataclass(frozen=True, slots=True, order=True)
class PortRef:
    """静态 Program 中一个逻辑数据端口的紧凑身份。"""

    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("PortRef must be non-negative")


@dataclass(frozen=True, slots=True, order=True)
class DomainRef:
    """一个实体粒度层级的身份；不同 Domain 的整数不参与对齐。"""

    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("DomainRef must be non-negative")


@dataclass(frozen=True, slots=True, order=True)
class EntityRef:
    """一个 Domain 中的一次逻辑 occurrence。

    第一版中 ``value`` 仅在 Arena 内有效；Domain 本身属于身份的一部分，
    因此无关坐标不会仅因整数相同而意外对齐。
    """

    domain: DomainRef
    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("EntityRef.value must be non-negative")


@dataclass(frozen=True, slots=True, order=True)
class ItemRef:
    """Port 与 Entity 的交点，表示一次逻辑数据 occurrence。"""

    port: PortRef
    entity: EntityRef


@dataclass(frozen=True, slots=True, order=True)
class GrainRef:
    """Call 与执行 Domain 中 Entity 的交点，表示一次逻辑调用。"""

    call: CallRef
    entity: EntityRef


class InputMode(Enum):
    """Call 输入在上游非 PRESENT 时的传播策略。"""

    REQUIRED = auto()
    OPTIONAL = auto()


class ItemOutcome(Enum):
    """Item 的互斥终态；PRESENT 才允许关联 ValueBinding。"""

    PRESENT = auto()
    DROPPED = auto()
    FAILED = auto()
    SUPPRESSED = auto()


class GrainPhase(Enum):
    """Grain 从可调度到执行中再到封闭的生命周期阶段。"""

    READY = auto()
    IN_FLIGHT = auto()
    SEALED = auto()


class ShapeState(Enum):
    """一次 fan-out Shape 的终态及其 cardinality 可知性。"""

    SUCCEEDED = auto()
    DROPPED = auto()
    FAILED = auto()


class _Missing:
    __slots__ = ()

    def __repr__(self) -> str:
        return "MISSING"


MISSING = _Missing()


__all__ = [
    "CallRef",
    "CompileError",
    "DomainRef",
    "EntityRef",
    "ExecutionError",
    "GrainPhase",
    "GrainRef",
    "InputMode",
    "ItemOutcome",
    "ItemRef",
    "MISSING",
    "PortRef",
    "ShapeState",
]
