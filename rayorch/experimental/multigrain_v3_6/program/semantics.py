"""Port primitive 的封闭、穷尽语义表。

所有 compiler stage 都先调用 :func:`describe_origin`，而不是各自重新猜测
PortOrigin。新增 primitive 若未在这里登记，会立即落入 assert_never。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import assert_never

from .logical import (
    BroadcastOrigin,
    CallOutputOrigin,
    ExpandOrigin,
    FilterOrigin,
    ReduceOrigin,
    PortOrigin,
    SourceOrigin,
)
from ..model import CallRef, PortRef


class PrimitiveKind(Enum):
    SOURCE = auto()
    CALL_OUTPUT = auto()
    EXPAND = auto()
    REDUCE = auto()
    BROADCAST = auto()
    FILTER = auto()


class InputRole(Enum):
    EXPAND_GROUP = auto()
    REDUCE_VALUE = auto()
    REDUCE_MEMBERS = auto()
    BROADCAST_SOURCE = auto()
    FILTER_SOURCE = auto()
    FILTER_MASK = auto()


@dataclass(frozen=True, slots=True)
class PrimitiveInput:
    port: PortRef
    role: InputRole


@dataclass(frozen=True, slots=True)
class PrimitiveSemantics:
    """一个 origin 对 analysis/lowering 暴露的完整静态合同。"""

    kind: PrimitiveKind
    inputs: tuple[PrimitiveInput, ...] = ()
    control_demands: tuple[PortRef, ...] = ()
    control_predecessors: tuple[PortRef, ...] = ()
    rejects_control: bool = False
    producing_call: CallRef | None = None
    output_index: int | None = None
    source_index: int | None = None


def describe_origin(origin: PortOrigin) -> PrimitiveSemantics:
    """把每个 PortOrigin 穷尽翻译成统一语义描述。"""

    match origin:
        case SourceOrigin(source_index=source_index):
            return PrimitiveSemantics(
                PrimitiveKind.SOURCE,
                source_index=source_index,
            )
        case CallOutputOrigin(call=call, output_index=output_index):
            return PrimitiveSemantics(
                PrimitiveKind.CALL_OUTPUT,
                producing_call=call,
                output_index=output_index,
            )
        case ExpandOrigin(group_port=group):
            return PrimitiveSemantics(
                PrimitiveKind.EXPAND,
                (PrimitiveInput(group, InputRole.EXPAND_GROUP),),
                control_predecessors=(group,),
            )
        case ReduceOrigin(value_port=value, members_port=members):
            return PrimitiveSemantics(
                PrimitiveKind.REDUCE,
                (
                    PrimitiveInput(value, InputRole.REDUCE_VALUE),
                    PrimitiveInput(members, InputRole.REDUCE_MEMBERS),
                ),
                rejects_control=True,
            )
        case BroadcastOrigin(source_port=source):
            return PrimitiveSemantics(
                PrimitiveKind.BROADCAST,
                (PrimitiveInput(source, InputRole.BROADCAST_SOURCE),),
                control_predecessors=(source,),
            )
        case FilterOrigin(source_port=source, mask_port=mask):
            return PrimitiveSemantics(
                PrimitiveKind.FILTER,
                (
                    PrimitiveInput(source, InputRole.FILTER_SOURCE),
                    PrimitiveInput(mask, InputRole.FILTER_MASK),
                ),
                control_demands=(mask,),
                control_predecessors=(source,),
            )
        case _:
            assert_never(origin)


__all__ = [
    "InputRole",
    "PrimitiveInput",
    "PrimitiveKind",
    "PrimitiveSemantics",
    "describe_origin",
]
