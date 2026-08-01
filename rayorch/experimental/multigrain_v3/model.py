"""Multigrain V3 的语义身份、血缘记录与 Grain 生命周期。

本模块只定义与 correctness 有关的稳定数据结构，不包含 Actor、ObjectRef、物理 batch
或调度策略。大多数类型是不可变 dataclass，可安全作为字典 key、跨组件 DTO 字段和
确定性 identity 编码输入。
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeAlias


DIGEST_BYTES = 16
RUN_SALT_BYTES = 16
PERSONALIZATION = b"RayOrchMGV3"


class InvariantError(RuntimeError):
    """表示框架内部语义或生命周期不变量被破坏。"""


@dataclass(frozen=True, slots=True)
class PortId:
    """静态 DAG 中一个 Stage output 的轻量坐标。

    `stage` 指向 producer Stage，`output` 指向该 Stage 的输出序号。PortId 不携带
    consumer、UDF 或运行时状态；这些信息由 CompiledDAG 唯一解释。
    """

    stage: int
    output: int

    def __post_init__(self) -> None:
        """拒绝负数坐标，保证 PortId 可直接用于数组索引和 identity 编码。"""

        if self.stage < 0 or self.output < 0:
            raise ValueError("PortId fields must be non-negative")


@dataclass(frozen=True, slots=True)
class EntityId:
    """固定宽度的 logical occurrence 身份。

    同一个 Entity 可以在不同 Port 上产生多个 ItemRef。EntityId 表达“这是哪个逻辑
    对象”，不表达它当前位于哪个 DAG Port，也不表达物理数据位置。
    """

    raw: bytes

    def __post_init__(self) -> None:
        """校验 EntityId 始终使用固定 128-bit 表示。"""

        if len(self.raw) != DIGEST_BYTES:
            raise ValueError("EntityId must contain exactly 16 bytes")

    def hex(self) -> str:
        """返回适合日志和诊断展示的十六进制身份。"""

        return self.raw.hex()


@dataclass(frozen=True, slots=True)
class ItemRef:
    """逐 Port 的逻辑 value 坐标。

    ItemRef 等于 `PortId + EntityId`，是 ItemTable 和 ValueTable 的共同 key。它只表示
    logical value，不携带 terminal 状态、producer 或 BlockRow。
    """

    port: PortId
    entity: EntityId


@dataclass(frozen=True, slots=True)
class GrainId:
    """固定宽度的 Logical Grain 身份。

    GrainId 由 Stage 和按编译顺序排列的 InputBinding 确定，与 Actor、DispatchId、
    generation、batch packing 和 completion order 无关。
    """

    raw: bytes

    def __post_init__(self) -> None:
        """校验 GrainId 始终使用固定 128-bit 表示。"""

        if len(self.raw) != DIGEST_BYTES:
            raise ValueError("GrainId must contain exactly 16 bytes")

    def hex(self) -> str:
        """返回适合日志和错误报告展示的十六进制身份。"""

        return self.raw.hex()


@dataclass(frozen=True, slots=True)
class GroupShape:
    """使用逐层 CSR offsets 表示 canonical nested-list 形状。

    第一层 offsets 把唯一 Reduce anchor 映射到第一层 child；后续每层把上一层 node
    映射到下一层 node。最后一个 offset 等于 flat leaves 数量。

    设计理念：

    - 不创建大量递归 Python tree node；
    - 保留 intermediate `N=0` 对应的空 list；
    - 让 flat ItemRef/RowTake 与 tree shape 分离；
    - 使不同 tree shape 参与 GrainId 编码，避免 identity 冲突。
    """

    offsets_by_level: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        """校验各层 CSR offsets 连续、单调且层间 node 数一致。"""

        if not self.offsets_by_level:
            raise ValueError("GroupShape requires at least one Expand level")
        expected_parents = 1
        for offsets in self.offsets_by_level:
            if len(offsets) != expected_parents + 1:
                raise ValueError("GroupShape offset arity is inconsistent")
            if not offsets or offsets[0] != 0:
                raise ValueError("GroupShape offsets must start at zero")
            if any(left > right for left, right in zip(offsets, offsets[1:])):
                raise ValueError("GroupShape offsets must be non-decreasing")
            expected_parents = offsets[-1]

    @property
    def depth(self) -> int:
        """返回 nested list 深度，也就是跨越的 Expand 层数。"""

        return len(self.offsets_by_level)

    @property
    def leaf_count(self) -> int:
        """返回 shape 对应的 flat leaf 数量。"""

        return self.offsets_by_level[-1][-1]


@dataclass(frozen=True, slots=True)
class InputBinding:
    """一个编译期 input name 对应的有序逻辑绑定。

    scalar/anchor 输入通常绑定一个 ItemRef；GROUP 输入绑定 flat ordered leaves，并通过
    `group_shape` 保存 nested 边界。InputBinding 是 GrainId 的语义输入，不保存 payload
    或 BlockRow。
    """

    name: str
    items: tuple[ItemRef, ...]
    group_shape: GroupShape | None = None

    def __post_init__(self) -> None:
        """校验名称、tuple 表示以及 GROUP shape 与 leaf 数量一致。"""

        if not self.name:
            raise ValueError("input binding name must be non-empty")
        if not isinstance(self.items, tuple):
            raise TypeError("InputBinding.items must be a tuple")
        if self.group_shape is not None and (
            self.group_shape.leaf_count != len(self.items)
        ):
            raise ValueError("GroupShape leaf count does not match items")


@dataclass(frozen=True, slots=True)
class AttemptToken:
    """一次 Logical Grain 物理执行尝试的 generation fence。

    Token 用于拒绝 retry 之前的迟到 completion。它属于短命执行状态，不参与 GrainId。
    """
    arena: int
    dispatch: int
    grain: GrainId
    generation: int


@dataclass(frozen=True, slots=True)
class Emission:
    """Grain 在一个 output Port 上产生的有序 logical output。"""
    item: ItemRef
    ordinal: int

    def __post_init__(self) -> None:
        """保证 ordinal 非负，便于 Expand/Reduce 稳定恢复顺序。"""

        if self.ordinal < 0:
            raise ValueError("emission ordinal must be non-negative")


@dataclass(frozen=True, slots=True)
class Success:
    """Grain 成功终态；每个 output Port 对应一条有序 emission 序列。"""
    emissions_by_port: tuple[tuple[Emission, ...], ...]


LineageCause: TypeAlias = ItemRef | GrainId


@dataclass(frozen=True, slots=True)
class GrainFailure:
    """一个 Logical Grain 的用户可见失败归因。

    `direct_causes` 只保存直接血缘原因，不构造通用 recovery graph。
    """
    kind: str
    message: str
    direct_causes: tuple[LineageCause, ...] = ()


@dataclass(frozen=True, slots=True)
class Failed:
    """Grain 已执行但失败的终态。"""
    failure: GrainFailure


@dataclass(frozen=True, slots=True)
class Suppressed:
    """Grain 因 required dependency 无法执行的终态。"""
    direct_causes: tuple[LineageCause, ...]


GrainOutcome: TypeAlias = Success | Failed | Suppressed


class GrainPhase(Enum):
    """Logical Grain 唯一允许的执行生命周期。"""
    READY = "ready"
    IN_FLIGHT = "in_flight"
    SEALED = "sealed"


@dataclass(slots=True)
class GrainRecord:
    """Logical Grain 的权威语义记录与短命 attempt 状态。

    语义字段包括 Stage、inputs、output Ports 和 outcome；短命字段只包括 phase、
    generation、active attempt 和 infra failure 计数。Actor、ObjectRef、物理 batch、
    recovery tree 等信息必须保存在 Arena/StageExecutor，而不能进入 GrainRecord。
    """

    id: GrainId
    stage: int
    inputs: tuple[InputBinding, ...]
    output_ports: tuple[PortId, ...]
    outcome: GrainOutcome | None = None
    phase: GrainPhase = GrainPhase.READY
    generation: int = 0
    active_attempt: AttemptToken | None = None
    infra_failures: int = 0

    def __post_init__(self) -> None:
        """构造后立即验证 phase、outcome 和 active attempt 的组合是否合法。"""

        self.validate()

    @classmethod
    def sealed(
        cls,
        *,
        id: GrainId,
        stage: int,
        inputs: tuple[InputBinding, ...],
        output_ports: tuple[PortId, ...],
        outcome: GrainOutcome,
    ) -> "GrainRecord":
        """构造一个无需 RPC 的 terminal Grain，例如 Source 或 Suppressed Grain。"""

        return cls(
            id=id,
            stage=stage,
            inputs=inputs,
            output_ports=output_ports,
            outcome=outcome,
            phase=GrainPhase.SEALED,
        )

    def validate(self) -> None:
        """检查 Grain 生命周期和 attempt generation 的内部一致性。"""

        legal = (
            self.phase is GrainPhase.READY
            and self.outcome is None
            and self.active_attempt is None
        ) or (
            self.phase is GrainPhase.IN_FLIGHT
            and self.outcome is None
            and self.active_attempt is not None
        ) or (
            self.phase is GrainPhase.SEALED
            and self.outcome is not None
            and self.active_attempt is None
        )
        if not legal:
            raise InvariantError("illegal GrainRecord lifecycle state")
        if self.generation < 0 or self.infra_failures < 0:
            raise InvariantError("grain counters must be non-negative")
        if self.active_attempt is not None:
            if self.active_attempt.grain != self.id:
                raise InvariantError("active attempt targets another grain")
            if self.active_attempt.generation != self.generation:
                raise InvariantError("active attempt generation mismatch")

    def reserve(self, arena: int, dispatch: int) -> AttemptToken:
        """将 READY Grain 预留给一个 dispatch，并生成新 generation token。"""

        if self.phase is not GrainPhase.READY:
            raise InvariantError("only READY grains can be reserved")
        self.generation += 1
        token = AttemptToken(arena, dispatch, self.id, self.generation)
        self.active_attempt = token
        self.phase = GrainPhase.IN_FLIGHT
        self.validate()
        return token

    def release(self, token: AttemptToken, *, infrastructure: bool = False) -> bool:
        """释放当前 attempt，使 Grain 回到 READY。

        Token 已过期时返回 False；基础设施失败重试会额外增加 infra failure 计数。
        """

        if self.active_attempt != token:
            return False
        self.active_attempt = None
        self.phase = GrainPhase.READY
        if infrastructure:
            self.infra_failures += 1
        self.validate()
        return True

    def seal(self, outcome: GrainOutcome, token: AttemptToken | None = None) -> bool:
        """以给定 outcome 原子封口 Grain。

        IN_FLIGHT Grain 只接受当前 active token；迟到 token 返回 False，不覆盖新结果。
        """

        if self.phase is GrainPhase.SEALED:
            if self.outcome != outcome:
                raise InvariantError("grain already sealed with another outcome")
            return False
        if self.phase is GrainPhase.IN_FLIGHT and self.active_attempt != token:
            return False
        if self.phase is GrainPhase.READY and token is not None:
            raise InvariantError("READY grain cannot accept an attempt token")
        self.outcome = outcome
        self.phase = GrainPhase.SEALED
        self.active_attempt = None
        self.validate()
        return True


class ItemTerminal(Enum):
    """ItemRef 在某个 Port 上的终态分类。"""
    PRESENT = "present"
    DROPPED = "dropped"
    FAILED = "failed"
    SUPPRESSED = "suppressed"


@dataclass(frozen=True, slots=True)
class ItemRecord:
    """逐 Port terminal lineage 记录。

    `producer` 和 `cause` 描述语义来源；PRESENT payload 的物理位置单独保存在
    ArenaEngine.values，避免 ItemRecord 与 ObjectRef ownership 耦合。
    """

    ref: ItemRef
    producer: GrainId | None
    terminal: ItemTerminal
    cause: GrainId | None = None

    def __post_init__(self) -> None:
        """校验 PRESENT/FAILED/SUPPRESSED 必须具有明确 producer。"""

        if self.terminal is ItemTerminal.PRESENT and self.producer is None:
            raise ValueError("PRESENT item must have a producer")
        if self.terminal in {ItemTerminal.FAILED, ItemTerminal.SUPPRESSED}:
            if self.producer is None:
                raise ValueError("failed/suppressed item must have a producer")


@dataclass(frozen=True, slots=True)
class EntityOrigin:
    """Expand 为 child Entity 引入的 parent/ordinal ancestry。

    Parent link 形成隐式 scope stack，使 Reduce 可以沿 lineage 恢复任意深度 ordinal path。
    """
    parent_entity: EntityId
    expand_stage: int
    expand_grain: GrainId
    ordinal: int

    def __post_init__(self) -> None:
        """校验 child ordinal 非负。"""

        if self.ordinal < 0:
            raise ValueError("entity ordinal must be non-negative")


@dataclass(frozen=True, slots=True)
class BlockRow:
    """Arena 内 coarse block 的物理行坐标。

    BlockRow 只在 ValueTable/协议构造中使用，不参与 ItemRef 或 GrainId。
    """
    block: int
    row: int

    def __post_init__(self) -> None:
        """校验 block id 和 row 均为非负整数。"""

        if self.block < 0 or self.row < 0:
            raise ValueError("BlockRow fields must be non-negative")


def _u64(value: int) -> bytes:
    """把非负长度或计数编码为固定 8-byte big-endian。"""

    if value < 0:
        raise ValueError("length/count cannot be negative")
    return struct.pack(">Q", value)


def canonical_encode(value: Any) -> bytes:
    """对封闭的 identity 类型集合进行确定性、type-sensitive 编码。"""

    if value is None:
        return b"n"
    if isinstance(value, bool):
        return b"b" + (b"\x01" if value else b"\x00")
    if isinstance(value, int):
        magnitude = abs(value)
        raw = (
            b""
            if magnitude == 0
            else magnitude.to_bytes((magnitude.bit_length() + 7) // 8, "big")
        )
        return b"i" + (b"\x01" if value < 0 else b"\x00") + _u64(len(raw)) + raw
    if isinstance(value, bytes):
        return b"y" + _u64(len(value)) + value
    if isinstance(value, str):
        raw = value.encode("utf-8")
        return b"s" + _u64(len(raw)) + raw
    if isinstance(value, tuple):
        return b"t" + _u64(len(value)) + b"".join(
            canonical_encode(part) for part in value
        )
    if isinstance(value, PortId):
        return b"p" + canonical_encode(value.stage) + canonical_encode(value.output)
    if isinstance(value, EntityId):
        return b"e" + value.raw
    if isinstance(value, ItemRef):
        return b"r" + canonical_encode(value.port) + canonical_encode(value.entity)
    if isinstance(value, GrainId):
        return b"g" + value.raw
    if isinstance(value, InputBinding):
        return (
            b"o"
            + canonical_encode(value.name)
            + canonical_encode(value.items)
            + canonical_encode(value.group_shape)
        )
    if isinstance(value, GroupShape):
        return b"h" + canonical_encode(value.offsets_by_level)
    raise TypeError(f"unsupported canonical identity type: {type(value)!r}")


def _check_run_salt(run_salt: bytes) -> None:
    """校验 run_salt 类型和固定宽度。"""

    if type(run_salt) is not bytes or len(run_salt) != RUN_SALT_BYTES:
        raise ValueError("run_salt must be exactly 16 bytes")


def semantic_hash(domain: str, run_salt: bytes, *parts: Any) -> bytes:
    """使用 domain separation 计算固定宽度语义 hash。"""

    _check_run_salt(run_salt)
    return hashlib.blake2b(
        canonical_encode((domain, run_salt, *parts)),
        digest_size=DIGEST_BYTES,
        person=PERSONALIZATION,
    ).digest()


def source_entity(run_salt: bytes, source_position: int) -> EntityId:
    """按 run-global source position 派生跨 source 对齐的 EntityId。"""

    return EntityId(semantic_hash("source-entity", run_salt, source_position))


def source_grain_id(
    run_salt: bytes,
    source_port: PortId,
    source_position: int,
) -> GrainId:
    """按 source Port 和 run-global position 派生 Source GrainId。"""

    return GrainId(
        semantic_hash("source-grain", run_salt, source_port, source_position)
    )


def stage_grain_id(
    run_salt: bytes,
    stage: int,
    inputs: tuple[InputBinding, ...],
) -> GrainId:
    """按 Stage 和 canonical InputBinding 派生普通/Reduce GrainId。"""

    return GrainId(semantic_hash("stage-grain", run_salt, stage, inputs))


def expand_entity(
    run_salt: bytes,
    expand_stage: int,
    parent_entity: EntityId,
    ordinal: int,
) -> EntityId:
    """按 parent Entity、Expand Stage 和 ordinal 派生稳定 child EntityId。"""

    return EntityId(
        semantic_hash(
            "expand-entity",
            run_salt,
            expand_stage,
            parent_entity,
            ordinal,
        )
    )
