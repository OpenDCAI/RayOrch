"""V3 不可变 General DAG schema 与编译期校验。

该模块是静态图的唯一 authority：描述 Stage、输入模式、执行配置和 direct routing
索引。它不导入 Arena、Driver、Worker 或 Ray，也不保存任何运行时状态。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from .model import PortId


class CompileError(ValueError):
    """表示用户 DAG 不满足 V3 编程模型或静态语义约束。"""


class Primitive(Enum):
    """V3 编译后支持的 Stage 原语类型。"""
    SOURCE = "source"
    MAP = "map"
    FILTER = "filter"
    EXPAND = "expand"
    REDUCE = "reduce"


class InputMode(Enum):
    """一个 input 在对齐语义和 Actor payload 中的参与方式。"""
    ONE = "one"
    OPTIONAL_ONE = "optional_one"
    GROUP = "group"
    ANCHOR = "anchor"


class RecoveryPreset(Enum):
    """对外暴露的少量恢复策略预设，由 Arena 映射为具体恢复动作。"""
    RAISE = "raise"
    RETRY_BATCH = "retry_batch"
    RETRY_TAIL = "retry_tail"
    ISOLATE_TAIL = "isolate_tail"
    FAIL_BATCH = "fail_batch"


@dataclass(frozen=True, slots=True)
class RecoveryLimits:
    """基础设施重试和 UDF recovery 的硬预算。

    这些预算属于 Stage 静态执行合同；运行中的计数位于 Arena RecoveryTask。
    """
    max_infra_retries: int = 1
    max_recovery_attempts: int = 2
    max_split_depth: int = 16
    max_extra_rpcs: int = 1024
    max_reexecuted_grains: int = 100_000

    def __post_init__(self) -> None:
        """保证所有恢复预算非负。"""

        if min(
            self.max_infra_retries,
            self.max_recovery_attempts,
            self.max_split_depth,
            self.max_extra_rpcs,
            self.max_reexecuted_grains,
        ) < 0:
            raise ValueError("recovery limits must be non-negative")


@dataclass(frozen=True, slots=True)
class RecoverySpec:
    """绑定到单个 Stage 的不可变 recovery preset 与预算。"""
    preset: RecoveryPreset = RecoveryPreset.RAISE
    limits: RecoveryLimits = RecoveryLimits()


@dataclass(frozen=True, slots=True)
class UdfSpec:
    """persistent Worker 构造用户 UDF instance 所需的 recipe。"""
    target: Any
    init_args: tuple[Any, ...] = ()
    init_kwargs: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    """Stage 的物理副本、合批、恢复和 Ray resource 配置。"""
    replicas: int = 1
    batch_size: int = 1
    max_batch_wait_ms: float = 2.0
    batch_scope: str = "elastic"
    max_outstanding_per_actor: int | None = None
    recovery: RecoverySpec = RecoverySpec()
    ray_options: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        """校验 replicas、batch trigger 和 batch_scope 配置。"""

        if self.replicas <= 0 or self.batch_size <= 0:
            raise ValueError("replicas and batch_size must be positive")
        if self.max_batch_wait_ms < 0:
            raise ValueError("max_batch_wait_ms must be non-negative")
        if self.batch_scope not in {"elastic", "parent_bound"}:
            raise ValueError("batch_scope must be elastic or parent_bound")
        if (
            self.max_outstanding_per_actor is not None
            and self.max_outstanding_per_actor <= 0
        ):
            raise ValueError("max_outstanding_per_actor must be positive")


@dataclass(frozen=True, slots=True)
class InputSpec:
    """进入 Stage 的一条命名静态边。

    `mode` 明确 scalar、optional、GROUP 或 semantic-only ANCHOR；运行时不再猜测输入
    角色。
    """
    name: str
    port: PortId
    mode: InputMode = InputMode.ONE

    def __post_init__(self) -> None:
        """校验输入名称非空。"""

        if not self.name:
            raise ValueError("input name must be non-empty")


@dataclass(frozen=True, slots=True)
class ReduceSpec:
    """ordered hierarchical Reduce 所需的最小静态合同。

    `scope_path` 是 anchor scope 到 member scope 之间唯一的 Expand Stage 序列；每个
    Expand 对应 UDF nested GROUP 中的一层 list。
    """
    members_input: int
    scope_path: tuple[int, ...]

    def __post_init__(self) -> None:
        """禁止没有 Expand 层级的 GROUP Reduce。"""

        if not self.scope_path:
            raise ValueError("Reduce scope_path must contain at least one Expand")

@dataclass(frozen=True, slots=True)
class StageSpec:
    """一个逻辑 Stage 的不可变编译结果。

    输出 Port 由 `(id, output_index)` 推导，不重复存储 PortSpec；kind-specific 合法性由
    compiler validator 保证。
    """
    id: int
    kind: Primitive
    inputs: tuple[InputSpec, ...]
    output_count: int
    driving_input: int | None
    udf: UdfSpec | None
    execution: ExecutionSpec | None
    reduce: ReduceSpec | None = None

    def output_ports(self) -> tuple[PortId, ...]:
        """按 output_count 派生该 Stage 的全部 PortId。"""

        return tuple(PortId(self.id, index) for index in range(self.output_count))


@dataclass(frozen=True, slots=True)
class ConsumerEdge:
    """从 output Port 到 consumer input slot 的可重建 direct route。"""
    stage: int
    input_index: int


@dataclass(frozen=True, slots=True)
class CompiledDAG:
    """一次 run 中由所有 Arena 只读共享的不可变 General DAG。

    `consumers_by_port` 驱动 Item receipt 路由；`reduces_by_expand` 驱动 fanout fact 只
    投递到相关 hierarchical Reduce，避免全表扫描。
    """
    stages: tuple[StageSpec, ...]
    consumers_by_port: Mapping[PortId, tuple[ConsumerEdge, ...]] = field(
        repr=False
    )
    reduces_by_expand: Mapping[int, tuple[int, ...]] = field(
        repr=False
    )
    source_ports: tuple[PortId, ...] = ()
    output_ports: tuple[PortId, ...] = ()

    def stage(self, stage_id: int) -> StageSpec:
        """按 dense StageId 返回 StageSpec，并校验索引一致性。"""

        try:
            stage = self.stages[stage_id]
        except IndexError as error:
            raise CompileError(f"unknown stage id: {stage_id}") from error
        if stage.id != stage_id:
            raise CompileError("stage ids are not dense and ordered")
        return stage

    def producer(self, port: PortId) -> StageSpec:
        """返回指定 PortId 的 producer Stage，并校验 output index。"""

        stage = self.stage(port.stage)
        if port.output >= stage.output_count:
            raise CompileError(f"unknown output port: {port}")
        return stage

    def consumers(self, port: PortId) -> tuple[ConsumerEdge, ...]:
        """返回一个 output Port 的直接 consumer edges。"""

        return self.consumers_by_port.get(port, ())


def _validate_stage_shape(stage: StageSpec) -> None:
    """校验单个 StageSpec 的 kind-specific 字段组合。"""

    if stage.id < 0 or stage.output_count <= 0:
        raise CompileError("stage id/output_count must be valid")
    names = tuple(input_spec.name for input_spec in stage.inputs)
    if len(names) != len(set(names)):
        raise CompileError(f"stage {stage.id} has duplicate input names")

    if stage.kind is Primitive.SOURCE:
        if stage.inputs or stage.output_count != 1:
            raise CompileError("Source requires no inputs and one output")
        if any(
            value is not None
            for value in (
                stage.driving_input,
                stage.udf,
                stage.execution,
                stage.reduce,
            )
        ):
            raise CompileError("Source cannot declare execution metadata")
        return

    if stage.udf is None or stage.execution is None:
        raise CompileError(f"stage {stage.id} needs UDF and execution specs")

    if stage.kind is Primitive.REDUCE:
        if stage.driving_input is not None or stage.reduce is None:
            raise CompileError("Reduce requires ReduceSpec and no driving_input")
        anchors = [
            index
            for index, spec in enumerate(stage.inputs)
            if spec.mode is InputMode.ANCHOR
        ]
        groups = [
            index
            for index, spec in enumerate(stage.inputs)
            if spec.mode is InputMode.GROUP
        ]
        if len(anchors) != 1 or not groups:
            raise CompileError("Reduce needs one ANCHOR and at least one GROUP")
        if stage.reduce.members_input not in groups:
            raise CompileError("Reduce members_input must reference a GROUP")
        return

    if stage.reduce is not None:
        raise CompileError("ReduceSpec is only valid on Reduce")
    if stage.driving_input is None or not (
        0 <= stage.driving_input < len(stage.inputs)
    ):
        raise CompileError(f"{stage.kind.value} needs a valid driving_input")
    if stage.inputs[stage.driving_input].mode is not InputMode.ONE:
        raise CompileError("driving input must be required ONE")
    if any(
        spec.mode not in {InputMode.ONE, InputMode.OPTIONAL_ONE}
        for spec in stage.inputs
    ):
        raise CompileError(f"{stage.kind.value} only accepts aligned ONE inputs")
    if stage.kind is Primitive.FILTER:
        if any(spec.mode is not InputMode.ONE for spec in stage.inputs):
            raise CompileError("Filter only accepts required ONE inputs")
        if stage.output_count != len(stage.inputs):
            raise CompileError("Filter output_count must equal input count")


def _validate_scope_and_reduce(
    stages: tuple[StageSpec, ...],
) -> dict[PortId, tuple[int, ...]]:
    """推导每个 Port 的 Expand scope，并校验 aligned/Reduce scope 关系。"""

    scopes: dict[PortId, tuple[int, ...]] = {}
    for stage in stages:
        if stage.kind is Primitive.SOURCE:
            output_scope = ()
        elif stage.kind in {
            Primitive.MAP,
            Primitive.FILTER,
            Primitive.EXPAND,
        }:
            assert stage.driving_input is not None
            parent_scope = scopes[stage.inputs[stage.driving_input].port]
            for spec in stage.inputs:
                if scopes[spec.port] != parent_scope:
                    raise CompileError(
                        f"stage {stage.id} aligned inputs have different scopes"
                    )
            output_scope = (
                (*parent_scope, stage.id)
                if stage.kind is Primitive.EXPAND
                else parent_scope
            )
        else:
            assert stage.reduce is not None
            anchor = next(
                spec for spec in stage.inputs if spec.mode is InputMode.ANCHOR
            )
            anchor_scope = scopes[anchor.port]
            for spec in stage.inputs:
                input_scope = scopes[spec.port]
                if spec.mode is InputMode.GROUP:
                    expected_scope = (*anchor_scope, *stage.reduce.scope_path)
                    if input_scope != expected_scope:
                        raise CompileError(
                            f"Reduce {stage.id} GROUP has wrong scope path"
                        )
                elif spec.mode in {InputMode.ONE, InputMode.OPTIONAL_ONE}:
                    if input_scope != anchor_scope:
                        raise CompileError(
                            f"Reduce {stage.id} scalar context is not anchor-aligned"
                        )
            for expand_id in stage.reduce.scope_path:
                if stages[expand_id].kind is not Primitive.EXPAND:
                    raise CompileError("Reduce scope_path must reference Expands")
            origin = stages[stage.reduce.scope_path[0]]
            assert origin.driving_input is not None
            origin_parent = origin.inputs[origin.driving_input].port
            if anchor.port != origin_parent:
                raise CompileError(
                    "Reduce anchor must be the exact origin Expand input port"
                )
            output_scope = anchor_scope
        for port in stage.output_ports():
            scopes[port] = output_scope
    return scopes


def compile_dag(
    stages: tuple[StageSpec, ...],
    *,
    source_ports: tuple[PortId, ...],
    output_ports: tuple[PortId, ...],
) -> CompiledDAG:
    """校验拓扑并构建 immutable CompiledDAG 与 direct routing indexes。"""

    if tuple(stage.id for stage in stages) != tuple(range(len(stages))):
        raise CompileError("stage ids must be dense and topologically ordered")

    producer_ports: set[PortId] = set()
    consumers: dict[PortId, list[ConsumerEdge]] = {}
    reduces_by_expand: dict[int, list[int]] = {}
    for stage in stages:
        _validate_stage_shape(stage)
        for port in stage.output_ports():
            producer_ports.add(port)
        for index, input_spec in enumerate(stage.inputs):
            if input_spec.port not in producer_ports:
                raise CompileError(
                    f"stage {stage.id} input is not produced earlier: "
                    f"{input_spec.port}"
                )
            consumers.setdefault(input_spec.port, []).append(
                ConsumerEdge(stage.id, index)
            )
        if stage.kind is Primitive.REDUCE:
            assert stage.reduce is not None
            for expand_stage in stage.reduce.scope_path:
                reduces_by_expand.setdefault(expand_stage, []).append(stage.id)

    _validate_scope_and_reduce(stages)
    for port in (*source_ports, *output_ports):
        if port not in producer_ports:
            raise CompileError(f"unknown source/output port: {port}")
    if any(stages[port.stage].kind is not Primitive.SOURCE for port in source_ports):
        raise CompileError("source_ports must reference Source stages")

    return CompiledDAG(
        stages=stages,
        consumers_by_port=MappingProxyType(
            {port: tuple(edges) for port, edges in consumers.items()}
        ),
        reduces_by_expand=MappingProxyType(
            {
                stage: tuple(reduces)
                for stage, reduces in reduces_by_expand.items()
            }
        ),
        source_ports=source_ports,
        output_ports=output_ports,
    )
