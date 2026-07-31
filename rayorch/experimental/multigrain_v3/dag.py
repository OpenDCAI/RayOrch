"""Immutable General-DAG schema and compile-time validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from .model import PortId


class CompileError(ValueError):
    """The user graph violates the V3 programming model."""


class Primitive(Enum):
    """Compiled Stage kinds supported by the V3 prototype."""
    SOURCE = "source"
    MAP = "map"
    FILTER = "filter"
    EXPAND = "expand"
    REDUCE = "reduce"


class InputMode(Enum):
    """How one compiled input participates in alignment and actor payloads."""
    ONE = "one"
    OPTIONAL_ONE = "optional_one"
    GROUP = "group"
    ANCHOR = "anchor"


class RecoveryPreset(Enum):
    """Small public presets mapped to Arena-local recovery actions."""
    RAISE = "raise"
    RETRY_BATCH = "retry_batch"
    RETRY_TAIL = "retry_tail"
    ISOLATE_TAIL = "isolate_tail"
    FAIL_BATCH = "fail_batch"


@dataclass(frozen=True, slots=True)
class RecoveryLimits:
    """Hard budgets for infrastructure retry and UDF recovery."""
    max_infra_retries: int = 1
    max_recovery_attempts: int = 2
    max_split_depth: int = 16
    max_extra_rpcs: int = 1024
    max_reexecuted_grains: int = 100_000

    def __post_init__(self) -> None:
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
    """Immutable recovery policy attached to one Stage."""
    preset: RecoveryPreset = RecoveryPreset.RAISE
    limits: RecoveryLimits = RecoveryLimits()


@dataclass(frozen=True, slots=True)
class UdfSpec:
    """How a persistent worker constructs the user UDF instance."""
    target: Any
    init_args: tuple[Any, ...] = ()
    init_kwargs: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    """Physical replicas, batching, recovery, and Ray resource options."""
    replicas: int = 1
    batch_size: int = 1
    max_batch_wait_ms: float = 2.0
    batch_scope: str = "elastic"
    recovery: RecoverySpec = RecoverySpec()
    ray_options: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.replicas <= 0 or self.batch_size <= 0:
            raise ValueError("replicas and batch_size must be positive")
        if self.max_batch_wait_ms < 0:
            raise ValueError("max_batch_wait_ms must be non-negative")
        if self.batch_scope not in {"elastic", "parent_bound"}:
            raise ValueError("batch_scope must be elastic or parent_bound")


@dataclass(frozen=True, slots=True)
class InputSpec:
    """One named static edge into a Stage."""
    name: str
    port: PortId
    mode: InputMode = InputMode.ONE

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("input name must be non-empty")


@dataclass(frozen=True, slots=True)
class ReduceSpec:
    """The minimal extra contract needed for ordered Expand-scoped Reduce."""
    members_input: int
    origin_expand: int


@dataclass(frozen=True, slots=True)
class StageSpec:
    """Immutable compiled description of one logical Stage."""
    id: int
    kind: Primitive
    inputs: tuple[InputSpec, ...]
    output_count: int
    driving_input: int | None
    udf: UdfSpec | None
    execution: ExecutionSpec | None
    reduce: ReduceSpec | None = None

    def output_ports(self) -> tuple[PortId, ...]:
        return tuple(PortId(self.id, index) for index in range(self.output_count))


@dataclass(frozen=True, slots=True)
class ConsumerEdge:
    """Derived route from an output Port to one consumer input slot."""
    stage: int
    input_index: int


@dataclass(frozen=True, slots=True)
class CompiledDAG:
    """Immutable General DAG shared read-only by every Arena in a run."""
    stages: tuple[StageSpec, ...]
    consumers_by_port: Mapping[PortId, tuple[ConsumerEdge, ...]] = field(
        repr=False
    )
    source_ports: tuple[PortId, ...] = ()
    output_ports: tuple[PortId, ...] = ()

    def stage(self, stage_id: int) -> StageSpec:
        try:
            stage = self.stages[stage_id]
        except IndexError as error:
            raise CompileError(f"unknown stage id: {stage_id}") from error
        if stage.id != stage_id:
            raise CompileError("stage ids are not dense and ordered")
        return stage

    def producer(self, port: PortId) -> StageSpec:
        stage = self.stage(port.stage)
        if port.output >= stage.output_count:
            raise CompileError(f"unknown output port: {port}")
        return stage

    def consumers(self, port: PortId) -> tuple[ConsumerEdge, ...]:
        return self.consumers_by_port.get(port, ())


def _validate_stage_shape(stage: StageSpec) -> None:
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
                    if not input_scope or input_scope[-1] != stage.reduce.origin_expand:
                        raise CompileError(
                            f"Reduce {stage.id} GROUP has wrong origin scope"
                        )
                    if input_scope[:-1] != anchor_scope:
                        raise CompileError(
                            f"Reduce {stage.id} anchor does not close GROUP scope"
                        )
                elif spec.mode in {InputMode.ONE, InputMode.OPTIONAL_ONE}:
                    if input_scope != anchor_scope:
                        raise CompileError(
                            f"Reduce {stage.id} scalar context is not anchor-aligned"
                        )
            origin = stages[stage.reduce.origin_expand]
            if origin.kind is not Primitive.EXPAND:
                raise CompileError("Reduce origin_expand must reference Expand")
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
    if tuple(stage.id for stage in stages) != tuple(range(len(stages))):
        raise CompileError("stage ids must be dense and topologically ordered")

    producer_ports: set[PortId] = set()
    consumers: dict[PortId, list[ConsumerEdge]] = {}
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
        source_ports=source_ports,
        output_ports=output_ports,
    )
