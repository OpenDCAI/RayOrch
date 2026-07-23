"""Compiled graph schema, validation, and Phase 1 semantic planners."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import product
from types import MappingProxyType
from typing import Any, Mapping

from .api import CompileError
from .grain import (
    Emission,
    ExpandOrigin,
    ExpandOriginIndex,
    Failed,
    FiberBarrier,
    FiberId,
    FiberState,
    GrainFailure,
    GrainId,
    GrainInvariantError,
    GrainRecord,
    GrainTable,
    ItemRef,
    PortId,
    RoleItems,
    Success,
    Suppressed,
    expand_entity,
    expand_grain_id,
    filter_grain_id,
    map_grain_id,
    reduce_grain_id,
    relate_entity,
    relate_grain_id,
    source_entity,
    source_grain_id,
)


class Primitive(Enum):
    SOURCE = "source"
    MAP = "map"
    FILTER = "filter"
    EXPAND = "expand"
    REDUCE = "reduce"
    RELATE = "relate"


@dataclass(frozen=True, slots=True)
class UdfRecipe:
    target: Any
    init_args: tuple[Any, ...] = ()
    init_kwargs: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class ExecutionOptions:
    replicas: int = 1
    batch_size: int = 1
    max_batch_wait_ms: float = 2.0
    batch_scope: str = "elastic"
    error_policy: str = "raise"
    max_retries: int = 0
    options: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.replicas <= 0 or self.batch_size <= 0:
            raise ValueError("replicas and batch_size must be positive")
        if self.max_batch_wait_ms < 0:
            raise ValueError("max_batch_wait_ms must be non-negative")
        if self.batch_scope not in {"elastic", "parent_bound"}:
            raise ValueError("batch_scope must be 'elastic' or 'parent_bound'")
        if self.error_policy not in {"raise", "isolate"}:
            raise ValueError("error_policy must be 'raise' or 'isolate'")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")


@dataclass(frozen=True, slots=True)
class KeyProjection:
    role: str
    by: Any


@dataclass(frozen=True, slots=True)
class InputBinding:
    role: str
    port: PortId


@dataclass(frozen=True, slots=True)
class NodeSpec:
    id: int
    kind: Primitive
    inputs: tuple[InputBinding, ...]
    output_ports: tuple[PortId, ...]
    udf_recipe: UdfRecipe | None
    execution: ExecutionOptions | None
    reduce_anchor: PortId | None = None
    reduce_members: PortId | None = None
    relate_keys: tuple[KeyProjection, ...] = ()


@dataclass(frozen=True, slots=True)
class CompiledGraph:
    nodes: tuple[NodeSpec, ...]
    _producer_by_port: Mapping[PortId, NodeSpec] = field(
        repr=False,
        compare=False,
    )

    def producer(self, port: PortId) -> NodeSpec:
        try:
            return self._producer_by_port[port]
        except KeyError as error:
            raise CompileError(f"port has no compiled producer: {port}") from error

    def node(self, node_id: int) -> NodeSpec:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise CompileError(f"unknown node id: {node_id}")


def _binding(node: NodeSpec, role: str) -> InputBinding:
    matches = [binding for binding in node.inputs if binding.role == role]
    if len(matches) != 1:
        raise CompileError(
            f"{node.kind.value} node {node.id} needs exactly one {role!r} role"
        )
    return matches[0]


def _validate_node_shape(node: NodeSpec) -> None:
    if node.id < 0:
        raise CompileError("node ids must be non-negative")
    roles = [binding.role for binding in node.inputs]
    if len(roles) != len(set(roles)) or any(not role for role in roles):
        raise CompileError(f"node {node.id} has duplicate or empty input roles")
    expected_ports = tuple(PortId(node.id, slot) for slot in range(len(node.output_ports)))
    if node.output_ports != expected_ports:
        raise CompileError(
            f"node {node.id} output ports must be contiguous slots starting at zero"
        )

    if node.kind is Primitive.SOURCE:
        if node.inputs or len(node.output_ports) != 1:
            raise CompileError("internal Source needs no inputs and one output port")
        if node.udf_recipe is not None or node.execution is not None:
            raise CompileError("internal Source cannot have UDF/execution recipes")
        if (
            node.reduce_anchor is not None
            or node.reduce_members is not None
            or node.relate_keys
        ):
            raise CompileError("internal Source has no Reduce/Relate metadata")
        return

    if node.udf_recipe is None or node.execution is None:
        raise CompileError(f"node {node.id} needs UDF and execution recipes")
    if not node.output_ports:
        raise CompileError(f"node {node.id} must declare at least one output port")

    if node.kind is Primitive.MAP:
        _binding(node, "primary")
    elif node.kind is Primitive.FILTER:
        _binding(node, "target")
    elif node.kind is Primitive.EXPAND:
        _binding(node, "parent")
    elif node.kind is Primitive.REDUCE:
        _binding(node, "anchor")
        _binding(node, "members")
        if node.reduce_anchor is None or node.reduce_members is None:
            raise CompileError("Reduce must declare anchor and members ports")
        if _binding(node, "anchor").port != node.reduce_anchor:
            raise CompileError("Reduce anchor metadata does not match its binding")
        if _binding(node, "members").port != node.reduce_members:
            raise CompileError("Reduce members metadata does not match its binding")
    elif node.kind is Primitive.RELATE:
        if len(node.inputs) < 2:
            raise CompileError("Relate requires at least two roles")
        key_roles = tuple(key.role for key in node.relate_keys)
        if key_roles != tuple(roles):
            raise CompileError("Relate key projections must follow input role order")

    if node.kind is not Primitive.REDUCE and (
        node.reduce_anchor is not None or node.reduce_members is not None
    ):
        raise CompileError("Reduce metadata is only legal on Reduce nodes")
    if node.kind is not Primitive.RELATE and node.relate_keys:
        raise CompileError("Relate keys are only legal on Relate nodes")


def _trace_reduce_origin(
    reduce_node: NodeSpec,
    producer_by_port: Mapping[PortId, NodeSpec],
) -> None:
    assert reduce_node.reduce_anchor is not None
    assert reduce_node.reduce_members is not None
    current = reduce_node.reduce_members
    visited: set[PortId] = set()
    while True:
        if current in visited:
            raise CompileError("cycle while tracing Reduce members path")
        visited.add(current)
        producer = producer_by_port.get(current)
        if producer is None:
            raise CompileError("Reduce members port has no producer")
        if producer.kind is Primitive.MAP:
            current = _binding(producer, "primary").port
            continue
        if producer.kind is Primitive.FILTER:
            current = _binding(producer, "target").port
            continue
        if producer.kind is not Primitive.EXPAND:
            raise CompileError(
                "Reduce members path must contain exactly one origin Expand "
                "followed only by Map/Filter"
            )
        parent_port = _binding(producer, "parent").port
        if parent_port != reduce_node.reduce_anchor:
            raise CompileError(
                "Reduce anchor must be the exact input port of its origin Expand"
            )
        return


def compile_graph(nodes: tuple[NodeSpec, ...]) -> CompiledGraph:
    node_ids: set[int] = set()
    producer_by_port: dict[PortId, NodeSpec] = {}
    node_order: dict[int, int] = {}

    for order, node in enumerate(nodes):
        _validate_node_shape(node)
        if node.id in node_ids:
            raise CompileError(f"duplicate node id: {node.id}")
        node_ids.add(node.id)
        node_order[node.id] = order
        for port in node.output_ports:
            if port in producer_by_port:
                raise CompileError(f"duplicate output port: {port}")
            producer_by_port[port] = node

    for node in nodes:
        for binding in node.inputs:
            producer = producer_by_port.get(binding.port)
            if producer is None:
                raise CompileError(
                    f"input port {binding.port} has no compiled producer"
                )
            if node_order[producer.id] >= node_order[node.id]:
                raise CompileError("compiled graph is not topologically ordered")
        if node.kind is Primitive.REDUCE:
            _trace_reduce_origin(node, producer_by_port)

    return CompiledGraph(
        nodes=nodes,
        _producer_by_port=MappingProxyType(dict(producer_by_port)),
    )


class ReceiptState(Enum):
    PENDING = "pending"
    PRESENT = "present"
    NORMAL_ABSENCE = "normal_absence"
    FAILED = "failed"
    SUPPRESSED = "suppressed"


@dataclass(frozen=True, slots=True)
class BindingReceipt:
    role: str
    state: ReceiptState
    item: ItemRef | None = None
    cause: GrainId | None = None

    @classmethod
    def pending(
        cls,
        role: str,
        item: ItemRef | None = None,
    ) -> "BindingReceipt":
        return cls(role, ReceiptState.PENDING, item=item)

    @classmethod
    def present(cls, role: str, item: ItemRef) -> "BindingReceipt":
        return cls(role, ReceiptState.PRESENT, item=item)

    @classmethod
    def absent(cls, role: str, item: ItemRef) -> "BindingReceipt":
        return cls(role, ReceiptState.NORMAL_ABSENCE, item=item)

    @classmethod
    def failed(
        cls,
        role: str,
        item: ItemRef,
        cause: GrainId,
    ) -> "BindingReceipt":
        return cls(role, ReceiptState.FAILED, item=item, cause=cause)

    @classmethod
    def suppressed(
        cls,
        role: str,
        item: ItemRef,
        cause: GrainId,
    ) -> "BindingReceipt":
        return cls(role, ReceiptState.SUPPRESSED, item=item, cause=cause)


class PlanAction(Enum):
    WAIT = "wait"
    ENSURE_EXECUTABLE = "ensure_executable"
    ENSURE_SUPPRESSED = "ensure_suppressed"
    NORMAL_ABSENCE = "normal_absence"


@dataclass(frozen=True, slots=True)
class PlanDecision:
    action: PlanAction
    grain: GrainRecord | None = None


class PlannerContractError(RuntimeError):
    """A planner found a run-control contract or invariant violation."""


def _validate_receipts(
    node: NodeSpec,
    receipts: tuple[BindingReceipt, ...],
) -> None:
    if tuple(receipt.role for receipt in receipts) != tuple(
        binding.role for binding in node.inputs
    ):
        raise PlannerContractError("receipts must follow compiled role order")
    for receipt in receipts:
        if receipt.state is ReceiptState.PENDING:
            if receipt.cause is not None:
                raise PlannerContractError("pending receipt cannot have a cause")
        elif receipt.state in {
            ReceiptState.PRESENT,
            ReceiptState.NORMAL_ABSENCE,
        }:
            if receipt.item is None or receipt.cause is not None:
                raise PlannerContractError("settled normal receipt has invalid fields")
        elif receipt.item is None or receipt.cause is None:
            raise PlannerContractError("failure receipt needs item and cause")


def _grain_id_for(
    kind: Primitive,
    run_salt: bytes,
    node: int,
    inputs: tuple[RoleItems, ...],
) -> GrainId:
    if kind is Primitive.MAP:
        return map_grain_id(run_salt, node, inputs)
    if kind is Primitive.FILTER:
        return filter_grain_id(run_salt, node, inputs)
    if kind is Primitive.EXPAND:
        return expand_grain_id(run_salt, node, inputs)
    raise PlannerContractError(f"unsupported unary planner kind: {kind}")


def _plan_unary(
    node: NodeSpec,
    run_salt: bytes,
    receipts: tuple[BindingReceipt, ...],
    driving_role: str,
) -> PlanDecision:
    _validate_receipts(node, receipts)
    by_role = {receipt.role: receipt for receipt in receipts}
    driving = by_role[driving_role]

    if driving.state is ReceiptState.NORMAL_ABSENCE:
        return PlanDecision(PlanAction.NORMAL_ABSENCE)
    if driving.state is ReceiptState.PENDING:
        return PlanDecision(PlanAction.WAIT)
    if any(receipt.state is ReceiptState.PENDING for receipt in receipts):
        return PlanDecision(PlanAction.WAIT)

    for receipt in receipts:
        if (
            receipt.role != driving_role
            and receipt.state is ReceiptState.NORMAL_ABSENCE
        ):
            raise PlannerContractError(
                f"required aligned role {receipt.role!r} is normally absent"
            )

    driving_item = driving.item
    assert driving_item is not None
    for receipt in receipts:
        if receipt.item is not None and receipt.item.entity != driving_item.entity:
            raise PlannerContractError(
                f"role {receipt.role!r} is not aligned to the driving entity"
            )

    inputs = tuple(
        RoleItems(receipt.role, (receipt.item,))
        for receipt in receipts
        if receipt.item is not None
    )
    if len(inputs) != len(receipts):
        raise PlannerContractError("settled unary receipt lost its logical item")
    output_slots = (
        ()
        if node.kind is Primitive.EXPAND
        else tuple(
            ItemRef(port, driving_item.entity) for port in node.output_ports
        )
    )
    grain_id = _grain_id_for(node.kind, run_salt, node.id, inputs)
    causes = tuple(
        receipt.cause
        for receipt in receipts
        if receipt.state in {ReceiptState.FAILED, ReceiptState.SUPPRESSED}
        and receipt.cause is not None
    )
    if causes:
        return PlanDecision(
            PlanAction.ENSURE_SUPPRESSED,
            GrainRecord.sealed(
                id=grain_id,
                node=node.id,
                inputs=inputs,
                output_slots=output_slots,
                outcome=Suppressed(causes),
            ),
        )
    if any(receipt.state is not ReceiptState.PRESENT for receipt in receipts):
        raise PlannerContractError("unhandled unary receipt combination")
    return PlanDecision(
        PlanAction.ENSURE_EXECUTABLE,
        GrainRecord(
            id=grain_id,
            node=node.id,
            inputs=inputs,
            output_slots=output_slots,
        ),
    )


def plan_map(
    node: NodeSpec,
    run_salt: bytes,
    receipts: tuple[BindingReceipt, ...],
) -> PlanDecision:
    if node.kind is not Primitive.MAP:
        raise PlannerContractError("plan_map requires a Map NodeSpec")
    return _plan_unary(node, run_salt, receipts, "primary")


def plan_filter(
    node: NodeSpec,
    run_salt: bytes,
    receipts: tuple[BindingReceipt, ...],
) -> PlanDecision:
    if node.kind is not Primitive.FILTER:
        raise PlannerContractError("plan_filter requires a Filter NodeSpec")
    return _plan_unary(node, run_salt, receipts, "target")


def plan_expand(
    node: NodeSpec,
    run_salt: bytes,
    receipts: tuple[BindingReceipt, ...],
) -> PlanDecision:
    if node.kind is not Primitive.EXPAND:
        raise PlannerContractError("plan_expand requires an Expand NodeSpec")
    return _plan_unary(node, run_salt, receipts, "parent")


def plan_reduce(
    node: NodeSpec,
    run_salt: bytes,
    anchor: BindingReceipt,
    barrier: FiberBarrier,
) -> PlanDecision:
    if node.kind is not Primitive.REDUCE:
        raise PlannerContractError("plan_reduce requires a Reduce NodeSpec")
    if anchor.role != "anchor":
        raise PlannerContractError("Reduce anchor receipt has wrong role")
    if anchor.state is ReceiptState.NORMAL_ABSENCE:
        return PlanDecision(PlanAction.NORMAL_ABSENCE)
    if anchor.state is ReceiptState.PENDING:
        return PlanDecision(PlanAction.WAIT)
    if anchor.item is None:
        raise PlannerContractError("settled Reduce anchor lost its logical item")
    if barrier.id != FiberId(node.id, anchor.item):
        raise PlannerContractError("FiberBarrier belongs to another Reduce grain")

    slots = tuple(ItemRef(port, anchor.item.entity) for port in node.output_ports)
    grain_id = reduce_grain_id(run_salt, node.id, anchor.item)
    anchor_binding = RoleItems("anchor", (anchor.item,))

    if anchor.state in {ReceiptState.FAILED, ReceiptState.SUPPRESSED}:
        assert anchor.cause is not None
        return PlanDecision(
            PlanAction.ENSURE_SUPPRESSED,
            GrainRecord.sealed(
                id=grain_id,
                node=node.id,
                inputs=(anchor_binding,),
                output_slots=slots,
                outcome=Suppressed((anchor.cause,)),
            ),
        )

    if barrier.state is FiberState.OPEN:
        return PlanDecision(PlanAction.WAIT)
    if barrier.state is FiberState.READY:
        return PlanDecision(
            PlanAction.ENSURE_EXECUTABLE,
            GrainRecord(
                id=grain_id,
                node=node.id,
                inputs=(
                    anchor_binding,
                    RoleItems("members", barrier.present_members()),
                ),
                output_slots=slots,
            ),
        )

    inputs = (
        (anchor_binding,)
        if barrier.expected is None
        else (
            anchor_binding,
            RoleItems("members", barrier.known_members()),
        )
    )
    causes = barrier.suppression_causes()
    if not causes:
        raise PlannerContractError("suppressed fiber has no direct cause")
    return PlanDecision(
        PlanAction.ENSURE_SUPPRESSED,
        GrainRecord.sealed(
            id=grain_id,
            node=node.id,
            inputs=inputs,
            output_slots=slots,
            outcome=Suppressed(causes),
        ),
    )


def plan_relate(
    node: NodeSpec,
    run_salt: bytes,
    roles: tuple[tuple[str, tuple[tuple[ItemRef, int], ...]], ...],
    *,
    sealed_roles: frozenset[str],
    max_cardinality: int,
) -> "RelatePlanResult":
    return plan_relate_bounded(
        node,
        run_salt,
        roles,
        sealed_roles=sealed_roles,
        max_cardinality=max_cardinality,
    )


@dataclass(frozen=True, slots=True)
class RelatePlanResult:
    decisions: tuple[PlanDecision, ...]
    unmatched: tuple[ItemRef, ...]
    complete: bool


def plan_relate_bounded(
    node: NodeSpec,
    run_salt: bytes,
    roles: tuple[tuple[str, tuple[tuple[ItemRef, int], ...]], ...],
    *,
    sealed_roles: frozenset[str],
    max_cardinality: int,
) -> RelatePlanResult:
    """Bounded sealed-port int-key prototype used by integration coverage."""

    if node.kind is not Primitive.RELATE:
        raise PlannerContractError("bounded Relate needs a Relate NodeSpec")
    role_names = tuple(role for role, _ in roles)
    compiled_roles = tuple(binding.role for binding in node.inputs)
    if role_names != compiled_roles:
        raise PlannerContractError("Relate rows must follow compiled role order")
    if sealed_roles != frozenset(role_names):
        return RelatePlanResult((), (), False)
    if max_cardinality < 0:
        raise PlannerContractError("relation cardinality limit is negative")

    by_role: dict[str, dict[int, list[ItemRef]]] = {}
    for role, rows in roles:
        keyed: dict[int, list[ItemRef]] = {}
        for item, key in rows:
            if type(key) is not int:
                raise PlannerContractError(
                    "bounded Relate accepts exact int keys only"
                )
            keyed.setdefault(key, []).append(item)
        by_role[role] = keyed
    matched_keys = (
        set.intersection(*(set(by_role[role]) for role in role_names))
        if role_names
        else set()
    )

    parent_tuples: list[tuple[ItemRef, ...]] = []
    for key in sorted(matched_keys):
        parent_tuples.extend(
            product(*(by_role[role][key] for role in role_names))
        )
        if len(parent_tuples) > max_cardinality:
            raise PlannerContractError(
                "max_relation_cardinality exceeded for node/arena"
            )

    matched: set[ItemRef] = set()
    decisions: list[PlanDecision] = []
    for parents in parent_tuples:
        inputs = tuple(
            RoleItems(role, (item,))
            for role, item in zip(role_names, parents)
        )
        entity = relate_entity(run_salt, node.id, inputs)
        slots = tuple(ItemRef(port, entity) for port in node.output_ports)
        decisions.append(
            PlanDecision(
                PlanAction.ENSURE_EXECUTABLE,
                GrainRecord(
                    id=relate_grain_id(run_salt, node.id, inputs),
                    node=node.id,
                    inputs=inputs,
                    output_slots=slots,
                ),
            )
        )
        matched.update(parents)
    unmatched = tuple(
        item
        for role, rows in roles
        for item, _ in rows
        if item not in matched
    )
    return RelatePlanResult(tuple(decisions), unmatched, True)


def ensure_decision(
    table: GrainTable,
    decision: PlanDecision,
) -> GrainRecord | None:
    if decision.action in {PlanAction.WAIT, PlanAction.NORMAL_ABSENCE}:
        if decision.grain is not None:
            raise PlannerContractError("non-ensure decision cannot carry a grain")
        return None
    if decision.grain is None:
        raise PlannerContractError("ensure decision must carry a grain")
    try:
        if decision.action is PlanAction.ENSURE_EXECUTABLE:
            return table.ensure_executable(decision.grain)
        return table.ensure_suppressed(decision.grain)
    except GrainInvariantError as error:
        raise PlannerContractError(str(error)) from error


def admit_source(
    node: NodeSpec,
    run_salt: bytes,
    position: int,
    *,
    failure: str | None = None,
) -> GrainRecord:
    if node.kind is not Primitive.SOURCE or len(node.output_ports) != 1:
        raise PlannerContractError("source admission requires one Source port")
    port = node.output_ports[0]
    entity = source_entity(run_salt, port, position)
    item = ItemRef(port, entity)
    outcome = (
        Success(((Emission(item, 0),),))
        if failure is None
        else Failed(GrainFailure("source", failure, ()))
    )
    return GrainRecord.sealed(
        id=source_grain_id(run_salt, port, position),
        node=node.id,
        inputs=(),
        output_slots=(item,),
        outcome=outcome,
    )


def map_success(node: NodeSpec, record: GrainRecord) -> Success:
    if node.kind is not Primitive.MAP:
        raise PlannerContractError("map_success requires a Map node")
    return Success(
        tuple((Emission(item, 0),) for item in record.output_slots)
    )


def filter_success(
    node: NodeSpec,
    record: GrainRecord,
    *,
    keep: bool,
) -> Success:
    if node.kind is not Primitive.FILTER:
        raise PlannerContractError("filter_success requires a Filter node")
    if keep:
        return Success(
            tuple((Emission(item, 0),) for item in record.output_slots)
        )
    return Success(tuple(() for _ in node.output_ports))


def expand_success(
    node: NodeSpec,
    record: GrainRecord,
    run_salt: bytes,
    *,
    cardinality: int,
) -> Success:
    if node.kind is not Primitive.EXPAND:
        raise PlannerContractError("expand_success requires an Expand node")
    if cardinality < 0:
        raise PlannerContractError("Expand cardinality must be non-negative")
    parent = _binding_item(record.inputs, "parent")
    entities = tuple(
        expand_entity(run_salt, node.id, parent.entity, ordinal)
        for ordinal in range(cardinality)
    )
    return Success(
        tuple(
            tuple(
                Emission(ItemRef(port, entity), ordinal)
                for ordinal, entity in enumerate(entities)
            )
            for port in node.output_ports
        )
    )


def reduce_success(node: NodeSpec, record: GrainRecord) -> Success:
    if node.kind is not Primitive.REDUCE:
        raise PlannerContractError("reduce_success requires a Reduce node")
    return Success(
        tuple((Emission(item, 0),) for item in record.output_slots)
    )


def failed_outcome(
    record: GrainRecord,
    *,
    kind: str,
    message: str,
) -> Failed:
    return Failed(
        GrainFailure(
            kind,
            message,
            tuple(item for role in record.inputs for item in role.items),
        )
    )


def register_expand_origins(
    index: ExpandOriginIndex,
    record: GrainRecord,
    outcome: Success,
) -> None:
    parent = _binding_item(record.inputs, "parent")
    for emissions in outcome.emissions_by_port:
        for emission in emissions:
            index.add(
                emission.item.entity,
                ExpandOrigin(parent, emission.ordinal, record.id),
            )


def _binding_item(inputs: tuple[RoleItems, ...], role: str) -> ItemRef:
    matches = [binding for binding in inputs if binding.role == role]
    if len(matches) != 1 or len(matches[0].items) != 1:
        raise PlannerContractError(f"grain needs one item for role {role!r}")
    return matches[0].items[0]
