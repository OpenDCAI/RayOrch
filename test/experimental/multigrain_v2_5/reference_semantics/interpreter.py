"""Small deterministic Logical-Grain reference interpreter."""

from __future__ import annotations

from itertools import product
from typing import Iterable, Mapping

from .identity_oracle import (
    RUN_SALT_BYTES,
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
from .models import (
    Emission,
    Failed,
    GrainFailure,
    GrainId,
    GrainRecord,
    ItemRef,
    MemberSettlement,
    PortId,
    RelateResult,
    RoleItems,
    SettlementKind,
    Success,
    Suppressed,
)


class SourcePositionOracle:
    """Run-scoped per-source-port logical ordinal allocator."""

    def __init__(self) -> None:
        self._next: dict[PortId, int] = {}

    def allocate(self, source_port: PortId) -> int:
        position = self._next.get(source_port, 0)
        self._next[source_port] = position + 1
        return position

    def peek(self, source_port: PortId) -> int:
        return self._next.get(source_port, 0)


class ReferenceInterpreter:
    """Executable oracle for Phase 0 golden and property fixtures."""

    def __init__(self, run_salt: bytes) -> None:
        if type(run_salt) is not bytes or len(run_salt) != RUN_SALT_BYTES:
            raise ValueError("run_salt must be exactly 16 bytes")
        self.run_salt = run_salt

    def source(
        self,
        source_port: PortId,
        position: int,
        *,
        failure: str | None = None,
    ) -> GrainRecord:
        entity = source_entity(self.run_salt, source_port, position)
        item = ItemRef(source_port, entity)
        if failure is None:
            outcome = Success(((Emission(item, 0),),))
        else:
            outcome = Failed(GrainFailure("source", failure, ()))
        return GrainRecord(
            id=source_grain_id(self.run_salt, source_port, position),
            node=source_port.node,
            inputs=(),
            output_slots=(item,),
            outcome=outcome,
        )

    @staticmethod
    def _driving(inputs: tuple[RoleItems, ...]) -> ItemRef:
        if not inputs or len(inputs[0].items) != 1:
            raise ValueError("unary oracle requires one driving item")
        return inputs[0].items[0]

    @staticmethod
    def _output_slots(
        driving: ItemRef,
        output_ports: tuple[PortId, ...],
    ) -> tuple[ItemRef, ...]:
        return tuple(ItemRef(port, driving.entity) for port in output_ports)

    @staticmethod
    def _input_causes(inputs: tuple[RoleItems, ...]) -> tuple[ItemRef, ...]:
        return tuple(item for role in inputs for item in role.items)

    def map(
        self,
        node: int,
        inputs: tuple[RoleItems, ...],
        output_ports: tuple[PortId, ...],
        *,
        failure: str | None = None,
        suppressed_by: tuple[GrainId, ...] = (),
    ) -> GrainRecord:
        driving = self._driving(inputs)
        slots = self._output_slots(driving, output_ports)
        if suppressed_by:
            outcome = Suppressed(suppressed_by)
        elif failure is not None:
            outcome = Failed(
                GrainFailure(
                    "udf",
                    failure,
                    self._input_causes(inputs),
                )
            )
        else:
            outcome = Success(tuple((Emission(item, 0),) for item in slots))
        return GrainRecord(
            map_grain_id(self.run_salt, node, inputs),
            node,
            inputs,
            slots,
            outcome,
        )

    def filter(
        self,
        node: int,
        inputs: tuple[RoleItems, ...],
        output_ports: tuple[PortId, ...],
        *,
        keep: bool | None = True,
        failure: str | None = None,
        suppressed_by: tuple[GrainId, ...] = (),
    ) -> GrainRecord:
        driving = self._driving(inputs)
        slots = self._output_slots(driving, output_ports)
        if suppressed_by:
            outcome = Suppressed(suppressed_by)
        elif failure is not None:
            outcome = Failed(
                GrainFailure(
                    "udf",
                    failure,
                    self._input_causes(inputs),
                )
            )
        elif keep is True:
            outcome = Success(tuple((Emission(item, 0),) for item in slots))
        elif keep is False:
            outcome = Success(tuple(() for _ in slots))
        else:
            raise ValueError("keep must be bool unless failure/suppression is set")
        return GrainRecord(
            filter_grain_id(self.run_salt, node, inputs),
            node,
            inputs,
            slots,
            outcome,
        )

    def expand(
        self,
        node: int,
        inputs: tuple[RoleItems, ...],
        output_ports: tuple[PortId, ...],
        cardinality: int | None,
        *,
        failure: str | None = None,
        suppressed_by: tuple[GrainId, ...] = (),
    ) -> GrainRecord:
        parent = self._driving(inputs)
        if suppressed_by:
            outcome = Suppressed(suppressed_by)
        elif failure is not None:
            outcome = Failed(
                GrainFailure(
                    "udf",
                    failure,
                    self._input_causes(inputs),
                )
            )
        else:
            if cardinality is None or cardinality < 0:
                raise ValueError("successful Expand needs non-negative cardinality")
            entities = tuple(
                expand_entity(self.run_salt, node, parent.entity, ordinal)
                for ordinal in range(cardinality)
            )
            outcome = Success(
                tuple(
                    tuple(
                        Emission(ItemRef(port, entity), ordinal)
                        for ordinal, entity in enumerate(entities)
                    )
                    for port in output_ports
                )
            )
        return GrainRecord(
            expand_grain_id(self.run_salt, node, inputs),
            node,
            inputs,
            (),
            outcome,
        )

    def reduce(
        self,
        node: int,
        anchor: ItemRef,
        members_port: PortId,
        output_ports: tuple[PortId, ...],
        *,
        expected: int | None,
        settlements: Iterable[MemberSettlement] = (),
        origin_failure: GrainId | None = None,
    ) -> GrainRecord:
        grain_id = reduce_grain_id(self.run_salt, node, anchor)
        slots = tuple(ItemRef(port, anchor.entity) for port in output_ports)
        anchor_binding = RoleItems("anchor", (anchor,))

        if expected is None:
            if origin_failure is None:
                raise ValueError("unknown cardinality remains open without origin failure")
            return GrainRecord(
                grain_id,
                node,
                (anchor_binding,),
                slots,
                Suppressed((origin_failure,)),
            )
        if expected < 0:
            raise ValueError("expected cardinality must be non-negative")

        ordered = sorted(settlements, key=lambda settlement: settlement.ordinal)
        ordinals = [settlement.ordinal for settlement in ordered]
        if ordinals != list(range(expected)):
            raise ValueError(
                f"settlements must cover each ordinal in [0,{expected}): {ordinals}"
            )

        members: list[ItemRef] = []
        failure_receipts: list[GrainId] = []
        for settlement in ordered:
            if settlement.kind is SettlementKind.PRESENT:
                if settlement.item is None or settlement.receipt is not None:
                    raise ValueError("present settlement requires only an item")
                if settlement.item.port != members_port:
                    raise ValueError("present item must be on the members port")
                members.append(settlement.item)
            elif settlement.kind is SettlementKind.DROPPED:
                if settlement.item is not None or settlement.receipt is not None:
                    raise ValueError("dropped settlement has no binding or cause")
            else:
                if settlement.item is None or settlement.receipt is None:
                    raise ValueError(
                        "failed/suppressed settlement needs item and receipt"
                    )
                if settlement.item.port != members_port:
                    raise ValueError(
                        "failure coordinate must be on the final members port"
                    )
                members.append(settlement.item)
                failure_receipts.append(settlement.receipt)

        inputs = (anchor_binding, RoleItems("members", tuple(members)))
        if failure_receipts:
            outcome = Suppressed(tuple(failure_receipts))
        else:
            outcome = Success(tuple((Emission(item, 0),) for item in slots))
        return GrainRecord(grain_id, node, inputs, slots, outcome)

    def relate(
        self,
        node: int,
        roles: Mapping[str, tuple[tuple[ItemRef, int], ...]],
        output_ports: tuple[PortId, ...],
        *,
        sealed_roles: frozenset[str],
        max_cardinality: int = 10_000,
    ) -> RelateResult:
        role_names = tuple(roles)
        if sealed_roles != frozenset(role_names):
            return RelateResult((), (), False)

        by_role: dict[str, dict[int, list[ItemRef]]] = {}
        for role in role_names:
            keyed: dict[int, list[ItemRef]] = {}
            for item, key in roles[role]:
                if type(key) is not int:
                    raise TypeError("Phase 0 Relate accepts exact int keys only")
                keyed.setdefault(key, []).append(item)
            by_role[role] = keyed

        if role_names:
            matched_keys = set.intersection(
                *(set(by_role[role]) for role in role_names)
            )
        else:
            matched_keys = set()

        parent_tuples: list[tuple[ItemRef, ...]] = []
        for key in sorted(matched_keys):
            parent_tuples.extend(
                product(*(by_role[role][key] for role in role_names))
            )
            if len(parent_tuples) > max_cardinality:
                raise ValueError("relation cardinality exceeds guard")

        records: list[GrainRecord] = []
        matched_items: set[ItemRef] = set()
        for parents in parent_tuples:
            inputs = tuple(
                RoleItems(role, (item,))
                for role, item in zip(role_names, parents)
            )
            entity = relate_entity(self.run_salt, node, inputs)
            slots = tuple(ItemRef(port, entity) for port in output_ports)
            records.append(
                GrainRecord(
                    relate_grain_id(self.run_salt, node, inputs),
                    node,
                    inputs,
                    slots,
                    Success(tuple((Emission(item, 0),) for item in slots)),
                )
            )
            matched_items.update(parents)

        unmatched = tuple(
            item
            for role in role_names
            for item, _ in roles[role]
            if item not in matched_items
        )
        return RelateResult(tuple(records), unmatched, True)
