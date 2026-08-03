"""Single-writer mutable state and owner-ledger stores for Multigrain V3."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, TypeAlias

from .semantics import (
    BlockId,
    CreditReservationId,
    DispatchId,
    GrainId,
    InvariantViolation,
    ItemRef,
    NodeId,
    OccurrenceContext,
    PortId,
    Receipt,
    ReceiptState,
    RootId,
    ScopeDefId,
    ScopeInstanceId,
    StructuralLeaseId,
    ValueNodeId,
)


class ScopeState(Enum):
    """Lifecycle phase of one dynamic scope instance."""

    PENDING_OPEN = "pending_open"
    OPEN = "open"
    ABSENT = "absent"
    FAILED = "failed"
    RECLAIMED = "reclaimed"


@dataclass(slots=True)
class ScopeInstance:
    """Mutable lifecycle record created for one EXPAND parent coordinate."""

    id: ScopeInstanceId
    definition: ScopeDefId
    root: RootId
    parent_item: ItemRef
    parent_context: OccurrenceContext
    expected: int | None
    state: ScopeState
    origin_grain: GrainId
    failure: GrainId | None = None
    structural_lease: StructuralLeaseId | None = None

    def __post_init__(self) -> None:
        """Validate that lifecycle-specific fields form one legal state."""

        if self.parent_context.root != self.root:
            raise ValueError("scope parent context must belong to scope root")
        if self.expected is not None and self.expected < 0:
            raise ValueError("scope expected width must be non-negative")
        if self.state is ScopeState.PENDING_OPEN:
            if (
                self.expected is not None
                or self.failure is not None
                or self.structural_lease is not None
            ):
                raise ValueError("pending scope cannot have terminal/open data")
        elif self.state is ScopeState.OPEN:
            if self.expected is None or self.structural_lease is None:
                raise ValueError("open scope requires width and structural lease")
            if self.failure is not None:
                raise ValueError("open scope cannot carry a failure")
        elif self.state is ScopeState.ABSENT:
            if self.expected is not None or self.failure is not None:
                raise ValueError("absent scope has neither width nor failure")
        elif self.state is ScopeState.FAILED:
            if self.expected is not None or self.failure is None:
                raise ValueError("failed scope requires only a failure grain")


@dataclass(frozen=True, slots=True)
class ScopeOpened:
    """Event announcing an immutable known fan-out width."""

    scope: ScopeInstanceId
    expected: int

    def __post_init__(self) -> None:
        """Validate the announced immutable fan-out width."""

        if self.expected < 0:
            raise ValueError("opened scope width must be non-negative")


@dataclass(frozen=True, slots=True)
class ScopeAbsent:
    """Event closing reducers when the EXPAND origin is normally absent."""

    scope: ScopeInstanceId
    origin: Receipt

    def __post_init__(self) -> None:
        """Require a normal-absence origin rather than failure overloading."""

        if self.origin.state is not ReceiptState.NORMAL_ABSENCE:
            raise ValueError("ScopeAbsent origin must be NORMAL_ABSENCE")


@dataclass(frozen=True, slots=True)
class ScopeFailed:
    """Event closing reducers when EXPAND itself is failed or suppressed."""

    scope: ScopeInstanceId
    cause: GrainId


ScopeEvent: TypeAlias = ScopeOpened | ScopeAbsent | ScopeFailed


@dataclass(frozen=True, slots=True)
class CompressedOrdinalSet:
    """Canonical sorted ordinal set with a compact immutable interface."""

    ordinals: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Require canonical ascending, duplicate-free ordinals."""

        if any(
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal < 0
            for ordinal in self.ordinals
        ):
            raise ValueError("ordinals must be non-negative integers")
        if tuple(sorted(set(self.ordinals))) != self.ordinals:
            raise ValueError("ordinals must be sorted and unique")

    @classmethod
    def from_ordinals(
        cls,
        ordinals: Iterator[int] | tuple[int, ...] | list[int],
    ) -> "CompressedOrdinalSet":
        """Canonicalize an ordinal iterable into sorted unique storage."""

        return cls(tuple(sorted(set(ordinals))))

    def __contains__(self, ordinal: object) -> bool:
        """Return whether an ordinal is represented."""

        return ordinal in self.ordinals

    def __iter__(self) -> Iterator[int]:
        """Iterate ordinals in ascending order."""

        return iter(self.ordinals)

    def __len__(self) -> int:
        """Return represented ordinal count."""

        return len(self.ordinals)


@dataclass(frozen=True, slots=True)
class ReduceBinding:
    """Canonical lineage retained by one REDUCE GrainSpec."""

    scope: ScopeInstanceId
    anchor: ItemRef
    allocation_owner: tuple[ScopeInstanceId, NodeId] | None
    expected: int | None
    present_members: tuple[ItemRef, ...] = ()
    absent_ordinals: CompressedOrdinalSet = CompressedOrdinalSet()
    direct_failures: tuple[tuple[int, GrainId], ...] = ()
    origin_receipt: Receipt | None = None
    origin_failure: GrainId | None = None

    def __post_init__(self) -> None:
        """Validate the mutually exclusive known/absent/failed forms."""

        origin_forms = int(self.origin_receipt is not None) + int(
            self.origin_failure is not None
        )
        if self.expected is None:
            if origin_forms != 1:
                raise ValueError(
                    "unknown-width reduce binding needs one origin terminal"
                )
            if (
                self.allocation_owner is not None
                or self.present_members
                or self.absent_ordinals
                or self.direct_failures
            ):
                raise ValueError(
                    "origin-terminal binding cannot contain known-width slots"
                )
            if (
                self.origin_receipt is not None
                and self.origin_receipt.state
                is not ReceiptState.NORMAL_ABSENCE
            ):
                raise ValueError(
                    "origin receipt must represent normal absence"
                )
            return
        if self.expected < 0:
            raise ValueError("reduce expected width must be non-negative")
        if origin_forms:
            raise ValueError(
                "known-width binding cannot carry an origin terminal"
            )
        if self.allocation_owner is None:
            raise ValueError(
                "known-width binding requires its structural allocation owner"
            )
        if self.allocation_owner[0] != self.scope:
            raise ValueError("allocation owner must belong to binding scope")
        failure_ordinals = tuple(ordinal for ordinal, _ in self.direct_failures)
        if tuple(sorted(set(failure_ordinals))) != failure_ordinals:
            raise ValueError("failure ordinals must be sorted and unique")
        if any(
            ordinal < 0 or ordinal >= self.expected
            for ordinal in failure_ordinals
        ):
            raise ValueError("failure ordinal outside reduce width")
        if any(ordinal >= self.expected for ordinal in self.absent_ordinals):
            raise ValueError("absent ordinal outside reduce width")
        if set(failure_ordinals).intersection(self.absent_ordinals.ordinals):
            raise ValueError("absent and failed ordinals must not overlap")
        total = (
            len(self.present_members)
            + len(self.absent_ordinals)
            + len(self.direct_failures)
        )
        if total != self.expected:
            raise ValueError("reduce binding must account for every ordinal")


@dataclass(slots=True)
class ClosureView:
    """Independent per-REDUCE receipt slots for one known-width scope."""

    reduce_node: NodeId
    input_port: PortId
    scope: ScopeInstanceId
    expected: int
    allocation_owner: tuple[ScopeInstanceId, NodeId]
    settled: int = field(default=0, init=False)
    slots: list[Receipt | None] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Allocate exactly one terminal slot for every known ordinal."""

        if self.expected < 0:
            raise ValueError("closure expected width must be non-negative")
        if self.allocation_owner != (self.scope, self.reduce_node):
            raise ValueError("closure allocation owner must be (scope, node)")
        self.slots = [None] * self.expected

    @property
    def ready(self) -> bool:
        """Whether every known ordinal has reached a terminal receipt."""

        return self.settled == self.expected

    def settle(self, receipt: Receipt) -> bool:
        """Install one ordinal receipt and report whether closure is ready."""

        if receipt.item.port != self.input_port:
            raise InvariantViolation("closure receipt uses the wrong input port")
        if not receipt.context.positions:
            raise InvariantViolation("closure receipt has no scope position")
        position = receipt.context.positions[-1]
        if position.instance != self.scope:
            raise InvariantViolation("closure receipt uses the wrong scope")
        if position.ordinal >= self.expected:
            raise InvariantViolation("closure receipt ordinal is out of range")
        previous = self.slots[position.ordinal]
        if previous is not None:
            if previous == receipt:
                return self.ready
            raise InvariantViolation(
                "conflicting terminal receipt for one closure ordinal"
            )
        self.slots[position.ordinal] = receipt
        self.settled += 1
        return self.ready

    def present_members(self) -> tuple[ItemRef, ...]:
        """Return PRESENT members in ordinal order after closure is ready."""

        self._require_ready()
        return tuple(
            receipt.item
            for receipt in self.slots
            if receipt is not None
            and receipt.state is ReceiptState.PRESENT
        )

    def direct_causes(self) -> tuple[GrainId, ...]:
        """Return canonical direct failed/suppressed producers by ordinal."""

        self._require_ready()
        causes: list[GrainId] = []
        seen: set[GrainId] = set()
        for receipt in self.slots:
            if (
                receipt is not None
                and receipt.state.is_failure
                and receipt.producer not in seen
            ):
                seen.add(receipt.producer)
                causes.append(receipt.producer)
        return tuple(causes)

    def to_binding(self, scope: ScopeInstance) -> ReduceBinding:
        """Compact a ready closure into a durable REDUCE lineage binding."""

        self._require_ready()
        if scope.id != self.scope or scope.state is not ScopeState.OPEN:
            raise InvariantViolation("closure does not match an open scope")
        absent = CompressedOrdinalSet(
            tuple(
                ordinal
                for ordinal, receipt in enumerate(self.slots)
                if receipt is not None
                and receipt.state is ReceiptState.NORMAL_ABSENCE
            )
        )
        failures = tuple(
            (ordinal, receipt.producer)
            for ordinal, receipt in enumerate(self.slots)
            if receipt is not None and receipt.state.is_failure
        )
        return ReduceBinding(
            scope=self.scope,
            anchor=scope.parent_item,
            allocation_owner=self.allocation_owner,
            expected=self.expected,
            present_members=self.present_members(),
            absent_ordinals=absent,
            direct_failures=failures,
        )

    def _require_ready(self) -> None:
        """Reject attempts to observe a partial closure."""

        if not self.ready:
            raise InvariantViolation("REDUCE closure is not terminal")


@dataclass(slots=True)
class ScopeTracker:
    """Store dynamic scopes and independent reducer closure views."""

    instances: dict[ScopeInstanceId, ScopeInstance] = field(
        default_factory=dict
    )
    closures: dict[tuple[NodeId, ScopeInstanceId], ClosureView] = field(
        default_factory=dict
    )
    _by_root: dict[RootId, set[ScopeInstanceId]] = field(
        default_factory=dict,
        init=False,
    )
    _absence_origins: dict[ScopeInstanceId, Receipt] = field(
        default_factory=dict,
        init=False,
    )

    def create_pending(
        self,
        id: ScopeInstanceId,
        definition: ScopeDefId,
        root: RootId,
        parent_item: ItemRef,
        parent_context: OccurrenceContext,
        origin_grain: GrainId,
    ) -> ScopeInstance:
        """Create the unique pending scope for an EXPAND parent."""

        previous = self.instances.get(id)
        if previous is not None:
            static_facts = (
                previous.definition,
                previous.root,
                previous.parent_item,
                previous.parent_context,
                previous.origin_grain,
            )
            requested = (
                definition,
                root,
                parent_item,
                parent_context,
                origin_grain,
            )
            if static_facts != requested:
                raise InvariantViolation("conflicting dynamic scope identity")
            return previous
        instance = ScopeInstance(
            id=id,
            definition=definition,
            root=root,
            parent_item=parent_item,
            parent_context=parent_context,
            expected=None,
            state=ScopeState.PENDING_OPEN,
            origin_grain=origin_grain,
        )
        self.instances[id] = instance
        self._by_root.setdefault(root, set()).add(id)
        return instance

    def open(
        self,
        scope: ScopeInstanceId,
        expected: int,
        lease: StructuralLeaseId,
    ) -> ScopeOpened:
        """Transition a pending scope to OPEN with immutable width."""

        instance = self._require(scope)
        if instance.state is ScopeState.OPEN:
            if (
                instance.expected == expected
                and instance.structural_lease == lease
            ):
                return ScopeOpened(scope, expected)
            raise InvariantViolation("OPEN scope width/lease cannot change")
        if instance.state is not ScopeState.PENDING_OPEN:
            raise InvariantViolation("only a pending scope can open")
        if expected < 0:
            raise ValueError("scope width must be non-negative")
        instance.expected = expected
        instance.structural_lease = lease
        instance.state = ScopeState.OPEN
        return ScopeOpened(scope, expected)

    def mark_absent(
        self,
        scope: ScopeInstanceId,
        origin: Receipt,
    ) -> ScopeAbsent:
        """Close a pending scope through normal origin absence."""

        instance = self._require(scope)
        event = ScopeAbsent(scope, origin)
        if instance.state is ScopeState.ABSENT:
            if self._absence_origins.get(scope) != origin:
                raise InvariantViolation("ABSENT scope origin cannot change")
            return event
        if instance.state is not ScopeState.PENDING_OPEN:
            raise InvariantViolation("only a pending scope can become absent")
        if origin.context.root != instance.root:
            raise InvariantViolation("scope absence belongs to another root")
        self._absence_origins[scope] = origin
        instance.state = ScopeState.ABSENT
        return event

    def mark_failed(
        self,
        scope: ScopeInstanceId,
        cause: GrainId,
    ) -> ScopeFailed:
        """Close a pending scope through its EXPAND failure grain."""

        instance = self._require(scope)
        if instance.state is ScopeState.FAILED:
            if instance.failure == cause:
                return ScopeFailed(scope, cause)
            raise InvariantViolation("FAILED scope cause cannot change")
        if instance.state is not ScopeState.PENDING_OPEN:
            raise InvariantViolation("only a pending scope can fail")
        instance.failure = cause
        instance.state = ScopeState.FAILED
        return ScopeFailed(scope, cause)

    def ensure_closure(
        self,
        reduce_node: NodeId,
        scope: ScopeInstanceId,
        input_port: PortId,
    ) -> ClosureView:
        """Create one reducer-specific closure after scope width is known."""

        instance = self._require(scope)
        if instance.state is not ScopeState.OPEN or instance.expected is None:
            raise InvariantViolation("closure requires an OPEN scope")
        key = (reduce_node, scope)
        previous = self.closures.get(key)
        if previous is not None:
            if (
                previous.input_port != input_port
                or previous.expected != instance.expected
            ):
                raise InvariantViolation("conflicting closure definition")
            return previous
        closure = ClosureView(
            reduce_node=reduce_node,
            input_port=input_port,
            scope=scope,
            expected=instance.expected,
            allocation_owner=(scope, reduce_node),
        )
        self.closures[key] = closure
        return closure

    def settle(self, reduce_node: NodeId, receipt: Receipt) -> bool:
        """Settle a receipt in an existing reducer closure."""

        if not receipt.context.positions:
            raise InvariantViolation("scoped receipt requires a position")
        scope = receipt.context.positions[-1].instance
        try:
            closure = self.closures[(reduce_node, scope)]
        except KeyError as error:
            raise KeyError("no closure for reducer and scope") from error
        return closure.settle(receipt)

    def reclaim_closure(
        self,
        reduce_node: NodeId,
        scope: ScopeInstanceId,
    ) -> None:
        """Remove one reducer closure without touching sibling reducers."""

        self.closures.pop((reduce_node, scope), None)

    def reclaim_scope(self, scope: ScopeInstanceId) -> None:
        """Reclaim a scope and all remaining closures at root quiescence."""

        instance = self._require(scope)
        for key in tuple(self.closures):
            if key[1] == scope:
                self.closures.pop(key, None)
        instance.state = ScopeState.RECLAIMED
        self.instances.pop(scope, None)
        self._absence_origins.pop(scope, None)
        root_scopes = self._by_root.get(instance.root)
        if root_scopes is not None:
            root_scopes.discard(scope)
            if not root_scopes:
                self._by_root.pop(instance.root, None)

    def remove_root(self, root: RootId) -> None:
        """Reclaim every scope indexed to one quiescent root."""

        for scope in tuple(self._by_root.get(root, ())):
            self.reclaim_scope(scope)

    def definition_of(self, scope: ScopeInstanceId) -> ScopeDefId:
        """Return the static definition of a dynamic scope."""

        return self._require(scope).definition

    def _require(self, scope: ScopeInstanceId) -> ScopeInstance:
        """Return a scope instance or raise for an unknown ID."""

        try:
            return self.instances[scope]
        except KeyError as error:
            raise KeyError(f"unknown scope {scope.hex()}") from error


@dataclass(frozen=True, slots=True)
class ScalarNode:
    """One row in an opaque physical block."""

    block: BlockId
    row: int

    def __post_init__(self) -> None:
        """Validate the non-negative row selector."""

        if self.row < 0:
            raise ValueError("scalar row must be non-negative")


@dataclass(frozen=True, slots=True)
class FlatListNode:
    """One contiguous outer-list interval in a physical block."""

    block: BlockId
    start: int
    stop: int

    def __post_init__(self) -> None:
        """Validate the half-open contiguous row interval."""

        if self.start < 0 or self.stop < self.start:
            raise ValueError("flat list interval must satisfy 0 <= start <= stop")


@dataclass(frozen=True, slots=True)
class CompositeListNode:
    """Structural list whose ordered children are existing value DAG nodes."""

    children: tuple[ValueNodeId, ...]


ValueNode: TypeAlias = ScalarNode | FlatListNode | CompositeListNode


@dataclass(slots=True)
class ValueNodeRecord:
    """A value DAG node plus incoming owner/edge reference count."""

    value: ValueNode
    refs: int = 0

    def __post_init__(self) -> None:
        """Reject impossible negative owner counts."""

        if self.refs < 0:
            raise ValueError("value node refs must be non-negative")


@dataclass(slots=True)
class BlockRecord:
    """Opaque backend handle and its independent logical/dispatch leases."""

    ref: Any
    rows: int
    estimated_bytes: int | None
    logical_node_leases: int = 0
    inflight_dispatch_leases: int = 0

    def __post_init__(self) -> None:
        """Validate row/byte metadata and both independent lease ledgers."""

        if (
            isinstance(self.rows, bool)
            or not isinstance(self.rows, int)
            or self.rows < 0
        ):
            raise ValueError("block row count must be non-negative")
        if self.estimated_bytes is not None and (
            isinstance(self.estimated_bytes, bool)
            or not isinstance(self.estimated_bytes, int)
            or self.estimated_bytes < 0
        ):
            raise ValueError("estimated bytes must be non-negative")
        if (
            self.logical_node_leases < 0
            or self.inflight_dispatch_leases < 0
        ):
            raise ValueError("block lease counts must be non-negative")


@dataclass(slots=True)
class BlockStore:
    """Backend-neutral block registry with owner-keyed dispatch leases."""

    _blocks: dict[BlockId, BlockRecord] = field(
        default_factory=dict,
        init=False,
    )
    _dispatch_owners: set[tuple[DispatchId, BlockId]] = field(
        default_factory=set,
        init=False,
    )
    _next_id: int = field(default=0, init=False)

    def install(
        self,
        ref: Any,
        rows: int,
        estimated_bytes: int | None = None,
    ) -> BlockId:
        """Install an opaque handle and return a fresh session-local BlockId."""

        record = BlockRecord(ref, rows, estimated_bytes)
        block = BlockId(self._next_id)
        self._next_id += 1
        self._blocks[block] = record
        return block

    def preview_ids(self, count: int) -> tuple[BlockId, ...]:
        """Return the IDs the next atomic installation would allocate.

        Previewing does not mutate the store.  The single-writer transaction
        must install the returned IDs contiguously before another preview is
        consumed.
        """

        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("block preview count must be an int")
        if count < 0:
            raise ValueError("block preview count must be non-negative")
        return tuple(BlockId(self._next_id + offset) for offset in range(count))

    def install_prepared(self, prepared: Any) -> BlockId:
        """Install one preallocated block from a transaction DTO."""

        expected = BlockId(self._next_id)
        if getattr(prepared, "id", None) != expected:
            raise InvariantViolation("prepared BlockId is not the next allocation")
        return self.install(
            getattr(prepared, "ref"),
            getattr(prepared, "rows"),
            getattr(prepared, "estimated_bytes", None),
        )

    def resolve_ref(self, block: BlockId) -> Any:
        """Return the opaque backend handle without materializing its payload."""

        return self.get(block).ref

    def get(self, block: BlockId) -> BlockRecord:
        """Return a live block record."""

        try:
            return self._blocks[block]
        except KeyError as error:
            raise KeyError(f"unknown block {int(block)}") from error

    def retain_node(self, block: BlockId) -> None:
        """Retain one logical ValueNode lease on a block."""

        self.get(block).logical_node_leases += 1

    def release_node(self, block: BlockId) -> None:
        """Release one logical node lease and collect an unowned block."""

        record = self.get(block)
        if record.logical_node_leases <= 0:
            raise InvariantViolation("block has no logical node lease")
        record.logical_node_leases -= 1
        self._collect_if_unowned(block, record)

    def retain_dispatch(self, dispatch: DispatchId, block: BlockId) -> bool:
        """Retain an idempotent ``(dispatch, block)`` physical owner."""

        record = self.get(block)
        owner = (dispatch, block)
        if owner in self._dispatch_owners:
            return False
        self._dispatch_owners.add(owner)
        record.inflight_dispatch_leases += 1
        return True

    def release_dispatch(self, dispatch: DispatchId, block: BlockId) -> bool:
        """Release one dispatch owner, returning false if already released."""

        owner = (dispatch, block)
        if owner not in self._dispatch_owners:
            return False
        record = self.get(block)
        self._dispatch_owners.remove(owner)
        if record.inflight_dispatch_leases <= 0:
            raise InvariantViolation("dispatch lease ledger underflow")
        record.inflight_dispatch_leases -= 1
        self._collect_if_unowned(block, record)
        return True

    def discard_unleased(self, block: BlockId) -> None:
        """Discard an installed block that never acquired any owner."""

        record = self.get(block)
        if record.logical_node_leases or record.inflight_dispatch_leases:
            raise InvariantViolation("cannot discard an owned block")
        self._blocks.pop(block)

    def _collect_if_unowned(
        self,
        block: BlockId,
        record: BlockRecord,
    ) -> None:
        """Delete a block record after both independent lease classes drain."""

        if (
            record.logical_node_leases == 0
            and record.inflight_dispatch_leases == 0
        ):
            self._blocks.pop(block, None)

    def __contains__(self, block: object) -> bool:
        """Return whether a block still has at least one live registry record."""

        return block in self._blocks

    def __len__(self) -> int:
        """Return live block count."""

        return len(self._blocks)


@dataclass(slots=True)
class ValueNodeStore:
    """Acyclic value DAG with item ownership and recursive reference GC."""

    blocks: BlockStore = field(default_factory=BlockStore)
    _items: dict[ItemRef, ValueNodeId] = field(
        default_factory=dict,
        init=False,
    )
    _nodes: dict[ValueNodeId, ValueNodeRecord] = field(
        default_factory=dict,
        init=False,
    )
    _items_by_root: dict[RootId, set[ItemRef]] = field(
        default_factory=dict,
        init=False,
    )
    _item_roots: dict[ItemRef, RootId] = field(
        default_factory=dict,
        init=False,
    )
    _next_id: int = field(default=0, init=False)

    def create_scalar(self, block: BlockId, row: int) -> ValueNodeId:
        """Create an unbound scalar node and retain its block."""

        record = self.blocks.get(block)
        if row < 0 or row >= record.rows:
            raise ValueError("scalar row is outside block bounds")
        self.blocks.retain_node(block)
        return self._install_node(ScalarNode(block, row))

    def create_flat_list(
        self,
        block: BlockId,
        start: int,
        stop: int,
    ) -> ValueNodeId:
        """Create an unbound one-layer list node and retain its block."""

        record = self.blocks.get(block)
        if start < 0 or stop < start or stop > record.rows:
            raise ValueError("flat list interval is outside block bounds")
        self.blocks.retain_node(block)
        return self._install_node(FlatListNode(block, start, stop))

    def create_composite(
        self,
        children: tuple[ValueNodeId, ...],
    ) -> ValueNodeId:
        """Create an unbound list node and retain each ordered child edge."""

        for child in children:
            self._require_node(child)
        node = self._install_node(CompositeListNode(children))
        for child in children:
            self._retain(child)
        return node

    def preview_ids(self, count: int) -> tuple[ValueNodeId, ...]:
        """Return contiguous IDs for the next atomic value-node installation."""

        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("value-node preview count must be an int")
        if count < 0:
            raise ValueError("value-node preview count must be non-negative")
        return tuple(
            ValueNodeId(self._next_id + offset) for offset in range(count)
        )

    def install_prepared(self, prepared: Any) -> ValueNodeId:
        """Install one preallocated node while acquiring its physical owners."""

        expected = ValueNodeId(self._next_id)
        if getattr(prepared, "id", None) != expected:
            raise InvariantViolation(
                "prepared ValueNodeId is not the next allocation"
            )
        value = getattr(prepared, "value")
        if isinstance(value, ScalarNode):
            return self.create_scalar(value.block, value.row)
        if isinstance(value, FlatListNode):
            return self.create_flat_list(value.block, value.start, value.stop)
        if isinstance(value, CompositeListNode):
            return self.create_composite(value.children)
        raise TypeError(f"unsupported prepared ValueNode {type(value)!r}")

    def bind_item(
        self,
        root: RootId,
        item: ItemRef,
        node: ValueNodeId,
    ) -> bool:
        """Bind one coordinate to a value owner, idempotently if exact."""

        self._require_node(node)
        previous = self._items.get(item)
        if previous is not None:
            if previous == node and self._item_roots[item] == root:
                return False
            raise InvariantViolation("ItemRef already has another value binding")
        self._items[item] = node
        self._item_roots[item] = root
        self._items_by_root.setdefault(root, set()).add(item)
        self._retain(node)
        return True

    def alias_item(
        self,
        root: RootId,
        output: ItemRef,
        source: ItemRef,
    ) -> bool:
        """Create a new item owner for the exact source ValueNode."""

        try:
            node = self._items[source]
        except KeyError as error:
            raise KeyError("cannot alias an unbound source item") from error
        if self._item_roots[source] != root:
            raise InvariantViolation(
                "value alias cannot cross causal root ownership"
            )
        return self.bind_item(root, output, node)

    def unbind_item(self, item: ItemRef) -> bool:
        """Remove one item owner and recursively collect zero-ref nodes."""

        node = self._items.pop(item, None)
        if node is None:
            return False
        root = self._item_roots.pop(item)
        root_items = self._items_by_root[root]
        root_items.discard(item)
        if not root_items:
            self._items_by_root.pop(root, None)
        self._release(node)
        return True

    def discard_unbound(self, node: ValueNodeId) -> None:
        """Collect a newly prepared node that never acquired an item owner."""

        record = self._require_node(node)
        if record.refs != 0:
            raise InvariantViolation("cannot discard a referenced value node")
        self._collect_zero(node, record)

    def node_for_item(self, item: ItemRef) -> ValueNodeId:
        """Return the value DAG node bound to a PRESENT coordinate."""

        try:
            return self._items[item]
        except KeyError as error:
            raise KeyError("item has no value binding") from error

    def get(self, node: ValueNodeId) -> ValueNode:
        """Return an immutable ValueNode payload."""

        return self._require_node(node).value

    def has_binding(self, item: ItemRef) -> bool:
        """Return whether an ItemRef currently owns a value node."""

        return item in self._items

    def outer_length(self, node: ValueNodeId) -> int:
        """Return the structural outer-list width."""

        value = self.get(node)
        if isinstance(value, FlatListNode):
            return value.stop - value.start
        if isinstance(value, CompositeListNode):
            return len(value.children)
        raise InvariantViolation("EXPAND input is not a structural list node")

    def child_at(self, node: ValueNodeId, ordinal: int) -> ValueNodeId:
        """Resolve one structural child without recursively flattening it."""

        value = self.get(node)
        if ordinal < 0:
            raise IndexError("negative structural ordinal")
        if isinstance(value, FlatListNode):
            if ordinal >= value.stop - value.start:
                raise IndexError("structural ordinal out of range")
            return self.create_scalar(value.block, value.start + ordinal)
        if isinstance(value, CompositeListNode):
            try:
                return value.children[ordinal]
            except IndexError as error:
                raise IndexError("structural ordinal out of range") from error
        raise InvariantViolation("EXPAND input is not a structural list node")

    def remove_root(self, root: RootId) -> None:
        """Unbind all item owners indexed to one reclaimed root."""

        for item in tuple(self._items_by_root.get(root, ())):
            self.unbind_item(item)

    def _install_node(self, value: ValueNode) -> ValueNodeId:
        """Allocate a monotonic node ID for a prepared value."""

        node = ValueNodeId(self._next_id)
        self._next_id += 1
        self._nodes[node] = ValueNodeRecord(value)
        return node

    def _require_node(self, node: ValueNodeId) -> ValueNodeRecord:
        """Return a node record or raise for an unknown/collected ID."""

        try:
            return self._nodes[node]
        except KeyError as error:
            raise KeyError(f"unknown value node {int(node)}") from error

    def _retain(self, node: ValueNodeId) -> None:
        """Increment one item or composite-edge owner."""

        self._require_node(node).refs += 1

    def _release(self, node: ValueNodeId) -> None:
        """Release one owner and recursively collect a zero-ref DAG subtree."""

        record = self._require_node(node)
        if record.refs <= 0:
            raise InvariantViolation("value node reference ledger underflow")
        record.refs -= 1
        if record.refs == 0:
            self._collect_zero(node, record)

    def _collect_zero(
        self,
        node: ValueNodeId,
        record: ValueNodeRecord,
    ) -> None:
        """Collect one zero-ref node and release its physical/child owners."""

        self._nodes.pop(node)
        value = record.value
        if isinstance(value, (ScalarNode, FlatListNode)):
            self.blocks.release_node(value.block)
        else:
            for child in value.children:
                self._release(child)

    def __len__(self) -> int:
        """Return live ValueNode count."""

        return len(self._nodes)


@dataclass(frozen=True, slots=True)
class _ControlRecord:
    """Internal bool value plus its static remaining-consumer count."""

    root: RootId
    value: bool
    remaining_consumers: int


@dataclass(slots=True)
class ControlStore:
    """Retain FILTER bool projections until all consumers are terminal."""

    _bools: dict[ItemRef, _ControlRecord] = field(
        default_factory=dict,
        init=False,
    )
    _by_root: dict[RootId, set[ItemRef]] = field(
        default_factory=dict,
        init=False,
    )

    def install_bool(
        self,
        root: RootId,
        item: ItemRef,
        value: bool,
        remaining_consumers: int = 1,
    ) -> bool:
        """Install a strict bool and its positive static consumer count."""

        if type(value) is not bool:
            raise TypeError("control projection value must be exactly bool")
        if remaining_consumers <= 0:
            raise ValueError("control bool needs at least one consumer")
        requested = _ControlRecord(root, value, remaining_consumers)
        previous = self._bools.get(item)
        if previous is not None:
            if previous == requested:
                return False
            raise InvariantViolation("conflicting control bool installation")
        self._bools[item] = requested
        self._by_root.setdefault(root, set()).add(item)
        return True

    def get_bool(self, item: ItemRef) -> bool:
        """Return a retained FILTER control bit."""

        try:
            return self._bools[item].value
        except KeyError as error:
            raise KeyError("control bit is not retained") from error

    def release_consumer(self, item: ItemRef) -> bool:
        """Release one static consumer and delete the bit at zero."""

        try:
            previous = self._bools[item]
        except KeyError:
            return False
        remaining = previous.remaining_consumers - 1
        if remaining < 0:
            raise InvariantViolation("control consumer ledger underflow")
        if remaining:
            self._bools[item] = _ControlRecord(
                previous.root,
                previous.value,
                remaining,
            )
        else:
            self._bools.pop(item)
            root_items = self._by_root[previous.root]
            root_items.discard(item)
            if not root_items:
                self._by_root.pop(previous.root, None)
        return True

    def remove_root(self, root: RootId) -> None:
        """Drop all retained controls for an aborted/reclaimed root."""

        for item in self._by_root.pop(root, set()):
            self._bools.pop(item, None)

    def __len__(self) -> int:
        """Return retained control coordinate count."""

        return len(self._bools)


@dataclass(frozen=True, slots=True)
class RuntimeLimits:
    """Hard count/depth limits required for bounded V3 execution."""

    max_active_roots: int
    max_live_grains: int
    max_live_occurrences: int
    max_occurrences_per_root: int
    max_live_structural_edges: int
    max_structural_edges_per_root: int
    max_scope_width: int
    max_pending_expansions: int
    max_pending_dispatches: int
    max_local_events: int
    max_gather_depth: int
    max_gather_nodes_per_entry: int
    max_refs_per_dispatch: int
    max_manifest_bytes: int
    max_error_message_bytes: int
    max_buffered_results: int
    max_detached_result_nodes: int
    max_detached_result_refs: int

    def __post_init__(self) -> None:
        """Require every correctness/resource-safety limit to be positive."""

        for data_field in self.__dataclass_fields__.values():
            value = getattr(self, data_field.name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"{data_field.name} must be a positive integer"
                )


class StructuralAllocationState(Enum):
    """Ownership phase of one reducer's structural edge allocation."""

    CLOSURE = "closure"
    BINDING = "binding"
    RELEASED = "released"


@dataclass(slots=True)
class StructuralEdgeLease:
    """Per-scope owner ledger for all reducer closure slot allocations."""

    id: StructuralLeaseId
    scope: ScopeInstanceId
    root: RootId
    units: int
    allocations: dict[NodeId, StructuralAllocationState]

    def __post_init__(self) -> None:
        """Validate a fresh closure-owned structural allocation ledger."""

        if self.units < 0:
            raise ValueError("structural lease units must be non-negative")
        if any(
            state is not StructuralAllocationState.CLOSURE
            for state in self.allocations.values()
        ):
            raise ValueError("new structural allocations must own closures")

    def transfer_to_binding(self, reduce_node: NodeId) -> bool:
        """Transfer one reducer owner from closure slots to binding."""

        state = self.allocations.get(reduce_node)
        if state is StructuralAllocationState.BINDING:
            return False
        if state is not StructuralAllocationState.CLOSURE:
            raise InvariantViolation("closure allocation is not transferable")
        self.allocations[reduce_node] = StructuralAllocationState.BINDING
        return True

    def release(self, reduce_node: NodeId) -> bool:
        """Release one reducer allocation idempotently."""

        state = self.allocations.get(reduce_node)
        if state is None:
            raise KeyError("unknown reducer allocation")
        if state is StructuralAllocationState.RELEASED:
            return False
        self.allocations[reduce_node] = StructuralAllocationState.RELEASED
        return True


@dataclass(slots=True)
class PendingExpansion:
    """Bounded materialization cursor for an already successful EXPAND."""

    scope: ScopeInstanceId
    grain: GrainId
    structural_lease: StructuralLeaseId
    occurrence_reservation: CreditReservationId
    input_item: ItemRef
    list_node: ValueNodeId
    expected: int
    next_ordinal: int = 0

    def __post_init__(self) -> None:
        """Validate that the materialization cursor lies within its range."""

        if self.expected < 0:
            raise ValueError("pending expansion width must be non-negative")
        if self.next_ordinal < 0 or self.next_ordinal > self.expected:
            raise ValueError("pending expansion cursor is out of range")

    @property
    def complete(self) -> bool:
        """Whether every deterministic child coordinate was materialized."""

        return self.next_ordinal == self.expected

    def advance(self, count: int) -> range:
        """Reserve the next bounded ordinal chunk and advance the cursor."""

        if count <= 0:
            raise ValueError("expansion chunk count must be positive")
        start = self.next_ordinal
        stop = min(self.expected, start + count)
        self.next_ordinal = stop
        return range(start, stop)


__all__ = [
    "BlockRecord",
    "BlockStore",
    "ClosureView",
    "CompositeListNode",
    "CompressedOrdinalSet",
    "ControlStore",
    "FlatListNode",
    "PendingExpansion",
    "ReduceBinding",
    "RuntimeLimits",
    "ScalarNode",
    "ScopeAbsent",
    "ScopeEvent",
    "ScopeFailed",
    "ScopeInstance",
    "ScopeOpened",
    "ScopeState",
    "ScopeTracker",
    "StructuralAllocationState",
    "StructuralEdgeLease",
    "ValueNode",
    "ValueNodeRecord",
    "ValueNodeStore",
]
