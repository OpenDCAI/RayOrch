"""Local dispatch preparation and single-writer semantic transactions.

The objects in this module deliberately stay on the driver.  In particular,
``StorageGather`` contains central ``BlockId`` values and must be converted to
the versioned wire selectors owned by :mod:`multigrain_v3.ray.protocol` before
it crosses a process boundary.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Protocol, TypeAlias

from ..model import semantics as _sem
from ..model import state as _state


class DispatchInvariantError(RuntimeError):
    """Report a malformed dispatch or a violated transaction invariant."""


@dataclass(frozen=True, slots=True)
class BlockTake:
    """Select one row from a block retained by the central ``BlockStore``."""

    block: Any
    row: int

    def __post_init__(self) -> None:
        """Reject negative row selectors before they reach a worker."""

        if self.row < 0:
            raise ValueError("BlockTake.row must be non-negative")


@dataclass(frozen=True, slots=True)
class StorageList:
    """Represent a structural list whose leaves are central block selectors."""

    children: tuple["StorageGather", ...]

    def __post_init__(self) -> None:
        """Require an immutable child sequence for deterministic traversal."""

        if not isinstance(self.children, tuple):
            raise TypeError("StorageList.children must be a tuple")


StorageGather: TypeAlias = BlockTake | StorageList


@dataclass(frozen=True, slots=True)
class LocalSlotTake:
    """Fallback wire selector used when the Ray protocol is not imported."""

    ref_slot: int
    row: int


@dataclass(frozen=True, slots=True)
class LocalWireList:
    """Fallback wire list used by pure local unit tests."""

    children: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class LocalWorkerEntry:
    """Fallback worker entry mirroring the backend-neutral wire contract."""

    token: Any
    role_trees: tuple[Any, ...]


class RefInterner:
    """Intern ``BlockId`` values once across every entry in one dispatch."""

    def __init__(self, *, max_refs: int | None = None) -> None:
        """Create an empty insertion-ordered block interner."""

        if max_refs is not None and max_refs < 0:
            raise ValueError("max_refs must be non-negative")
        self._slots: dict[Any, int] = {}
        self._block_ids: list[Any] = []
        self._max_refs = max_refs

    def __len__(self) -> int:
        """Return the number of unique blocks interned so far."""

        return len(self._block_ids)

    @property
    def block_ids(self) -> tuple[Any, ...]:
        """Return block IDs in the exact order expected by wire ref slots."""

        return tuple(self._block_ids)

    def intern(self, block: Any) -> int:
        """Return a stable slot for ``block``, adding it on first use."""

        existing = self._slots.get(block)
        if existing is not None:
            return existing
        if self._max_refs is not None and len(self._block_ids) >= self._max_refs:
            raise DispatchInvariantError("dispatch exceeds max_refs_per_dispatch")
        slot = len(self._block_ids)
        self._slots[block] = slot
        self._block_ids.append(block)
        return slot

    def bind(
        self,
        tree: StorageGather,
        *,
        max_depth: int | None = None,
        max_nodes: int | None = None,
        slot_factory: Callable[[int, int], Any] = LocalSlotTake,
        list_factory: Callable[[tuple[Any, ...]], Any] = LocalWireList,
    ) -> Any:
        """Convert one storage tree to a wire tree using this global interner."""

        work: list[tuple[StorageGather, int, bool]] = [(tree, 0, False)]
        built: list[Any] = []
        count = 0
        while work:
            node, depth, visited = work.pop()
            if visited:
                child_count = len(node.children)
                children = (
                    tuple(built[-child_count:]) if child_count else ()
                )
                if child_count:
                    del built[-child_count:]
                built.append(list_factory(children))
                continue
            count += 1
            if max_nodes is not None and count > max_nodes:
                raise DispatchInvariantError(
                    "gather exceeds max_gather_nodes_per_entry"
                )
            if max_depth is not None and depth > max_depth:
                raise DispatchInvariantError("gather exceeds max_gather_depth")
            if isinstance(node, BlockTake):
                built.append(slot_factory(self.intern(node.block), node.row))
                continue
            if not isinstance(node, StorageList):
                raise TypeError(
                    f"unsupported StorageGather node: {type(node)!r}"
                )
            work.append((node, depth, True))
            for child in reversed(node.children):
                work.append((child, depth + 1, False))
        if len(built) != 1:
            raise DispatchInvariantError("gather binding produced invalid stack state")
        return built[0]


@dataclass(frozen=True, slots=True)
class BatchSelection:
    """Identify one same-node group selected from the READY scheduler."""

    node: Any
    grains: tuple[Any, ...]

    def __post_init__(self) -> None:
        """Reject empty batches because they have no executable meaning."""

        if not self.grains:
            raise ValueError("BatchSelection.grains must not be empty")


@dataclass(frozen=True, slots=True)
class PreparedDispatch:
    """Hold central selectors and immutable worker entries before submission."""

    id: Any
    run: Any
    node: Any
    attempts: tuple[Any, ...]
    role_trees: tuple[tuple[StorageGather, ...], ...]
    block_ids: tuple[Any, ...]
    wire_entries: tuple[Any, ...]


class DispatchStores(Protocol):
    """Describe the narrow store surface needed by ``prepare_dispatch``."""

    grains: Any
    values: Any
    blocks: Any


@dataclass(frozen=True, slots=True)
class RuntimeDispatchId:
    """Provide a deterministic local dispatch ID when no model factory is used."""

    raw: bytes


def _dispatch_id(run: Any, node: Any, grains: tuple[Any, ...]) -> Any:
    """Allocate a fresh physical owner key independent of logical identity."""

    del run, node, grains
    new = getattr(_sem.DispatchId, "new", None)
    if new is not None:
        return new()
    return RuntimeDispatchId(secrets.token_bytes(16))


def _lookup_grain(grains: Any, grain_id: Any) -> Any:
    """Read a grain from common narrow-store method shapes."""

    for name in ("get", "require", "get_state"):
        method = getattr(grains, name, None)
        if method is None:
            continue
        record = method(grain_id)
        if record is not None:
            return record
    try:
        return grains[grain_id]
    except (KeyError, TypeError):
        pass
    records = getattr(grains, "_grains", None)
    if isinstance(records, dict) and grain_id in records:
        return records[grain_id]
    raise DispatchInvariantError(f"unknown READY grain: {grain_id!r}")


def _reserve_attempt(grains: Any, record: Any, run: Any, dispatch: Any) -> Any:
    """Reserve one current attempt through the model's narrow lifecycle API."""

    grain_id = getattr(getattr(record, "spec", record), "id", None)
    reserve = getattr(grains, "reserve", None)
    if reserve is not None:
        for args in (
            (grain_id, run),
            (grain_id,),
            (record, run),
            (record,),
        ):
            try:
                return reserve(*args)
            except TypeError:
                continue
    reserve = getattr(record, "reserve", None)
    if reserve is not None:
        for args in ((run, dispatch), (run,), (dispatch,), ()):
            try:
                return reserve(*args)
            except TypeError:
                continue
    active = getattr(record, "active_attempt", None)
    if active is not None:
        return active
    raise DispatchInvariantError("grain store does not expose attempt reservation")


def _storage_tree(values: Any, item: Any) -> StorageGather:
    """Resolve an item through the ValueNodeStore's narrow gather interface."""

    resolver = getattr(values, "resolve_storage_tree", None)
    if resolver is None:
        node_for_item = getattr(values, "node_for_item", None)
        get_node = getattr(values, "get", None)
        if node_for_item is None or get_node is None:
            raise DispatchInvariantError(
                "ValueNodeStore must expose gather resolution or node access"
            )
        return _value_node_storage_tree(values, node_for_item(item), set())
    tree = resolver(item)
    if not isinstance(tree, (BlockTake, StorageList)):
        block = getattr(tree, "block", None)
        row = getattr(tree, "row", None)
        children = getattr(tree, "children", None)
        if block is not None and isinstance(row, int):
            return BlockTake(block, row)
        if children is not None:
            return StorageList(tuple(_coerce_storage_tree(child) for child in children))
        raise DispatchInvariantError("ValueNodeStore returned an invalid gather tree")
    return tree


def resolve_storage_gather(values: Any, item: Any) -> StorageGather:
    """Resolve one PRESENT item to a backend-neutral storage gather."""

    return _storage_tree(values, item)


def _gather_node_count(tree: StorageGather) -> int:
    """Count every selector/list node in one storage gather iteratively."""

    count = 0
    stack = [tree]
    while stack:
        node = stack.pop()
        count += 1
        if isinstance(node, StorageList):
            stack.extend(node.children)
    return count


def _value_node_storage_tree(
    values: Any,
    node_id: Any,
    ancestors: set[Any],
) -> StorageGather:
    """Resolve a model ValueNode DAG to a storage tree without flattening lists."""

    if node_id in ancestors:
        raise DispatchInvariantError("ValueNode gather contains a cycle")
    value = values.get(node_id)
    if isinstance(value, _state.ScalarNode):
        return BlockTake(value.block, value.row)
    if isinstance(value, _state.FlatListNode):
        return StorageList(
            tuple(BlockTake(value.block, row) for row in range(value.start, value.stop))
        )
    if not isinstance(value, _state.CompositeListNode):
        raise DispatchInvariantError(f"unsupported ValueNode: {type(value)!r}")
    next_ancestors = set(ancestors)
    next_ancestors.add(node_id)
    return StorageList(
        tuple(
            _value_node_storage_tree(values, child, next_ancestors)
            for child in value.children
        )
    )


def _coerce_storage_tree(tree: Any) -> StorageGather:
    """Coerce structurally compatible model selectors to runtime selectors."""

    if isinstance(tree, (BlockTake, StorageList)):
        return tree
    block = getattr(tree, "block", None)
    row = getattr(tree, "row", None)
    if block is not None and isinstance(row, int):
        return BlockTake(block, row)
    children = getattr(tree, "children", None)
    if children is not None:
        return StorageList(tuple(_coerce_storage_tree(child) for child in children))
    raise DispatchInvariantError(f"invalid storage gather node: {type(tree)!r}")


def _wire_factories() -> tuple[Callable[..., Any], Callable[..., Any], type[Any]]:
    """Load backend wire DTOs lazily, falling back to pure local records."""

    try:
        from ..ray.protocol import SlotTake, WireList, WorkerEntry

        return SlotTake, WireList, WorkerEntry
    except (ImportError, AttributeError):
        return LocalSlotTake, LocalWireList, LocalWorkerEntry


def _batch_limits(limits: Any) -> tuple[int | None, int | None, int | None]:
    """Extract optional gather and ref limits from a runtime limits record."""

    if limits is None:
        return None, None, None
    return (
        getattr(limits, "max_gather_depth", None),
        getattr(limits, "max_gather_nodes_per_entry", None),
        getattr(limits, "max_refs_per_dispatch", None),
    )


def prepare_dispatch(
    selection: BatchSelection,
    stores: DispatchStores,
    *,
    run: Any,
    dispatch_id: Any | None = None,
    limits: Any | None = None,
) -> PreparedDispatch:
    """Reserve selected grains and bind all role trees with one ref interner."""

    dispatch_id = dispatch_id or _dispatch_id(run, selection.node, selection.grains)
    depth_limit, node_limit, ref_limit = _batch_limits(limits)
    interner = RefInterner(max_refs=ref_limit)
    slot_type, list_type, entry_type = _wire_factories()
    prepared_rows: list[tuple[Any, tuple[StorageGather, ...], tuple[Any, ...]]] = []
    storage_entries: list[tuple[StorageGather, ...]] = []

    for grain_id in selection.grains:
        record = _lookup_grain(stores.grains, grain_id)
        spec = getattr(record, "spec", record)
        if getattr(spec, "node", None) != selection.node:
            raise DispatchInvariantError("scheduler mixed grains from different nodes")
        phase = getattr(getattr(record, "phase", None), "value", None)
        if phase is not None and phase != "ready":
            raise DispatchInvariantError("selected grain is not READY")
        role_trees = tuple(
            _role_storage_tree(stores.values, binding)
            for binding in getattr(spec, "inputs", ())
        )
        entry_nodes = sum(_gather_node_count(tree) for tree in role_trees)
        if node_limit is not None and entry_nodes > node_limit:
            raise DispatchInvariantError(
                "entry roles exceed max_gather_nodes_per_entry"
            )
        wire_trees = tuple(
            interner.bind(
                tree,
                max_depth=depth_limit,
                max_nodes=None,
                slot_factory=lambda slot, row: slot_type(slot, row),
                list_factory=lambda children: list_type(children),
            )
            for tree in role_trees
        )
        prepared_rows.append((record, role_trees, wire_trees))
        storage_entries.append(role_trees)

    attempts: list[Any] = []
    wire_entries: list[Any] = []
    try:
        for record, _, _ in prepared_rows:
            attempts.append(
                _reserve_attempt(stores.grains, record, run, dispatch_id)
            )
        for token, (_, _, wire_trees) in zip(attempts, prepared_rows):
            try:
                wire_entries.append(
                    entry_type(token=token, role_trees=wire_trees)
                )
            except TypeError:
                wire_entries.append(entry_type(token, wire_trees))
    except BaseException:
        retry = getattr(stores.grains, "retry", None)
        if retry is not None:
            for token in reversed(attempts):
                try:
                    retry(token)
                except BaseException:
                    pass
        raise

    return PreparedDispatch(
        id=dispatch_id,
        run=run,
        node=selection.node,
        attempts=tuple(attempts),
        role_trees=tuple(storage_entries),
        block_ids=interner.block_ids,
        wire_entries=tuple(wire_entries),
    )


def _role_storage_tree(values: Any, binding: Any) -> StorageGather:
    """Resolve one ordered role binding without flattening structural values."""

    items = tuple(getattr(binding, "items", ()))
    if not items:
        return StorageList(())
    trees = tuple(_storage_tree(values, item) for item in items)
    return trees[0] if len(trees) == 1 else StorageList(trees)


@dataclass(frozen=True, slots=True)
class PreparedBlock:
    """Describe one block installation validated during transaction prepare."""

    id: Any
    ref: Any
    rows: int
    estimated_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class PreparedValueNode:
    """Describe one preallocated ValueNode installation."""

    id: Any
    value: Any


@dataclass(frozen=True, slots=True)
class PreparedItemBinding:
    """Describe one root-owned ItemRef to ValueNode binding."""

    root: Any
    item: Any
    node: Any


@dataclass(frozen=True, slots=True)
class PreparedOutcome:
    """Describe one attempt-gated terminal grain outcome."""

    grain: Any
    outcome: Any
    token: Any | None = None


@dataclass(frozen=True, slots=True)
class CommitDelta:
    """Contain the complete semantic change for one atomic MAP completion."""

    errors: tuple[Any, ...] = ()
    blocks: tuple[PreparedBlock, ...] = ()
    value_nodes: tuple[PreparedValueNode, ...] = ()
    item_bindings: tuple[PreparedItemBinding, ...] = ()
    controls: tuple[tuple[Any, ...], ...] = ()
    grain_outcomes: tuple[PreparedOutcome, ...] = ()
    receipts: tuple[Any, ...] = ()
    scope_events: tuple[Any, ...] = ()
    credit_delta: Any | None = None


class PeerAction(Enum):
    """Describe the semantic action for a peer in a failed MAP batch."""

    SEAL_FAILED = "seal_failed"
    RETRY = "retry"
    CANCEL = "cancel"


@dataclass(frozen=True, slots=True)
class FailureDecision:
    """Freeze bad-entry handling and any deterministic isolation split."""

    bad_entry_index: int | None
    error: Any | None
    peer_actions: tuple[PeerAction, ...]
    isolation_split: tuple[tuple[Any, ...], ...] = ()


class PendingDisposition(Enum):
    """Classify a dispatch completion before physical finalization."""

    ACCEPTED = "accepted"
    CONTROLLED_FAILURE = "controlled_failure"
    RETRY = "retry"
    STALE = "stale"
    CANCELLED = "cancelled"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class PendingDecision:
    """Freeze one pending dispatch's at-most-once semantic decision."""

    kind: PendingDisposition
    semantic_delta: CommitDelta | None = None
    failure: FailureDecision | None = None


@dataclass(frozen=True, slots=True)
class CommitResult:
    """Report whether a decision mutated state and which events were emitted."""

    decision: PendingDecision
    applied: bool
    events: tuple[Any, ...] = ()


class ManifestDeltaBuilder:
    """Translate one successful worker manifest into an atomic semantic delta."""

    def __init__(self, graph: Any, stores: Any) -> None:
        """Bind immutable graph facts and one run's narrow state stores."""

        self.graph = graph
        self.stores = stores

    def __call__(
        self,
        pending: Any,
        manifest: Any,
        output_refs: tuple[Any, ...],
    ) -> CommitDelta:
        """Build blocks, value nodes, controls, outcomes, and receipts together."""

        from ..model.graph import StructuralListShape
        from ..ray.protocol import SuccessManifest

        if not isinstance(manifest, SuccessManifest):
            raise DispatchInvariantError(
                "ManifestDeltaBuilder accepts only SuccessManifest"
            )
        prepared = getattr(pending, "prepared", pending)
        node = self.graph.node(prepared.node)
        layouts = tuple(manifest.outputs)
        if len(layouts) != len(output_refs):
            raise DispatchInvariantError(
                "success output refs do not match manifest layouts"
            )
        expected_ports = tuple(output.id for output in node.outputs)
        if tuple(layout.port for layout in layouts) != expected_ports:
            raise DispatchInvariantError(
                "manifest output ports do not match compiled MAP outputs"
            )

        blocks = self.stores.blocks
        values = self.stores.values
        block_ids = blocks.preview_ids(len(layouts))
        prepared_blocks = tuple(
            PreparedBlock(
                id=block_id,
                ref=ref,
                rows=layout.row_count,
                estimated_bytes=layout.estimated_bytes,
            )
            for block_id, ref, layout in zip(block_ids, output_refs, layouts)
        )

        attempts = tuple(prepared.attempts)
        node_ids = values.preview_ids(len(attempts) * len(layouts))
        value_nodes: list[PreparedValueNode] = []
        item_bindings: list[PreparedItemBinding] = []
        controls: list[tuple[Any, ...]] = []
        receipts: list[Any] = []
        outcomes: list[PreparedOutcome] = []
        next_node = 0
        for row, token in enumerate(attempts):
            state = self.stores.grains.require(token.grain)
            spec = state.spec
            output_items = tuple(spec.output_slots)
            if len(output_items) != len(layouts):
                raise DispatchInvariantError(
                    "grain output slots do not match manifest layouts"
                )
            for slot, (layout, block_id, item) in enumerate(
                zip(layouts, block_ids, output_items)
            ):
                if item.port != layout.port:
                    raise DispatchInvariantError(
                        "grain output coordinate targets another port"
                    )
                if isinstance(layout.shape, StructuralListShape):
                    assert layout.offsets is not None
                    value = _state.FlatListNode(
                        block_id,
                        layout.offsets[row],
                        layout.offsets[row + 1],
                    )
                else:
                    value = _state.ScalarNode(block_id, row)
                value_id = node_ids[next_node]
                next_node += 1
                value_nodes.append(PreparedValueNode(value_id, value))
                item_bindings.append(
                    PreparedItemBinding(spec.root, item, value_id)
                )
                if layout.control_bits is not None:
                    controls.append(
                        (
                            item,
                            _bit_at(layout.control_bits, row),
                            self._control_consumers(layout.port),
                        )
                    )
                receipts.append(
                    _sem.Receipt(
                        item=item,
                        context=spec.context,
                        state=_sem.ReceiptState.PRESENT,
                        producer=spec.id,
                    )
                )
            outcomes.append(
                PreparedOutcome(spec.id, _sem.Success(), token=token)
            )
        return CommitDelta(
            blocks=prepared_blocks,
            value_nodes=tuple(value_nodes),
            item_bindings=tuple(item_bindings),
            controls=tuple(controls),
            grain_outcomes=tuple(outcomes),
            receipts=tuple(receipts),
        )

    def _control_consumers(self, port: Any) -> int:
        """Return the positive static FILTER consumer count for a bool port."""

        from ..model.graph import FilterOp

        count = sum(
            isinstance(self.graph.node(node_id).op, FilterOp)
            and self.graph.node(node_id).op.mask == port
            for node_id in self.graph.consumers_by_port.get(port, ())
        )
        if count <= 0:
            raise DispatchInvariantError(
                "manifest emitted a control bit without a FILTER consumer"
            )
        return count


def _bit_at(bits: bytes, index: int) -> bool:
    """Read one little-endian packed control bit."""

    return bool(bits[index // 8] & (1 << (index % 8)))


class _UndoJournal:
    """Record delta-sized inverse operations in application order."""

    def __init__(self) -> None:
        """Create an empty transaction-local inverse stack."""

        self._entries: list[Callable[[], None]] = []

    def add(self, undo: Callable[[], None]) -> None:
        """Append one inverse that is safe before or after its mutation."""

        self._entries.append(undo)

    def rollback(self) -> None:
        """Run every inverse in reverse order and surface the first failure."""

        first_error: BaseException | None = None
        for undo in reversed(self._entries):
            try:
                undo()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        self._entries.clear()
        if first_error is not None:
            raise DispatchInvariantError(
                f"transaction rollback failed: {first_error}"
            ) from first_error

    def clear(self) -> None:
        """Discard inverses after a successful atomic commit."""

        self._entries.clear()


class LocalTransaction:
    """Apply prepared semantic deltas atomically on the coordinator thread."""

    def __init__(
        self,
        stores: Any,
        *,
        event_sink: Any | None = None,
        delta_builder: Callable[[Any, Any, tuple[Any, ...]], CommitDelta] | None = None,
    ) -> None:
        """Bind narrow semantic stores and an append-only downstream event sink."""

        self.stores = stores
        self.event_sink = event_sink
        self.delta_builder = delta_builder
        self._applying = False

    def prepare(
        self,
        pending: Any,
        manifest: Any,
        output_refs: Iterable[Any] = (),
    ) -> PendingDecision:
        """Validate a completion without mutating runtime semantic state."""

        existing = getattr(pending, "decision", None)
        if existing is not None:
            return existing
        self._validate_header(pending, manifest)
        refs = tuple(output_refs)
        builder = self.delta_builder
        if builder is None:
            delta = getattr(manifest, "semantic_delta", None)
            if delta is None and isinstance(manifest, CommitDelta):
                delta = manifest
            if delta is None:
                raise DispatchInvariantError(
                    "transaction needs a delta_builder or manifest.semantic_delta"
                )
        else:
            delta = builder(pending, manifest, refs)
        if not isinstance(delta, CommitDelta):
            raise DispatchInvariantError("delta builder returned a non-CommitDelta")
        self._preflight(delta)
        return PendingDecision(PendingDisposition.ACCEPTED, semantic_delta=delta)

    def apply(self, delta: CommitDelta) -> tuple[Any, ...]:
        """Install a preflighted delta or restore all touched stores on failure."""

        if self._applying:
            raise DispatchInvariantError("LocalTransaction.apply is not reentrant")
        self._preflight(delta)
        events = tuple(delta.receipts) + tuple(delta.scope_events)
        reserved_by_sink, reservations = self._reserve_events(events)
        journal = _UndoJournal()
        self._applying = True
        try:
            self._install_errors(delta.errors, journal)
            self._install_blocks(delta.blocks, journal)
            self._install_value_nodes(delta.value_nodes, journal)
            self._install_item_bindings(delta.item_bindings, journal)
            self._install_controls(delta.controls, delta.receipts, journal)
            self._seal_outcomes(delta.grain_outcomes, journal)
            self._publish_receipts(delta.receipts, journal)
            self._apply_credit_delta(delta.credit_delta, journal)
            self._append_events(
                events,
                reservations=reservations if reserved_by_sink else None,
            )
            journal.clear()
            return events
        except BaseException as apply_error:
            rollback_error: BaseException | None = None
            try:
                journal.rollback()
            except BaseException as error:
                rollback_error = error
            try:
                if reserved_by_sink:
                    self._release_reserved_events(reservations)
            except BaseException as error:
                if rollback_error is None:
                    rollback_error = error
            if rollback_error is not None:
                raise DispatchInvariantError(
                    f"semantic apply failed ({apply_error}) and rollback was incomplete"
                ) from rollback_error
            raise
        finally:
            self._applying = False

    def apply_disposition(
        self,
        pending: Any,
        decision: PendingDecision,
    ) -> CommitResult:
        """Freeze and apply one pending decision exactly once."""

        frozen = getattr(pending, "decision", None)
        if frozen is not None and frozen != decision:
            raise DispatchInvariantError("pending dispatch already has another decision")
        if frozen is None:
            setattr(pending, "decision", decision)
        if getattr(pending, "semantic_applied", False):
            return CommitResult(decision=decision, applied=False)

        events: tuple[Any, ...] = ()
        if decision.kind is PendingDisposition.ACCEPTED:
            if decision.semantic_delta is None:
                raise DispatchInvariantError("ACCEPTED decision has no semantic delta")
            events = self.apply(decision.semantic_delta)
        elif decision.kind is PendingDisposition.CONTROLLED_FAILURE:
            events = self._apply_failure(pending, decision.failure)
        elif decision.kind is PendingDisposition.RETRY:
            self._retry_attempts(pending)
        elif decision.kind in {
            PendingDisposition.CANCELLED,
            PendingDisposition.ABORTED,
        }:
            self._cancel_attempts(pending)
        elif decision.kind is not PendingDisposition.STALE:
            raise DispatchInvariantError(f"unsupported disposition: {decision.kind}")

        setattr(pending, "semantic_applied", True)
        return CommitResult(decision=decision, applied=True, events=events)

    def rollback(self, journal: _UndoJournal) -> None:
        """Apply a transaction-local undo journal for fault-injection callers."""

        if not isinstance(journal, _UndoJournal):
            raise TypeError("rollback requires a LocalTransaction undo journal")
        journal.rollback()

    def _validate_header(self, pending: Any, manifest: Any) -> None:
        """Validate immutable manifest fields against the pending dispatch."""

        prepared = getattr(pending, "prepared", pending)
        header = getattr(manifest, "header", manifest)
        for name in ("run", "dispatch", "node"):
            expected_name = "id" if name == "dispatch" else name
            expected = getattr(prepared, expected_name, None)
            actual = getattr(header, name, None)
            if expected is not None and actual is not None and actual != expected:
                raise DispatchInvariantError(f"manifest {name} does not match dispatch")
        expected_attempts = tuple(getattr(prepared, "attempts", ()))
        actual_attempts = getattr(header, "attempts", None)
        if actual_attempts is not None and tuple(actual_attempts) != expected_attempts:
            raise DispatchInvariantError("manifest attempts do not match dispatch")
        expected_lease = getattr(pending, "lease", None)
        actual_lease = getattr(header, "lease", None)
        if (
            expected_lease is not None
            and actual_lease is not None
            and expected_lease != actual_lease
        ):
            raise DispatchInvariantError("manifest actor lease does not match pending")
        grains = self._store("grains")
        if grains is not None:
            current = getattr(grains, "is_current", None)
            if current is not None:
                states = tuple(bool(current(token)) for token in expected_attempts)
                if any(states) and not all(states):
                    raise DispatchInvariantError("mixed current and stale attempts")
                if states and not any(states):
                    raise DispatchInvariantError("manifest attempts are stale")

    def _preflight(self, delta: CommitDelta) -> None:
        """Reject conflicts before the first store is mutated."""

        self._preflight_unique(delta.errors, "id", "error")
        self._preflight_unique(delta.blocks, "id", "block")
        self._preflight_unique(delta.value_nodes, "id", "value node")
        self._preflight_unique(delta.item_bindings, "item", "item binding")
        self._preflight_unique(delta.grain_outcomes, "grain", "grain outcome")
        self._preflight_receipts(delta.receipts)
        self._preflight_controls(delta.controls)
        for binding in delta.item_bindings:
            existing = self._existing_item_node(binding.item)
            if existing is not None and existing != binding.node:
                raise DispatchInvariantError("conflicting item binding")
        for prepared in delta.grain_outcomes:
            grains = self._store("grains")
            current = getattr(grains, "is_current", None) if grains is not None else None
            if prepared.token is not None and current is not None:
                if not current(prepared.token):
                    raise DispatchInvariantError("outcome attempt is not current")
        if delta.credit_delta is not None:
            credits = self._store("credits")
            if credits is None or not any(
                callable(getattr(credits, name, None))
                for name in ("undo_delta", "revert_delta")
            ):
                raise DispatchInvariantError(
                    "credit delta requires an explicit inverse operation"
                )

    def _reserve_events(
        self,
        events: tuple[Any, ...],
    ) -> tuple[bool, tuple[Any, ...]]:
        """Reserve all sink credits before any semantic mutation begins."""

        if not events or self.event_sink is None:
            return False, ()
        reserve = getattr(self.event_sink, "reserve_events", None)
        if reserve is None:
            return False, ()
        reservations = reserve(events)
        if reservations is None:
            raise DispatchInvariantError(
                "event sink cannot reserve the complete transaction batch"
            )
        return True, tuple(reservations)

    def _release_reserved_events(self, reservations: tuple[Any, ...]) -> None:
        """Release event credits after a transaction rollback."""

        release = getattr(self.event_sink, "release_reserved_events", None)
        if release is None:
            raise DispatchInvariantError(
                "event sink reserved credits but cannot release them"
            )
        release(reservations)

    def _preflight_unique(
        self,
        values: Iterable[Any],
        attribute: str,
        label: str,
    ) -> None:
        """Require duplicate keys inside one delta to be exact duplicates."""

        seen: dict[Any, Any] = {}
        for value in values:
            key = getattr(value, attribute)
            previous = seen.get(key)
            if previous is not None and previous != value:
                raise DispatchInvariantError(f"conflicting duplicate {label}")
            seen[key] = value

    def _preflight_receipts(self, receipts: Iterable[Any]) -> None:
        """Require receipt publication to be new or exactly idempotent."""

        store = self._store("receipts")
        if store is None:
            return
        getter = getattr(store, "get", None)
        if getter is None:
            return
        for receipt in receipts:
            existing = getter(receipt.item)
            if existing is not None and existing != receipt:
                raise DispatchInvariantError("conflicting receipt publication")

    def _preflight_controls(self, controls: Iterable[tuple[Any, ...]]) -> None:
        """Require duplicate FILTER controls to preserve the exact bool."""

        store = self._store("controls")
        if store is None:
            return
        values = getattr(store, "_bools", {})
        for control in controls:
            if len(control) not in {2, 3}:
                raise DispatchInvariantError("invalid control installation tuple")
            item, value = control[:2]
            if type(value) is not bool:
                raise DispatchInvariantError("control value must have exact bool type")
            if len(control) == 3 and (
                isinstance(control[2], bool)
                or not isinstance(control[2], int)
                or control[2] <= 0
            ):
                raise DispatchInvariantError(
                    "control consumer count must be a positive int"
                )
            existing = values.get(item)
            existing_value = getattr(existing, "value", existing)
            if existing is not None and existing_value is not value:
                raise DispatchInvariantError("conflicting control bool")

    def _install_errors(
        self,
        errors: Iterable[Any],
        journal: _UndoJournal,
    ) -> None:
        """Install prepared semantic errors before failed outcomes reference them."""

        store = self._store("errors")
        values = tuple(errors)
        if store is None and values:
            raise DispatchInvariantError("transaction has errors but no ErrorStore")
        for error in values:
            mapping = getattr(store, "_errors", {})
            if error.id not in mapping:
                journal.add(
                    lambda store=store, error=error: self._undo_indexed_insert(
                        store,
                        "_errors",
                        error.id,
                        "_by_root",
                        error.root,
                    )
                )
            store.put(error)

    def _install_blocks(
        self,
        blocks: Iterable[PreparedBlock],
        journal: _UndoJournal,
    ) -> None:
        """Install output blocks before any ValueNode can reference them."""

        store = self._store("blocks")
        if store is None and tuple(blocks):
            raise DispatchInvariantError("transaction has blocks but no BlockStore")
        for block in blocks:
            existed = block.id in store
            previous_next = getattr(store, "_next_id", None)
            if not existed:
                journal.add(
                    lambda store=store,
                    block_id=block.id,
                    previous_next=previous_next: self._undo_block_install(
                        store, block_id, previous_next
                    )
                )
            install_prepared = getattr(store, "install_prepared", None)
            if install_prepared is not None:
                install_prepared(block)
                continue
            install = getattr(store, "install")
            installed = install(block.ref, block.rows, block.estimated_bytes)
            if installed != block.id:
                raise DispatchInvariantError("BlockStore returned another prepared ID")

    def _install_value_nodes(
        self,
        nodes: Iterable[PreparedValueNode],
        journal: _UndoJournal,
    ) -> None:
        """Install preallocated ValueNodes after their blocks exist."""

        store = self._store("values")
        if store is None and tuple(nodes):
            raise DispatchInvariantError("transaction has values but no ValueNodeStore")
        for node in nodes:
            records = getattr(store, "_nodes", {})
            existed = node.id in records
            previous_next = getattr(store, "_next_id", None)
            if not existed:
                journal.add(
                    lambda store=store,
                    node_id=node.id,
                    previous_next=previous_next: self._undo_value_install(
                        store, node_id, previous_next
                    )
                )
            install = getattr(store, "install_prepared", None)
            if install is None:
                install = getattr(store, "install_node", None)
            if install is None:
                installed = self._create_value_node(store, node.value)
                if installed != node.id:
                    raise DispatchInvariantError(
                        "ValueNodeStore returned another prepared ID"
                    )
            else:
                install(node)

    def _create_value_node(self, store: Any, value: Any) -> Any:
        """Create one model ValueNode through its public narrow constructors."""

        if isinstance(value, _state.ScalarNode):
            return store.create_scalar(value.block, value.row)
        if isinstance(value, _state.FlatListNode):
            return store.create_flat_list(value.block, value.start, value.stop)
        if isinstance(value, _state.CompositeListNode):
            return store.create_composite(value.children)
        raise DispatchInvariantError(f"unsupported prepared ValueNode: {type(value)!r}")

    def _install_item_bindings(
        self,
        bindings: Iterable[PreparedItemBinding],
        journal: _UndoJournal,
    ) -> None:
        """Bind output coordinates only after all referenced nodes exist."""

        store = self._store("values")
        for binding in bindings:
            existing = self._existing_item_node(binding.item)
            if existing == binding.node:
                continue
            journal.add(
                lambda store=store,
                item=binding.item,
                node=binding.node: self._undo_item_binding(
                    store, item, node
                )
            )
            store.bind_item(binding.root, binding.item, binding.node)

    def _install_controls(
        self,
        controls: Iterable[tuple[Any, ...]],
        receipts: Iterable[Any],
        journal: _UndoJournal,
    ) -> None:
        """Install FILTER control projections before PRESENT receipts publish."""

        store = self._store("controls")
        roots = {receipt.item: receipt.context.root for receipt in receipts}
        for control in controls:
            item, value = control[:2]
            remaining_consumers = control[2] if len(control) == 3 else 1
            root = roots.get(item)
            if root is None:
                raise DispatchInvariantError("control bool has no matching receipt root")
            existing = getattr(store, "_bools", {}).get(item)
            if existing is None:
                journal.add(
                    lambda store=store,
                    item=item,
                    root=root: self._undo_indexed_insert(
                        store, "_bools", item, "_by_root", root
                    )
                )
            store.install_bool(root, item, value, remaining_consumers)

    def _seal_outcomes(
        self,
        outcomes: Iterable[PreparedOutcome],
        journal: _UndoJournal,
    ) -> None:
        """Seal each grain with its exact current attempt token."""

        grains = self._store("grains")
        for prepared in outcomes:
            state = grains.require(prepared.grain)
            previous = (
                state.phase,
                state.generation,
                state.active_attempt,
                state.outcome,
            )
            journal.add(
                lambda state=state, previous=previous: self._restore_grain_state(
                    state, previous
                )
            )
            seal = getattr(grains, "seal")
            authority = (
                prepared.token if prepared.token is not None else prepared.grain
            )
            for args, kwargs in (
                ((authority, prepared.outcome), {}),
                ((prepared.grain, prepared.outcome), {"token": prepared.token}),
            ):
                try:
                    seal(*args, **kwargs)
                    break
                except TypeError:
                    continue
            else:
                raise DispatchInvariantError("GrainStore.seal signature unsupported")

    def _publish_receipts(
        self,
        receipts: Iterable[Any],
        journal: _UndoJournal,
    ) -> None:
        """Publish all output receipts after values and outcomes are visible."""

        store = self._store("receipts")
        for receipt in receipts:
            existing = store.get(receipt.item)
            if existing is None:
                journal.add(
                    lambda store=store,
                    receipt=receipt: self._undo_indexed_insert(
                        store,
                        "_receipts",
                        receipt.item,
                        "_by_root",
                        receipt.context.root,
                    )
                )
            store.publish_once(receipt)

    def _apply_credit_delta(
        self,
        delta: Any | None,
        journal: _UndoJournal,
    ) -> None:
        """Apply an optional owner-keyed credit delta last."""

        if delta is None:
            return
        credits = self._store("credits")
        apply_delta = getattr(credits, "apply_delta", None)
        if apply_delta is None:
            raise DispatchInvariantError("CreditManager cannot apply credit delta")
        undo = getattr(credits, "undo_delta", None)
        if undo is None:
            undo = getattr(credits, "revert_delta")
        journal.add(lambda undo=undo, delta=delta: undo(delta))
        apply_delta(delta)

    def _undo_indexed_insert(
        self,
        store: Any,
        mapping_name: str,
        key: Any,
        index_name: str,
        index_key: Any,
    ) -> None:
        """Remove one inserted key and only its matching root-index membership."""

        mapping = getattr(store, mapping_name)
        if key not in mapping:
            return
        mapping.pop(key, None)
        index = getattr(store, index_name)
        members = index.get(index_key)
        if members is None:
            return
        members.discard(key)
        if not members:
            index.pop(index_key, None)

    def _undo_block_install(
        self,
        store: Any,
        block: Any,
        previous_next: int | None,
    ) -> None:
        """Discard one newly installed unowned block and restore its allocator."""

        if block in store:
            store.discard_unleased(block)
        if previous_next is not None:
            store._next_id = previous_next

    def _undo_value_install(
        self,
        store: Any,
        node: Any,
        previous_next: int | None,
    ) -> None:
        """Discard one newly installed unbound value node and its retained edges."""

        records = getattr(store, "_nodes", {})
        if node in records:
            store.discard_unbound(node)
        if previous_next is not None:
            store._next_id = previous_next

    def _undo_item_binding(self, store: Any, item: Any, node: Any) -> None:
        """Unbind only the item installed by the current semantic delta."""

        items = getattr(store, "_items", {})
        if items.get(item) == node:
            store.unbind_item(item)

    def _restore_grain_state(
        self,
        state: Any,
        previous: tuple[Any, int, Any, Any],
    ) -> None:
        """Restore the four mutable fields of one touched grain record."""

        (
            state.phase,
            state.generation,
            state.active_attempt,
            state.outcome,
        ) = previous

    def _append_events(
        self,
        events: tuple[Any, ...],
        *,
        reservations: tuple[Any, ...] | None = None,
    ) -> None:
        """Append downstream events without synchronously invoking handlers."""

        if not events or self.event_sink is None:
            return
        if reservations is not None:
            commit = getattr(self.event_sink, "commit_reserved_events", None)
            if commit is None:
                raise DispatchInvariantError(
                    "event sink reserved credits but cannot commit them"
                )
            commit(reservations, events)
            return
        reserve = getattr(self.event_sink, "reserve_events", None)
        if reserve is not None:
            reserved = reserve(events)
            if reserved is None:
                raise DispatchInvariantError(
                    "event sink cannot reserve the complete event batch"
                )
            self.event_sink.commit_reserved_events(tuple(reserved), events)
            return
        extend = getattr(self.event_sink, "extend", None)
        if extend is not None:
            extend(events)
            return
        enqueue = getattr(self.event_sink, "enqueue", None)
        if enqueue is not None:
            for event in events:
                enqueue(event)
            return
        append = getattr(self.event_sink, "append", None)
        if append is None:
            raise DispatchInvariantError("event sink has no append interface")
        for event in events:
            append(event)

    def _apply_failure(
        self,
        pending: Any,
        failure: FailureDecision | None,
    ) -> tuple[Any, ...]:
        """Apply a frozen controlled-failure plan through grain lifecycle APIs."""

        if failure is None:
            raise DispatchInvariantError("controlled failure has no FailureDecision")
        prepared = getattr(pending, "prepared", pending)
        grains = self._store("grains")
        events: list[Any] = []
        for index, (token, action) in enumerate(
            zip(prepared.attempts, failure.peer_actions)
        ):
            if action is PeerAction.RETRY:
                self._retry_token(grains, token)
            elif action is PeerAction.CANCEL:
                self._cancel_token(grains, token)
            elif action is PeerAction.SEAL_FAILED:
                seal = getattr(grains, "seal_failed", None)
                if seal is None:
                    raise DispatchInvariantError(
                        "GrainStore lacks controlled failure sealing"
                    )
                produced = seal(token, failure.error, index=index)
                if produced is not None:
                    events.extend(produced)
        self._append_events(tuple(events))
        return tuple(events)

    def _retry_attempts(self, pending: Any) -> None:
        """Return every current attempt in a pending dispatch to READY."""

        grains = self._store("grains")
        prepared = getattr(pending, "prepared", pending)
        for token in getattr(prepared, "attempts", ()):
            self._retry_token(grains, token)

    def _cancel_attempts(self, pending: Any) -> None:
        """Cancel every current attempt in a pending dispatch idempotently."""

        grains = self._store("grains")
        prepared = getattr(pending, "prepared", pending)
        for token in getattr(prepared, "attempts", ()):
            self._cancel_token(grains, token)

    def _retry_token(self, grains: Any, token: Any) -> None:
        """Call the model retry API for one exact attempt token."""

        for name in ("retry", "retry_infrastructure_failure"):
            method = getattr(grains, name, None)
            if method is not None:
                method(token)
                return
        raise DispatchInvariantError("GrainStore has no retry operation")

    def _cancel_token(self, grains: Any, token: Any) -> None:
        """Call the model cancel/release API for one exact attempt token."""

        for name in ("cancel", "release_for_reexecution", "retry"):
            method = getattr(grains, name, None)
            if method is not None:
                method(token)
                return
        raise DispatchInvariantError("GrainStore has no cancel operation")

    def _existing_item_node(self, item: Any) -> Any | None:
        """Read an existing ItemRef binding through the narrow state surface."""

        store = self._store("values")
        if store is None:
            return None
        for name in ("node_for_item", "get_item", "item_node"):
            method = getattr(store, name, None)
            if method is None:
                continue
            try:
                return method(item)
            except KeyError:
                return None
        items = getattr(store, "_items", None)
        if isinstance(items, dict):
            return items.get(item)
        return None

    def _store(self, name: str) -> Any | None:
        """Resolve a named store from an aggregate or mapping."""

        if isinstance(self.stores, dict):
            return self.stores.get(name)
        return getattr(self.stores, name, None)

