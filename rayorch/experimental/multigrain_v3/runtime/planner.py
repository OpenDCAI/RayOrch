"""Incremental semantic planning for Multigrain V3.

The planner is intentionally single-threaded.  Receipt publication only
appends local events; handlers run later from :meth:`EventPlanner.drain`, which
prevents synchronous re-entry and gives every coordinator turn a hard budget.
"""

from __future__ import annotations

import hashlib
import inspect
import time
from collections import deque
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Mapping

from ..model import semantics as _sem
from ..model import state as _state
from .dispatch import BatchSelection


ClosureView = _state.ClosureView
ControlStore = _state.ControlStore


class PlannerInvariantError(RuntimeError):
    """Report a semantic conflict that requires aborting the current run."""


class CreditCapacityError(RuntimeError):
    """Report a reservation that can never fit the configured hard limits."""


class GrainBackpressure(RuntimeError):
    """Signal temporary exhaustion of the executable-grain hard ledger."""


class ReceiptStore:
    """Store exactly one immutable terminal receipt for each output coordinate."""

    def __init__(self, values: Any | None = None) -> None:
        """Create an empty root-indexed receipt store."""

        self._receipts: dict[Any, Any] = {}
        self._by_root: dict[Any, set[Any]] = {}
        self._values = values

    def __len__(self) -> int:
        """Return the number of live coordinate receipts."""

        return len(self._receipts)

    def publish_once(self, receipt: Any) -> bool:
        """Publish a receipt once, treating an exact duplicate as idempotent."""

        existing = self._receipts.get(receipt.item)
        if existing is not None:
            if existing != receipt:
                raise PlannerInvariantError(
                    f"conflicting receipt for coordinate {receipt.item!r}"
                )
            return False
        self._validate_value_presence(receipt)
        root = receipt.context.root
        self._receipts[receipt.item] = receipt
        self._by_root.setdefault(root, set()).add(receipt.item)
        return True

    def get(self, item: Any) -> Any | None:
        """Return the terminal receipt for ``item`` when it has been published."""

        return self._receipts.get(item)

    def require(self, item: Any) -> Any:
        """Return a receipt or raise for an unknown coordinate."""

        try:
            return self._receipts[item]
        except KeyError as error:
            raise KeyError(f"receipt is not terminal: {item!r}") from error

    def remove_root(self, root: Any) -> None:
        """Remove only receipts owned by one reclaimed root."""

        for item in self._by_root.pop(root, ()):
            self._receipts.pop(item, None)

    def items_for_root(self, root: Any) -> tuple[Any, ...]:
        """Return one root's live ItemRefs without scanning unrelated history."""

        return tuple(self._by_root.get(root, ()))

    def _validate_value_presence(self, receipt: Any) -> None:
        """Cross-check PRESENT state against an optional ValueNodeStore."""

        if self._values is None:
            return
        has_binding = _item_node(self._values, receipt.item) is not None
        state = _state_name(receipt.state)
        if state == "present" and not has_binding:
            raise PlannerInvariantError(
                "PRESENT receipt requires an installed ValueNode binding"
            )
        if state != "present" and has_binding:
            raise PlannerInvariantError(
                "non-PRESENT receipt cannot retain an ItemRef binding"
            )


@dataclass(slots=True)
class InputLatch:
    """Collect one aligned terminal receipt for every compiled input role."""

    node: Any
    context: Any
    expected_roles: tuple[str, ...]
    expected_ports: tuple[Any, ...]
    receipts: list[Any | None]
    settled: int = 0

    @property
    def ready(self) -> bool:
        """Return whether every expected role has settled."""

        return self.settled == len(self.receipts)

    def install(self, index: int, receipt: Any) -> bool:
        """Install one role receipt idempotently and reject conflicts."""

        if index < 0 or index >= len(self.receipts):
            raise PlannerInvariantError("latch role index is out of range")
        if receipt.item.port != self.expected_ports[index]:
            raise PlannerInvariantError("receipt port does not match latch role")
        existing = self.receipts[index]
        if existing is not None:
            if existing != receipt:
                raise PlannerInvariantError("conflicting latch receipt")
            return False
        if receipt.context != self.context:
            raise PlannerInvariantError("aligned role contexts do not match")
        self.receipts[index] = receipt
        self.settled += 1
        return True


@dataclass(frozen=True, slots=True)
class ReceiptPublished:
    """Queue one newly published receipt for downstream routing."""

    receipt: Any


@dataclass(frozen=True, slots=True)
class ClosureReady:
    """Queue a fully settled REDUCE closure for bounded finalization."""

    closure: Any


@dataclass(frozen=True, slots=True)
class ExpansionCreditGranted:
    """Queue another bounded materialization chunk for an open EXPAND."""

    scope: Any


@dataclass(slots=True)
class PlannerGrainRecord:
    """Provide a minimal local GrainStore record for focused planner tests."""

    spec: Any
    phase: str
    outcome: Any | None = None
    generation: int = 0
    active_attempt: Any | None = None


class ScopeTracker:
    """Own dynamic scope instances and independent per-REDUCE closure views."""

    def __init__(self, *, closure_factory: Callable[..., Any] | None = None) -> None:
        """Create empty scope and closure indexes."""

        self.instances: dict[Any, Any] = {}
        self.closures: dict[tuple[Any, Any], Any] = {}
        self._by_root: dict[Any, set[Any]] = {}
        self._closure_factory = closure_factory

    def create_pending(
        self,
        scope: Any,
        definition: Any,
        root: Any,
        parent_item: Any,
        parent_context: Any,
        origin_grain: Any,
    ) -> Any:
        """Create an idempotent PENDING_OPEN scope before fanout is known."""

        expected = _construct(
            _state.ScopeInstance,
            id=scope,
            definition=definition,
            root=root,
            parent_item=parent_item,
            parent_context=parent_context,
            expected=None,
            state=_enum_member(_state.ScopeState, "PENDING_OPEN"),
            origin_grain=origin_grain,
            failure=None,
            structural_lease=None,
        )
        existing = self.instances.get(scope)
        if existing is not None:
            static_existing = (
                existing.definition,
                existing.root,
                existing.parent_item,
                existing.parent_context,
                existing.origin_grain,
            )
            static_expected = (
                definition,
                root,
                parent_item,
                parent_context,
                origin_grain,
            )
            if static_existing != static_expected:
                raise PlannerInvariantError("conflicting ScopeInstance creation")
            return existing
        self.instances[scope] = expected
        self._by_root.setdefault(root, set()).add(scope)
        return expected

    def open(self, scope: Any, expected: int, lease: Any) -> Any:
        """Transition a pending scope to OPEN with a known cardinality."""

        if expected < 0:
            raise PlannerInvariantError("scope expected cardinality is negative")
        instance = self.require(scope)
        lease_id = getattr(lease, "id", lease)
        state = _state_name(instance.state)
        if state == "open":
            if (
                instance.expected != expected
                or instance.structural_lease != lease_id
            ):
                raise PlannerInvariantError(
                    "scope reopened with another cardinality or lease"
                )
            return _construct(_state.ScopeOpened, scope=scope, expected=expected)
        if state != "pending_open":
            raise PlannerInvariantError(f"cannot open scope in state {state!r}")
        instance.expected = expected
        instance.state = _enum_member(_state.ScopeState, "OPEN")
        instance.structural_lease = lease_id
        return _construct(_state.ScopeOpened, scope=scope, expected=expected)

    def mark_absent(self, scope: Any, origin: Any) -> Any:
        """Close a pending scope whose EXPAND input is normally absent."""

        instance = self.require(scope)
        state = _state_name(instance.state)
        event = _construct(_state.ScopeAbsent, scope=scope, origin=origin)
        if state == "absent":
            return event
        if state != "pending_open":
            raise PlannerInvariantError(f"cannot mark {state!r} scope absent")
        instance.state = _enum_member(_state.ScopeState, "ABSENT")
        return event

    def mark_failed(self, scope: Any, cause: Any) -> Any:
        """Close a pending scope with the EXPAND grain as direct cause."""

        instance = self.require(scope)
        state = _state_name(instance.state)
        event = _construct(_state.ScopeFailed, scope=scope, cause=cause)
        if state == "failed":
            if instance.failure != cause:
                raise PlannerInvariantError("scope failed with another cause")
            return event
        if state != "pending_open":
            raise PlannerInvariantError(f"cannot fail scope in state {state!r}")
        instance.state = _enum_member(_state.ScopeState, "FAILED")
        instance.failure = cause
        return event

    def ensure_closure(
        self,
        reduce_node: Any,
        scope: Any,
        input_port: Any | None = None,
    ) -> Any:
        """Create one known-N closure without affecting sibling reducers."""

        key = (reduce_node, scope)
        existing = self.closures.get(key)
        if existing is not None:
            if input_port is not None and existing.input_port != input_port:
                raise PlannerInvariantError("closure input port changed")
            return existing
        instance = self.require(scope)
        if _state_name(instance.state) != "open" or instance.expected is None:
            raise PlannerInvariantError("closure requires an OPEN known-N scope")
        if input_port is None:
            raise PlannerInvariantError("new closure requires its input port")
        closure = self._new_closure(
            reduce_node=reduce_node,
            input_port=input_port,
            scope=scope,
            expected=instance.expected,
        )
        self.closures[key] = closure
        return closure

    def settle(self, reduce_node: Any, receipt: Any) -> bool:
        """Settle the top-scope ordinal for one reducer and report readiness."""

        positions = receipt.context.positions
        if not positions:
            raise PlannerInvariantError("REDUCE receipt has no scope position")
        scope = positions[-1].instance
        closure = self.closures.get((reduce_node, scope))
        if closure is None:
            raise PlannerInvariantError("receipt arrived before reducer closure")
        return bool(closure.settle(receipt))

    def reclaim_closure(self, reduce_node: Any, scope: Any) -> None:
        """Reclaim one reducer closure without touching sibling closures."""

        self.closures.pop((reduce_node, scope), None)

    def reclaim_scope(self, scope: Any) -> None:
        """Reclaim one terminal scope and all of its remaining closures."""

        instance = self.require(scope)
        state = _state_name(instance.state)
        if state == "pending_open":
            raise PlannerInvariantError("cannot reclaim a pending scope")
        for key in tuple(self.closures):
            if key[1] == scope:
                self.closures.pop(key, None)
        instance.state = _enum_member(_state.ScopeState, "RECLAIMED")
        self.instances.pop(scope, None)
        root_scopes = self._by_root.get(instance.root)
        if root_scopes is not None:
            root_scopes.discard(scope)
            if not root_scopes:
                self._by_root.pop(instance.root, None)

    def remove_root(self, root: Any) -> None:
        """Drop all scope indexes for one reclaimed or aborted root."""

        for scope in tuple(self._by_root.get(root, ())):
            instance = self.instances.pop(scope, None)
            if instance is not None:
                instance.state = _enum_member(_state.ScopeState, "RECLAIMED")
            for key in tuple(self.closures):
                if key[1] == scope:
                    self.closures.pop(key, None)
        self._by_root.pop(root, None)

    def require(self, scope: Any) -> Any:
        """Return a known scope instance or raise a semantic invariant error."""

        try:
            return self.instances[scope]
        except KeyError as error:
            raise PlannerInvariantError(f"unknown scope instance: {scope!r}") from error

    def _new_closure(self, **values: Any) -> Any:
        """Build the model-owned closure and its ordinal slot storage."""

        if self._closure_factory is not None:
            return self._closure_factory(**values)
        create = getattr(ClosureView, "create", None)
        if create is not None:
            return create(**values)
        expected = values["expected"]
        slots_type = getattr(_state, "ChunkedReceiptSlots", None)
        if slots_type is None:
            slots: Any = [None] * expected
        else:
            try:
                slots = slots_type(expected)
            except TypeError:
                slots = slots_type(size=expected)
        return _construct(
            ClosureView,
            **values,
            settled=0,
            slots=slots,
            allocation_owner=(values["scope"], values["reduce_node"]),
        )


@dataclass(slots=True)
class OccurrenceReservation:
    """Track owner-keyed durable occurrence credits for one scope."""

    id: Any
    root: Any
    scope: Any
    count: int


class CreditManager:
    """Implement owner-keyed hard-count admission and structural ledgers."""

    def __init__(self, limits: Any) -> None:
        """Initialize empty ledgers under one immutable limits record."""

        self.limits = limits
        self.active_roots = 0
        self.live_grains = 0
        self.live_occurrences = 0
        self.live_structural_edges = 0
        self.pending_dispatches = 0
        self.local_events = 0
        self.structural_leases: dict[Any, Any] = {}
        self._roots: set[Any] = set()
        self._root_occurrences: dict[Any, int] = {}
        self._root_edges: dict[Any, int] = {}
        self._occurrence_reservations: dict[Any, OccurrenceReservation] = {}
        self._occurrence_owner: dict[tuple[Any, Any], Any] = {}
        self._dispatch_owners: set[Any] = set()
        self._event_owners: set[Any] = set()
        self._grain_owners: dict[Any, Any] = {}

    def can_admit_root(self) -> bool:
        """Return whether another root fits the global active-root limit."""

        return self.active_roots < self._limit("max_active_roots")

    def admit_root(self, root: Any) -> bool:
        """Reserve one active-root owner idempotently."""

        if root in self._roots:
            return False
        if not self.can_admit_root():
            return False
        self._roots.add(root)
        self.active_roots += 1
        return True

    def release_root(self, root: Any) -> bool:
        """Release one active root after its resources have been reclaimed."""

        if root not in self._roots:
            return False
        self._roots.remove(root)
        self.active_roots -= 1
        return True

    def release_root_resources(self, root: Any) -> None:
        """Release every structural and occurrence owner held by one root."""

        for scope, lease in tuple(self.structural_leases.items()):
            if lease.root == root:
                self.release_scope(scope)
        for reservation, record in tuple(self._occurrence_reservations.items()):
            if record.root == root:
                self.release_occurrences(reservation)
        for grain, owner_root in tuple(self._grain_owners.items()):
            if owner_root == root:
                self.release_grain(grain)
        self.release_root(root)

    def reserve_grain(self, root: Any, grain: Any) -> bool:
        """Reserve one live executable grain or report global backpressure."""

        existing = self._grain_owners.get(grain)
        if existing is not None:
            if existing != root:
                raise PlannerInvariantError("grain credit changed causal root")
            return True
        if self.live_grains >= self._limit("max_live_grains"):
            return False
        self._grain_owners[grain] = root
        self.live_grains += 1
        return True

    def release_grain(self, grain: Any) -> bool:
        """Release one terminal executable-grain owner idempotently."""

        if grain not in self._grain_owners:
            return False
        self._grain_owners.pop(grain)
        self.live_grains -= 1
        return True

    def try_reserve_scope(
        self,
        root: Any,
        scope: Any,
        expected: int,
        reducer_nodes: Iterable[Any],
    ) -> Any | None:
        """Reserve all known-N closure slots atomically or return backpressure."""

        reducers = tuple(reducer_nodes)
        if expected < 0:
            raise CreditCapacityError("scope cardinality must be non-negative")
        if expected > self._limit("max_scope_width"):
            raise CreditCapacityError("scope exceeds max_scope_width")
        units = _checked_mul(expected, len(reducers))
        global_limit = self._limit("max_live_structural_edges")
        root_limit = self._limit("max_structural_edges_per_root")
        if units > global_limit or units > root_limit:
            raise CreditCapacityError("scope structural reservation can never fit")
        existing = self.structural_leases.get(scope)
        if existing is not None:
            if (
                existing.root != root
                or existing.units != units
                or tuple(existing.allocations) != reducers
            ):
                raise PlannerInvariantError("conflicting structural scope owner")
            return existing
        if self.live_structural_edges + units > global_limit:
            return None
        if self._root_edges.get(root, 0) + units > root_limit:
            return None
        lease_id = _derive_model_id(
            "StructuralLeaseId", "structural-lease", root, scope, expected, reducers
        )
        closure_state = _enum_member(_state.StructuralAllocationState, "CLOSURE")
        lease = _construct(
            _state.StructuralEdgeLease,
            id=lease_id,
            scope=scope,
            root=root,
            units=units,
            allocations={node: closure_state for node in reducers},
        )
        self.structural_leases[scope] = lease
        self.live_structural_edges += units
        self._root_edges[root] = self._root_edges.get(root, 0) + units
        return lease

    def transfer_to_binding(self, scope: Any, reduce_node: Any) -> None:
        """Transfer one reducer allocation from CLOSURE to BINDING."""

        lease = self._require_lease(scope)
        current = lease.allocations.get(reduce_node)
        closure = _enum_member(_state.StructuralAllocationState, "CLOSURE")
        binding = _enum_member(_state.StructuralAllocationState, "BINDING")
        if current == binding:
            return
        if current != closure:
            raise PlannerInvariantError("structural allocation is not a closure")
        lease.allocations[reduce_node] = binding

    def release_binding(self, scope: Any, reduce_node: Any) -> bool:
        """Release one reducer's transferred structural allocation idempotently."""

        lease = self.structural_leases.get(scope)
        if lease is None:
            return False
        released = _enum_member(_state.StructuralAllocationState, "RELEASED")
        current = lease.allocations.get(reduce_node)
        if current == released:
            return False
        if current is None:
            raise PlannerInvariantError("unknown reducer allocation")
        lease.allocations[reduce_node] = released
        self._release_edge_units(lease, self._scope_width(lease))
        return True

    def release_scope(self, scope: Any) -> bool:
        """Release every remaining allocation owned by a scope."""

        lease = self.structural_leases.pop(scope, None)
        if lease is None:
            return False
        released = _enum_member(_state.StructuralAllocationState, "RELEASED")
        width = self._scope_width(lease)
        for node, state in tuple(lease.allocations.items()):
            if state == released:
                continue
            lease.allocations[node] = released
            self._release_edge_units(lease, width)
        return True

    def try_reserve_occurrences(
        self,
        root: Any,
        scope: Any,
        count: int,
    ) -> Any | None:
        """Reserve durable child-coordinate credits for one scope."""

        if count < 0:
            raise CreditCapacityError("occurrence reservation is negative")
        global_limit = self._limit("max_live_occurrences")
        root_limit = self._limit("max_occurrences_per_root")
        if count > global_limit or count > root_limit:
            raise CreditCapacityError("scope occurrence reservation can never fit")
        owner = (root, scope)
        existing_id = self._occurrence_owner.get(owner)
        if existing_id is not None:
            existing = self._occurrence_reservations[existing_id]
            if existing.count != count:
                raise PlannerInvariantError("conflicting occurrence reservation")
            return existing.id
        if self.live_occurrences + count > global_limit:
            return None
        if self._root_occurrences.get(root, 0) + count > root_limit:
            return None
        reservation_id = _derive_model_id(
            "CreditReservationId", "occurrence-credit", root, scope, count
        )
        reservation = OccurrenceReservation(reservation_id, root, scope, count)
        self._occurrence_owner[owner] = reservation_id
        self._occurrence_reservations[reservation_id] = reservation
        self.live_occurrences += count
        self._root_occurrences[root] = self._root_occurrences.get(root, 0) + count
        return reservation_id

    def release_occurrences(self, reservation: Any) -> bool:
        """Release one exact occurrence reservation idempotently."""

        record = self._occurrence_reservations.pop(reservation, None)
        if record is None:
            return False
        self._occurrence_owner.pop((record.root, record.scope), None)
        self.live_occurrences -= record.count
        remaining = self._root_occurrences.get(record.root, 0) - record.count
        if remaining:
            self._root_occurrences[record.root] = remaining
        else:
            self._root_occurrences.pop(record.root, None)
        return True

    def reserve_dispatch(self, dispatch: Any) -> bool:
        """Reserve one pending dispatch owner or return backpressure."""

        if dispatch in self._dispatch_owners:
            return True
        if self.pending_dispatches >= self._limit("max_pending_dispatches"):
            return False
        self._dispatch_owners.add(dispatch)
        self.pending_dispatches += 1
        return True

    def release_dispatch(self, dispatch: Any) -> bool:
        """Release one pending dispatch owner idempotently."""

        if dispatch not in self._dispatch_owners:
            return False
        self._dispatch_owners.remove(dispatch)
        self.pending_dispatches -= 1
        return True

    def reserve_event(self, owner: Any) -> bool:
        """Reserve one bounded local-event owner idempotently."""

        if owner in self._event_owners:
            return True
        if self.local_events >= self._limit("max_local_events"):
            return False
        self._event_owners.add(owner)
        self.local_events += 1
        return True

    def release_event(self, owner: Any) -> bool:
        """Release one local-event owner after dequeue."""

        if owner not in self._event_owners:
            return False
        self._event_owners.remove(owner)
        self.local_events -= 1
        return True

    def _limit(self, name: str) -> int:
        """Read a non-negative hard limit from the runtime record."""

        value = getattr(self.limits, name, 2**63 - 1)
        if value is None:
            return 2**63 - 1
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        return int(value)

    def _require_lease(self, scope: Any) -> Any:
        """Return a structural lease or raise a semantic invariant error."""

        try:
            return self.structural_leases[scope]
        except KeyError as error:
            raise PlannerInvariantError("unknown structural scope lease") from error

    def _scope_width(self, lease: Any) -> int:
        """Recover per-reducer units from a structural lease."""

        count = len(lease.allocations)
        return 0 if count == 0 else lease.units // count

    def _release_edge_units(self, lease: Any, units: int) -> None:
        """Decrement global and per-root structural ledgers safely."""

        self.live_structural_edges -= units
        remaining = self._root_edges.get(lease.root, 0) - units
        if remaining:
            self._root_edges[lease.root] = remaining
        else:
            self._root_edges.pop(lease.root, None)


class ReadyScheduler:
    """Fairly batch READY grains by NodeId while allowing cross-root mixing."""

    def __init__(
        self,
        graph: Any | None = None,
        grains: Any | None = None,
        *,
        default_max_size: int = 1,
        default_wait_ms: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create empty per-node queues and a round-robin node ring."""

        if default_max_size <= 0:
            raise ValueError("default_max_size must be positive")
        if default_wait_ms < 0:
            raise ValueError("default_wait_ms must be non-negative")
        self.graph = graph
        self.grains = grains
        self.default_max_size = default_max_size
        self.default_wait_ms = default_wait_ms
        self.clock = clock
        self.queues: dict[Any, deque[Any]] = {}
        self.first_ready_at: dict[Any, float] = {}
        self.round_robin_nodes: deque[Any] = deque()
        self._queued: dict[Any, Any] = {}
        self._isolation: deque[BatchSelection] = deque()

    def __len__(self) -> int:
        """Return the number of ordinary queued grains."""

        return len(self._queued)

    def enqueue(
        self,
        grain: Any,
        *,
        node: Any | None = None,
        ready_at: float | None = None,
    ) -> bool:
        """Append one READY grain to its node queue idempotently."""

        grain_id, resolved_node = self._identity_and_node(grain, node)
        existing = self._queued.get(grain_id)
        if existing is not None:
            if existing != resolved_node:
                raise PlannerInvariantError("READY grain changed node")
            return False
        queue = self.queues.setdefault(resolved_node, deque())
        if not queue:
            self.first_ready_at[resolved_node] = (
                self.clock() if ready_at is None else ready_at
            )
            self.round_robin_nodes.append(resolved_node)
        queue.append(grain_id)
        self._queued[grain_id] = resolved_node
        return True

    def enqueue_isolation(self, node: Any, grains: Iterable[Any]) -> None:
        """Queue an isolation group that must not mix with ordinary READY work."""

        selected = tuple(grains)
        if not selected:
            return
        self._isolation.append(BatchSelection(node=node, grains=selected))

    def remove(self, grain: Any) -> bool:
        """Remove one grain from READY indexes idempotently."""

        grain_id = getattr(grain, "id", grain)
        node = self._queued.pop(grain_id, None)
        if node is None:
            return False
        queue = self.queues[node]
        queue.remove(grain_id)
        if not queue:
            self._drop_node(node)
        return True

    def select_batch(
        self,
        now: float | None = None,
        *,
        force: bool = False,
    ) -> BatchSelection | None:
        """Select one full, expired, or explicitly forced same-node batch."""

        if self._isolation:
            return self._isolation.popleft()
        if not self.round_robin_nodes:
            return None
        now = self.clock() if now is None else now
        for _ in range(len(self.round_robin_nodes)):
            node = self.round_robin_nodes[0]
            self.round_robin_nodes.rotate(-1)
            queue = self.queues[node]
            max_size, wait_ms = self._policy(node)
            elapsed_ms = (now - self.first_ready_at[node]) * 1000.0
            if len(queue) < max_size and not force and elapsed_ms < wait_ms:
                continue
            count = min(max_size, len(queue))
            selected = tuple(queue.popleft() for _ in range(count))
            for grain_id in selected:
                self._queued.pop(grain_id, None)
            if queue:
                self.first_ready_at[node] = now
            else:
                self._drop_node(node)
            return BatchSelection(node=node, grains=selected)
        return None

    def _identity_and_node(self, grain: Any, node: Any | None) -> tuple[Any, Any]:
        """Resolve a grain ID and NodeId from a record or GrainStore."""

        spec = getattr(grain, "spec", None)
        if spec is not None:
            return spec.id, spec.node if node is None else node
        grain_id = getattr(grain, "id", grain)
        if node is not None:
            return grain_id, node
        if self.grains is None:
            raise ValueError("enqueue(grain_id) requires a node or GrainStore")
        record = _grain_get(self.grains, grain_id)
        return grain_id, record.spec.node

    def _policy(self, node_id: Any) -> tuple[int, float]:
        """Read a MAP node's frozen batch policy."""

        if self.graph is None:
            return self.default_max_size, self.default_wait_ms
        node = _graph_node(self.graph, node_id)
        op = node.op
        execution = getattr(op, "execution", None)
        batch = getattr(execution, "batch", execution)
        max_size = getattr(batch, "max_size", self.default_max_size)
        wait_ms = getattr(batch, "max_wait_ms", self.default_wait_ms)
        return int(max_size), float(wait_ms)

    def _drop_node(self, node: Any) -> None:
        """Remove an empty node from all scheduler indexes."""

        self.queues.pop(node, None)
        self.first_ready_at.pop(node, None)
        try:
            self.round_robin_nodes.remove(node)
        except ValueError:
            pass


class EventPlanner:
    """Incrementally execute MAP planning and FILTER/EXPAND/REDUCE semantics."""

    def __init__(
        self,
        graph: Any,
        run: Any,
        *,
        receipts: ReceiptStore,
        controls: Any,
        grains: Any,
        values: Any,
        scopes: ScopeTracker,
        scheduler: ReadyScheduler,
        credits: CreditManager,
        errors: Any | None = None,
        expansion_chunk_size: int = 64,
    ) -> None:
        """Bind one run's stores, indexes, and bounded local event queue."""

        if expansion_chunk_size <= 0:
            raise ValueError("expansion_chunk_size must be positive")
        self.graph = graph
        self.run = run
        self.receipts = receipts
        self.controls = controls
        self.grains = grains
        self.values = values
        self.scopes = scopes
        self.scheduler = scheduler
        self.credits = credits
        self.errors = errors
        self.expansion_chunk_size = expansion_chunk_size
        self.events: deque[tuple[Any, Any]] = deque()
        self.latches: dict[tuple[Any, Any, tuple[Any, ...]], InputLatch] = {}
        self._resolved_invocations: set[
            tuple[Any, Any, tuple[Any, ...]]
        ] = set()
        self._resolved_by_root: dict[Any, set[tuple[Any, Any, tuple[Any, ...]]]] = {}
        self.pending_expansions: dict[Any, Any] = {}
        self.waiting_expansions: dict[Any, InputLatch] = {}
        self.waiting_grains: dict[
            tuple[Any, Any, tuple[Any, ...]], InputLatch
        ] = {}
        self._next_event_owner = 0
        self._nodes = {node.id: node for node in graph.nodes}

    def enqueue(self, event: Any) -> None:
        """Append one local event without invoking its handler synchronously."""

        reservations = self.reserve_events((event,))
        if reservations is None:
            raise PlannerInvariantError("local event queue exceeds max_local_events")
        self.commit_reserved_events(reservations, (event,))

    def reserve_events(self, events: Iterable[Any]) -> tuple[Any, ...] | None:
        """Reserve an entire event batch atomically without publishing it."""

        reservations: list[Any] = []
        for _event in events:
            owner = ("event", self._next_event_owner)
            self._next_event_owner += 1
            if not self.credits.reserve_event(owner):
                for reserved in reversed(reservations):
                    self.credits.release_event(reserved)
                return None
            reservations.append(owner)
        return tuple(reservations)

    def commit_reserved_events(
        self,
        reservations: tuple[Any, ...],
        events: tuple[Any, ...],
    ) -> None:
        """Append events whose complete credit batch is already reserved."""

        if len(reservations) != len(events):
            raise PlannerInvariantError("event reservation cardinality mismatch")
        self.events.extend(zip(reservations, events))

    def release_reserved_events(self, reservations: Iterable[Any]) -> None:
        """Release an uncommitted event reservation batch idempotently."""

        for owner in reservations:
            self.credits.release_event(owner)

    def publish(self, receipt: Any) -> bool:
        """Atomically publish one receipt with its downstream event credit."""

        return self._publish(receipt)

    def drain(self, budget: int) -> int:
        """Handle at most ``budget`` queued events and return the work count."""

        if budget < 0:
            raise ValueError("drain budget must be non-negative")
        handled = 0
        while self.events and handled < budget:
            owner, event = self.events.popleft()
            self.credits.release_event(owner)
            self._handle_event(event)
            handled += 1
        return handled

    def retry_waiting_expansions(self, budget: int) -> int:
        """Retry bounded transient EXPAND reservations without duplicating facts."""

        if budget < 0:
            raise ValueError("retry budget must be non-negative")
        attempted = 0
        for scope in tuple(self.waiting_expansions)[:budget]:
            latch = self.waiting_expansions.pop(scope)
            self.resolve_expand(latch)
            if scope not in self.waiting_expansions:
                attempted += 1
        return attempted

    def retry_waiting_grains(self, budget: int) -> int:
        """Retry MAP invocations blocked by the live-grain hard limit."""

        if budget < 0:
            raise ValueError("retry budget must be non-negative")
        processed = 0
        for key in tuple(self.waiting_grains)[:budget]:
            latch = self.waiting_grains.pop(key)
            try:
                self.resolve_map(latch)
            except GrainBackpressure:
                self.waiting_grains[key] = latch
                continue
            self._mark_resolved(key)
            processed += 1
        return processed

    def remove_root(self, root: Any) -> None:
        """Remove run-planner indexes owned by a reclaimed or aborted root."""

        for key, latch in tuple(self.latches.items()):
            if latch.context.root == root:
                self.latches.pop(key, None)
        for key, latch in tuple(self.waiting_grains.items()):
            if latch.context.root == root:
                self.waiting_grains.pop(key, None)
        for key in self._resolved_by_root.pop(root, set()):
            self._resolved_invocations.discard(key)
        for scope, latch in tuple(self.waiting_expansions.items()):
            if latch.context.root == root:
                self.waiting_expansions.pop(scope, None)
        for scope, pending in tuple(self.pending_expansions.items()):
            instance = self.scopes.instances.get(scope)
            if instance is not None and instance.root == root:
                self.pending_expansions.pop(scope, None)
                self.credits.release_occurrences(
                    pending.occurrence_reservation
                )
        retained: deque[tuple[Any, Any]] = deque()
        while self.events:
            owner, event = self.events.popleft()
            event_root = _event_root(event, self.scopes)
            if event_root == root:
                self.credits.release_event(owner)
            else:
                retained.append((owner, event))
        self.events = retained

    def on_receipt(self, receipt: Any) -> None:
        """Route one terminal coordinate to compiled consumers."""

        consumers = self.graph.consumers_by_port.get(receipt.item.port, ())
        for node_id in consumers:
            node = self._nodes[node_id]
            if _op_kind(node.op) == "reduce":
                self.settle_reduce(node_id, receipt)
            else:
                self._settle_latch(node, receipt)

    def on_scope_event(self, event: Any) -> None:
        """Route OPEN/ABSENT/FAILED lifecycle events to every static reducer."""

        instance = self.scopes.require(event.scope)
        plan = self.graph.scope_plans[instance.definition]
        kind = type(event).__name__.lower()
        for reduce_node in plan.reducers:
            node = self._nodes[reduce_node]
            input_port = node.op.input
            if "opened" in kind:
                closure = self.scopes.ensure_closure(
                    reduce_node, event.scope, input_port
                )
                if closure.expected == 0:
                    self.enqueue(ClosureReady(closure))
            elif "absent" in kind:
                self._finalize_origin_reduce(node, instance, origin=event.origin)
            elif "failed" in kind:
                self._finalize_origin_reduce(node, instance, cause=event.cause)
            else:
                raise PlannerInvariantError(f"unknown scope event: {event!r}")

    def resolve_map(self, latch: InputLatch) -> None:
        """Resolve a fully settled symmetric MAP input latch."""

        receipts = tuple(_require_receipts(latch))
        node = self._nodes[latch.node]
        failures = tuple(
            receipt.producer for receipt in receipts if _is_failure(receipt)
        )
        output_items = self._aligned_output_items(node, receipts[0].item.entity)
        if failures:
            self._terminal_outputs(
                node,
                latch.context,
                receipts,
                output_items,
                _make_outcome("Suppressed", causes=_dedupe(failures)),
                "SUPPRESSED",
            )
            return
        absent = tuple(
            receipt.item for receipt in receipts if _is_absent(receipt)
        )
        if absent:
            self._terminal_outputs(
                node,
                latch.context,
                receipts,
                output_items,
                _make_outcome("Skipped", reasons=_dedupe(absent)),
                "NORMAL_ABSENCE",
            )
            return
        spec = self._grain_spec(node, latch.context, receipts, output_items)
        record = self._ensure_grain(spec, executable=True)
        self.scheduler.enqueue(record, node=node.id)

    def resolve_filter(self, latch: InputLatch) -> None:
        """Resolve FILTER truth while preserving false as NORMAL_ABSENCE."""

        receipts = tuple(_require_receipts(latch))
        node = self._nodes[latch.node]
        by_port = {receipt.item.port: receipt for receipt in receipts}
        mask = by_port[node.op.mask]
        target = by_port[node.op.target]
        output = self._aligned_output_items(node, target.item.entity)
        if _is_failure(mask) or _is_failure(target):
            causes = _dedupe(
                tuple(
                    receipt.producer
                    for receipt in (mask, target)
                    if _is_failure(receipt)
                )
            )
            self._terminal_outputs(
                node,
                latch.context,
                receipts,
                output,
                _make_outcome("Suppressed", causes=causes),
                "SUPPRESSED",
            )
        elif _is_absent(mask) or _is_absent(target):
            reasons = tuple(
                receipt.item
                for receipt in (mask, target)
                if _is_absent(receipt)
            )
            self._terminal_outputs(
                node,
                latch.context,
                receipts,
                output,
                _make_outcome("Skipped", reasons=reasons),
                "NORMAL_ABSENCE",
            )
        elif self.controls.get_bool(mask.item):
            self.values.alias_item(
                latch.context.root, output[0], target.item
            )
            self._terminal_outputs(
                node,
                latch.context,
                receipts,
                output,
                _make_outcome("Success"),
                "PRESENT",
            )
        else:
            self._terminal_outputs(
                node,
                latch.context,
                receipts,
                output,
                _make_outcome("Success"),
                "NORMAL_ABSENCE",
            )
        self.controls.release_consumer(mask.item)

    def resolve_expand(self, latch: InputLatch) -> None:
        """Open one dynamic scope and expand only the outer structural layer."""

        receipts = tuple(_require_receipts(latch))
        if len(receipts) != 1:
            raise PlannerInvariantError("EXPAND requires exactly one input")
        receipt = receipts[0]
        node = self._nodes[latch.node]
        grain_id = _grain_id("expand", self.run, node.id, receipt.item)
        scope = _derive_model_id(
            "ScopeInstanceId", "scope-instance", self.run, node.id, receipt.item
        )
        self.scopes.create_pending(
            scope,
            node.op.scope,
            receipt.context.root,
            receipt.item,
            receipt.context,
            grain_id,
        )
        if _is_failure(receipt):
            empty_range = _construct(
                _sem.ExpandOutputRange,
                port=node.outputs[0].id,
                parent=receipt.item,
                count=0,
            )
            spec = self._grain_spec(
                node,
                receipt.context,
                receipts,
                empty_range,
                grain_id=grain_id,
            )
            self._seal_spec(
                spec,
                _make_outcome("Suppressed", causes=(receipt.producer,)),
            )
            self.enqueue(self.scopes.mark_failed(scope, grain_id))
            return
        if _is_absent(receipt):
            empty_range = _construct(
                _sem.ExpandOutputRange,
                port=node.outputs[0].id,
                parent=receipt.item,
                count=0,
            )
            spec = self._grain_spec(
                node,
                receipt.context,
                receipts,
                empty_range,
                grain_id=grain_id,
            )
            self._seal_spec(
                spec,
                _make_outcome("Skipped", reasons=(receipt.item,)),
            )
            self.enqueue(self.scopes.mark_absent(scope, receipt))
            return
        list_node = _item_node(self.values, receipt.item)
        if list_node is None:
            raise PlannerInvariantError("PRESENT EXPAND input has no ValueNode")
        expected = self.values.outer_length(list_node)
        reducers = self.graph.scope_plans[node.op.scope].reducers
        pending_limit = int(
            getattr(self.credits.limits, "max_pending_expansions", 2**63 - 1)
        )
        if (
            scope not in self.pending_expansions
            and len(self.pending_expansions) >= pending_limit
        ):
            self.waiting_expansions[scope] = latch
            return
        lease = None
        reservation = None
        try:
            lease = self.credits.try_reserve_scope(
                receipt.context.root, scope, expected, reducers
            )
            reservation = self.credits.try_reserve_occurrences(
                receipt.context.root, scope, expected
            )
        except CreditCapacityError as error:
            if lease is not None:
                self.credits.release_scope(scope)
            if reservation is not None:
                self.credits.release_occurrences(reservation)
            self._fail_expand_capacity(
                node, receipt, scope, grain_id, error
            )
            return
        if lease is None or reservation is None:
            if lease is not None:
                self.credits.release_scope(scope)
            if reservation is not None:
                self.credits.release_occurrences(reservation)
            self.waiting_expansions[scope] = latch
            return
        self.waiting_expansions.pop(scope, None)
        output_port = node.outputs[0].id
        output_range = _construct(
            _sem.ExpandOutputRange,
            port=output_port,
            parent=receipt.item,
            count=expected,
        )
        spec = self._grain_spec(
            node,
            receipt.context,
            receipts,
            output_range,
            grain_id=grain_id,
        )
        self._seal_spec(spec, _make_outcome("Success"))
        self.enqueue(self.scopes.open(scope, expected, lease))
        if expected == 0:
            return
        pending = _construct(
            _state.PendingExpansion,
            scope=scope,
            grain=grain_id,
            structural_lease=lease.id,
            occurrence_reservation=reservation,
            input_item=receipt.item,
            list_node=list_node,
            expected=expected,
            next_ordinal=0,
        )
        self.pending_expansions[scope] = pending
        self.enqueue(ExpansionCreditGranted(scope))

    def expand_chunk(self, pending: Any) -> None:
        """Materialize a bounded contiguous ordinal range for one EXPAND."""

        instance = self.scopes.require(pending.scope)
        node = self._nodes[
            self.graph.scope_plans[instance.definition].expand_node
        ]
        output_port = node.outputs[0].id
        stop = min(
            pending.expected,
            pending.next_ordinal + self.expansion_chunk_size,
        )
        for ordinal in range(pending.next_ordinal, stop):
            child_node = self.values.child_at(pending.list_node, ordinal)
            entity = _derive_model_id(
                "EntityId",
                "expand-entity",
                self.run,
                node.id,
                pending.input_item,
                ordinal,
            )
            item = _construct(
                _sem.ItemRef,
                port=output_port,
                entity=entity,
            )
            position = _construct(
                _sem.ScopePosition,
                instance=pending.scope,
                ordinal=ordinal,
            )
            context = _construct(
                _sem.OccurrenceContext,
                root=instance.root,
                positions=instance.parent_context.positions + (position,),
            )
            self.values.bind_item(instance.root, item, child_node)
            receipt = _construct(
                _sem.Receipt,
                item=item,
                context=context,
                state=_enum_member(_sem.ReceiptState, "PRESENT"),
                producer=pending.grain,
            )
            self._publish(receipt)
        pending.next_ordinal = stop
        if stop < pending.expected:
            self.enqueue(ExpansionCreditGranted(pending.scope))
        else:
            self.pending_expansions.pop(pending.scope, None)

    def settle_reduce(self, node: Any, receipt: Any) -> None:
        """Settle one REDUCE ordinal and enqueue closure finalization once."""

        became_ready = self.scopes.settle(node, receipt)
        if became_ready:
            scope = receipt.context.positions[-1].instance
            closure = self.scopes.closures[(node, scope)]
            self.enqueue(ClosureReady(closure))

    def finalize_reduce(self, closure: Any) -> None:
        """Create one terminal REDUCE grain from a complete ordinal closure."""

        if closure.settled != closure.expected:
            raise PlannerInvariantError("cannot finalize an incomplete closure")
        instance = self.scopes.require(closure.scope)
        node = self._nodes[closure.reduce_node]
        receipts = _closure_receipts(closure)
        failures = tuple(
            (ordinal, receipt.producer)
            for ordinal, receipt in enumerate(receipts)
            if _is_failure(receipt)
        )
        output = self._reduce_output_item(node, instance)
        binding = self._reduce_binding(
            instance,
            closure,
            receipts,
            failures=failures,
        )
        spec = self._grain_spec(
            node,
            instance.parent_context,
            (),
            (output,),
            reduce_binding=binding,
        )
        self.credits.transfer_to_binding(closure.scope, closure.reduce_node)
        if failures:
            self._seal_spec(
                spec,
                _make_outcome(
                    "Suppressed",
                    causes=_dedupe(tuple(cause for _, cause in failures)),
                ),
            )
            receipt_state = "SUPPRESSED"
        else:
            members = tuple(
                _item_node(self.values, receipt.item)
                for receipt in receipts
                if _is_present(receipt)
            )
            if any(member is None for member in members):
                raise PlannerInvariantError("REDUCE member lost its ValueNode")
            composite = self.values.create_composite(members)
            self.values.bind_item(instance.root, output, composite)
            self._seal_spec(spec, _make_outcome("Success"))
            receipt_state = "PRESENT"
        self._publish(
            _construct(
                _sem.Receipt,
                item=output,
                context=instance.parent_context,
                state=_enum_member(_sem.ReceiptState, receipt_state),
                producer=spec.id,
            )
        )
        self.scopes.reclaim_closure(closure.reduce_node, closure.scope)

    def _handle_event(self, event: Any) -> None:
        """Dispatch one local event to its bounded handler."""

        if isinstance(event, ReceiptPublished):
            self.on_receipt(event.receipt)
            return
        if isinstance(event, ClosureReady):
            self.finalize_reduce(event.closure)
            return
        if isinstance(event, ExpansionCreditGranted):
            pending = self.pending_expansions.get(event.scope)
            if pending is not None:
                self.expand_chunk(pending)
            return
        name = type(event).__name__
        if name in {"ScopeOpened", "ScopeAbsent", "ScopeFailed"}:
            self.on_scope_event(event)
            return
        if hasattr(event, "item") and hasattr(event, "context"):
            self.on_receipt(event)
            return
        raise PlannerInvariantError(f"unknown local event: {event!r}")

    def _settle_latch(self, node: Any, receipt: Any) -> None:
        """Install aligned receipts and resolve a complete invocation latch."""

        key = (node.id, receipt.context.root, receipt.context.positions)
        if key in self._resolved_invocations or key in self.waiting_grains:
            return
        latch = self.latches.get(key)
        if latch is None:
            latch = InputLatch(
                node=node.id,
                context=receipt.context,
                expected_roles=tuple(binding.role for binding in node.inputs),
                expected_ports=tuple(binding.port for binding in node.inputs),
                receipts=[None] * len(node.inputs),
            )
            self.latches[key] = latch
        for index, binding in enumerate(node.inputs):
            item = _construct(
                _sem.ItemRef,
                port=binding.port,
                entity=receipt.item.entity,
            )
            candidate = self.receipts.get(item)
            if candidate is not None:
                latch.install(index, candidate)
        if not latch.ready:
            return
        self.latches.pop(key, None)
        kind = _op_kind(node.op)
        if kind == "map":
            try:
                self.resolve_map(latch)
            except GrainBackpressure:
                self.waiting_grains[key] = latch
                return
        elif kind == "filter":
            self.resolve_filter(latch)
        elif kind == "expand":
            self.resolve_expand(latch)
        else:
            raise PlannerInvariantError(f"unsupported latch node kind: {kind}")
        self._mark_resolved(key)

    def _mark_resolved(self, key: tuple[Any, Any, tuple[Any, ...]]) -> None:
        """Index one resolved invocation by root for bounded reclamation."""

        self._resolved_invocations.add(key)
        self._resolved_by_root.setdefault(key[1], set()).add(key)

    def _aligned_output_items(self, node: Any, entity: Any) -> tuple[Any, ...]:
        """Build fixed output coordinates sharing one aligned entity."""

        return tuple(
            _construct(_sem.ItemRef, port=port.id, entity=entity)
            for port in node.outputs
        )

    def _grain_spec(
        self,
        node: Any,
        context: Any,
        receipts: Iterable[Any],
        output_slots: Any,
        *,
        grain_id: Any | None = None,
        reduce_binding: Any | None = None,
        lineage_inputs: tuple[Any, ...] = (),
    ) -> Any:
        """Build an immutable GrainSpec from compiled role order."""

        receipt_by_port = {receipt.item.port: receipt for receipt in receipts}
        role_bindings = tuple(
            _construct(
                _sem.RoleBinding,
                role=binding.role,
                items=(receipt_by_port[binding.port].item,),
            )
            for binding in node.inputs
            if binding.port in receipt_by_port
        )
        if lineage_inputs:
            role_bindings += (
                _construct(
                    _sem.RoleBinding,
                    role="origin",
                    items=lineage_inputs,
                ),
            )
        if grain_id is None:
            kind = _op_kind(node.op)
            ordered_items = tuple(
                item for role in role_bindings for item in role.items
            )
            if kind == "filter":
                grain_id = _sem.GrainId.filter(
                    self.run,
                    node.id,
                    receipt_by_port[node.op.mask].item,
                    receipt_by_port[node.op.target].item,
                )
            elif kind == "reduce":
                if reduce_binding is None:
                    raise PlannerInvariantError(
                        "REDUCE GrainSpec requires ReduceBinding"
                    )
                grain_id = _sem.GrainId.reduce(
                    self.run,
                    node.id,
                    reduce_binding.scope,
                )
            else:
                grain_id = _grain_id(
                    kind,
                    self.run,
                    node.id,
                    ordered_items,
                )
        return _construct(
            _sem.GrainSpec,
            id=grain_id,
            run=self.run,
            root=context.root,
            node=node.id,
            context=context,
            inputs=role_bindings,
            output_slots=output_slots,
            reduce_binding=reduce_binding,
        )

    def _ensure_grain(self, spec: Any, *, executable: bool) -> Any:
        """Ensure a grain record and return a stable existing/new record."""

        existing = _grain_get_optional(self.grains, spec.id)
        if existing is not None:
            if getattr(existing, "spec", None) != spec:
                raise PlannerInvariantError("conflicting GrainSpec")
            return existing
        if executable and not self.credits.reserve_grain(spec.root, spec.id):
            raise GrainBackpressure("max_live_grains is temporarily exhausted")
        ensure = getattr(self.grains, "ensure", None)
        if ensure is not None:
            for args, kwargs in (
                ((spec,), {"executable": executable}),
                ((spec,), {}),
            ):
                try:
                    record = ensure(*args, **kwargs)
                    return record if record is not None else _grain_get(
                        self.grains, spec.id
                    )
                except TypeError:
                    continue
                except BaseException:
                    if executable:
                        self.credits.release_grain(spec.id)
                    raise
        if isinstance(self.grains, dict):
            existing = self.grains.get(spec.id)
            if existing is not None:
                if existing.spec != spec:
                    raise PlannerInvariantError("conflicting GrainSpec")
                return existing
            phase = "ready" if executable else "terminal"
            record = PlannerGrainRecord(spec=spec, phase=phase)
            self.grains[spec.id] = record
            return record
        if executable:
            self.credits.release_grain(spec.id)
        raise PlannerInvariantError("GrainStore.ensure signature unsupported")

    def _seal_spec(self, spec: Any, outcome: Any) -> Any:
        """Ensure and seal one centrally executed system grain."""

        record = self._ensure_grain(spec, executable=False)
        seal = getattr(self.grains, "seal", None)
        if seal is not None:
            for args, kwargs in (
                ((spec.id, outcome), {}),
                ((spec.id,), {"outcome": outcome}),
                ((record, outcome), {}),
            ):
                try:
                    seal(*args, **kwargs)
                    return record
                except TypeError:
                    continue
        if isinstance(record, PlannerGrainRecord):
            if record.outcome is not None and record.outcome != outcome:
                raise PlannerInvariantError("grain already sealed differently")
            record.outcome = outcome
            record.phase = "terminal"
            return record
        record_seal = getattr(record, "seal", None)
        if record_seal is not None:
            record_seal(outcome)
            return record
        raise PlannerInvariantError("GrainStore cannot seal system grain")

    def _terminal_outputs(
        self,
        node: Any,
        context: Any,
        input_receipts: tuple[Any, ...],
        output_items: tuple[Any, ...],
        outcome: Any,
        receipt_state: str,
    ) -> None:
        """Atomically order system grain sealing before receipt publication."""

        spec = self._grain_spec(node, context, input_receipts, output_items)
        self._seal_spec(spec, outcome)
        state = _enum_member(_sem.ReceiptState, receipt_state)
        for item in output_items:
            self._publish(
                _construct(
                    _sem.Receipt,
                    item=item,
                    context=context,
                    state=state,
                    producer=spec.id,
                )
            )

    def _publish(self, receipt: Any) -> bool:
        """Publish once and enqueue only genuinely new downstream work."""

        existing = self.receipts.get(receipt.item)
        if existing is not None:
            return self.receipts.publish_once(receipt)
        event = ReceiptPublished(receipt)
        reservations = self.reserve_events((event,))
        if reservations is None:
            raise PlannerInvariantError("local event queue exceeds max_local_events")
        try:
            published = self.receipts.publish_once(receipt)
        except BaseException:
            self.release_reserved_events(reservations)
            raise
        if not published:
            self.release_reserved_events(reservations)
            return False
        self.commit_reserved_events(reservations, (event,))
        return True

    def _fail_expand_capacity(
        self,
        node: Any,
        receipt: Any,
        scope: Any,
        grain_id: Any,
        error: CreditCapacityError,
    ) -> None:
        """Convert a permanently impossible EXPAND into root-local failure."""

        output_range = _construct(
            _sem.ExpandOutputRange,
            port=node.outputs[0].id,
            parent=receipt.item,
            count=0,
        )
        spec = self._grain_spec(
            node,
            receipt.context,
            (receipt,),
            output_range,
            grain_id=grain_id,
        )
        error_id = _derive_model_id(
            "ErrorId", "expand-capacity-error", self.run, grain_id
        )
        if self.errors is not None:
            self.errors.put(
                _construct(
                    _sem.ErrorRecord,
                    id=error_id,
                    root=receipt.context.root,
                    grain=grain_id,
                    kind="EXPAND_CAPACITY",
                    message=str(error),
                    trace_digest=None,
                )
            )
        self._seal_spec(spec, _make_outcome("Failed", error=error_id))
        self.enqueue(self.scopes.mark_failed(scope, grain_id))

    def _finalize_origin_reduce(
        self,
        node: Any,
        instance: Any,
        *,
        origin: Any | None = None,
        cause: Any | None = None,
    ) -> None:
        """Finalize REDUCE directly for ScopeAbsent or ScopeFailed."""

        output = self._reduce_output_item(node, instance)
        binding = self._reduce_binding(
            instance,
            None,
            (),
            origin=origin,
            cause=cause,
        )
        spec = self._grain_spec(
            node,
            instance.parent_context,
            (),
            (output,),
            reduce_binding=binding,
            lineage_inputs=(() if origin is None else (origin.item,)),
        )
        if origin is not None:
            outcome = _make_outcome("Skipped", reasons=(origin.item,))
            state = "NORMAL_ABSENCE"
        else:
            outcome = _make_outcome("Suppressed", causes=(cause,))
            state = "SUPPRESSED"
        self._seal_spec(spec, outcome)
        self._publish(
            _construct(
                _sem.Receipt,
                item=output,
                context=instance.parent_context,
                state=_enum_member(_sem.ReceiptState, state),
                producer=spec.id,
            )
        )

    def _reduce_output_item(self, node: Any, instance: Any) -> Any:
        """Create the one root/outer-scope coordinate emitted by REDUCE."""

        return _construct(
            _sem.ItemRef,
            port=node.outputs[0].id,
            entity=instance.parent_item.entity,
        )

    def _reduce_binding(
        self,
        instance: Any,
        closure: Any | None,
        receipts: tuple[Any, ...],
        *,
        failures: tuple[tuple[int, Any], ...] = (),
        origin: Any | None = None,
        cause: Any | None = None,
    ) -> Any:
        """Canonicalize known-N or origin-terminal REDUCE lineage."""

        if closure is not None:
            to_binding = getattr(closure, "to_binding", None)
            if to_binding is not None:
                return to_binding(instance)
        present = tuple(
            receipt.item for receipt in receipts if _is_present(receipt)
        )
        absent = tuple(
            index for index, receipt in enumerate(receipts) if _is_absent(receipt)
        )
        compressed_type = getattr(_state, "CompressedOrdinalSet", None)
        if compressed_type is not None:
            from_ordinals = getattr(compressed_type, "from_ordinals", None)
            absent_value = (
                from_ordinals(absent)
                if from_ordinals is not None
                else compressed_type(absent)
            )
        else:
            absent_value = absent
        return _construct(
            _state.ReduceBinding,
            scope=instance.id,
            anchor=instance.parent_item,
            allocation_owner=(
                None
                if closure is None
                else (closure.scope, closure.reduce_node)
            ),
            expected=None if closure is None else closure.expected,
            present_members=present,
            absent_ordinals=absent_value,
            direct_failures=failures,
            origin_receipt=origin,
            origin_failure=cause,
        )


def _construct(cls: type[Any], /, **values: Any) -> Any:
    """Construct a model record while tolerating omitted default-only fields."""

    try:
        return cls(**values)
    except TypeError as first_error:
        try:
            signature = inspect.signature(cls)
        except (TypeError, ValueError):
            raise first_error
        accepted = {
            name: value
            for name, value in values.items()
            if name in signature.parameters
        }
        try:
            return cls(**accepted)
        except TypeError:
            raise first_error


def _enum_member(enum_type: type[Any], name: str) -> Any:
    """Resolve an enum member by canonical name or lowercase value."""

    member = getattr(enum_type, name, None)
    if member is not None:
        return member
    lowered = name.lower()
    for candidate in enum_type:
        if str(candidate.value).lower() == lowered:
            return candidate
    raise PlannerInvariantError(f"{enum_type.__name__} has no {name} member")


def _state_name(state: Any) -> str:
    """Normalize an Enum or string semantic state to lowercase."""

    return str(getattr(state, "value", state)).lower()


def _event_root(event: Any, scopes: ScopeTracker) -> Any | None:
    """Resolve a queued local event's causal root for abort-time cleanup."""

    receipt = getattr(event, "receipt", None)
    if receipt is not None:
        return receipt.context.root
    context = getattr(event, "context", None)
    if context is not None:
        return context.root
    closure = getattr(event, "closure", None)
    scope = getattr(closure, "scope", None)
    if scope is None:
        scope = getattr(event, "scope", None)
    instance = scopes.instances.get(scope)
    return None if instance is None else instance.root


def _is_present(receipt: Any) -> bool:
    """Return whether a receipt owns a ValueNode binding."""

    return _state_name(receipt.state) == "present"


def _is_absent(receipt: Any) -> bool:
    """Return whether a receipt is a normal FILTER/skip tombstone."""

    return _state_name(receipt.state) == "normal_absence"


def _is_failure(receipt: Any) -> bool:
    """Return whether a receipt propagates a failed/suppressed cause."""

    return _state_name(receipt.state) in {"failed", "suppressed"}


def _item_node(values: Any, item: Any) -> Any | None:
    """Read an ItemRef binding through the ValueNodeStore narrow interface."""

    for name in ("node_for_item", "get_item", "item_node"):
        method = getattr(values, name, None)
        if method is None:
            continue
        try:
            return method(item)
        except KeyError:
            return None
    items = getattr(values, "_items", None)
    if isinstance(items, dict):
        return items.get(item)
    return None


def _graph_node(graph: Any, node_id: Any) -> Any:
    """Return a compiled node by stable ID."""

    getter = getattr(graph, "node", None)
    if getter is not None:
        return getter(node_id)
    for node in graph.nodes:
        if node.id == node_id:
            return node
    raise KeyError(node_id)


def _grain_get(grains: Any, grain_id: Any) -> Any:
    """Return one grain record through common model store shapes."""

    getter = getattr(grains, "get", None)
    if getter is not None:
        record = getter(grain_id)
        if record is not None:
            return record
    try:
        return grains[grain_id]
    except (KeyError, TypeError):
        pass
    records = getattr(grains, "_grains", None)
    if isinstance(records, dict):
        return records[grain_id]
    raise KeyError(grain_id)


def _grain_get_optional(grains: Any, grain_id: Any) -> Any | None:
    """Return one grain record or ``None`` without broad exception masking."""

    try:
        return _grain_get(grains, grain_id)
    except KeyError:
        return None


def _op_kind(op: Any) -> str:
    """Map a frozen op DTO class to its primitive name."""

    name = type(op).__name__.lower()
    for kind in ("source", "map", "filter", "expand", "reduce"):
        if name == kind or name == f"{kind}op":
            return kind
    explicit = getattr(op, "kind", None)
    if explicit is not None:
        return _state_name(explicit)
    raise PlannerInvariantError(f"unknown compiled op type: {type(op)!r}")


def _require_receipts(latch: InputLatch) -> tuple[Any, ...]:
    """Return a ready latch's receipts with the Optional type removed."""

    if not latch.ready or any(receipt is None for receipt in latch.receipts):
        raise PlannerInvariantError("attempted to resolve an incomplete latch")
    return tuple(latch.receipts)


def _closure_receipts(closure: Any) -> tuple[Any, ...]:
    """Return every ready closure slot in ordinal order."""

    slots = getattr(closure, "slots", None)
    if slots is None:
        getter = getattr(closure, "receipts", None)
        if getter is None:
            raise PlannerInvariantError("ClosureView exposes no receipt slots")
        slots = getter()
    receipts = tuple(slots)
    if len(receipts) != closure.expected or any(
        receipt is None for receipt in receipts
    ):
        raise PlannerInvariantError("ClosureView is ready with incomplete slots")
    return receipts


def _dedupe(values: tuple[Any, ...]) -> tuple[Any, ...]:
    """Canonicalize repeated causes while preserving compiled arrival order."""

    return tuple(dict.fromkeys(values))


def _make_outcome(name: str, **values: Any) -> Any:
    """Construct one model outcome variant with frozen tuple fields."""

    cls = getattr(_sem, name)
    return _construct(cls, **values)


def _canonical_bytes(value: Any) -> bytes:
    """Encode runtime identity inputs without process-random operations."""

    if value is None:
        return b"n"
    if isinstance(value, bool):
        return b"b1" if value else b"b0"
    if isinstance(value, int):
        sign = b"-" if value < 0 else b"+"
        magnitude = abs(value)
        raw = (
            b""
            if magnitude == 0
            else magnitude.to_bytes((magnitude.bit_length() + 7) // 8, "big")
        )
        return b"i" + sign + len(raw).to_bytes(8, "big") + raw
    if isinstance(value, bytes):
        return b"y" + len(value).to_bytes(8, "big") + value
    if isinstance(value, str):
        raw = value.encode("utf-8")
        return b"s" + len(raw).to_bytes(8, "big") + raw
    if isinstance(value, Enum):
        return b"e" + _canonical_bytes(value.value)
    if isinstance(value, (tuple, list)):
        return b"t" + len(value).to_bytes(8, "big") + b"".join(
            _canonical_bytes(part) for part in value
        )
    if isinstance(value, Mapping):
        encoded = sorted(
            (_canonical_bytes(key), _canonical_bytes(item))
            for key, item in value.items()
        )
        return b"m" + b"".join(key + item for key, item in encoded)
    if is_dataclass(value):
        return b"d" + _canonical_bytes(
            tuple(
                (field_info.name, getattr(value, field_info.name))
                for field_info in fields(value)
            )
        )
    raw = getattr(value, "raw", None)
    if isinstance(raw, bytes):
        return b"r" + raw
    raise TypeError(f"unsupported identity component: {type(value)!r}")


@dataclass(frozen=True, slots=True)
class RuntimeSemanticId:
    """Fallback digest wrapper for optional model ID categories."""

    raw: bytes


def _derive_model_id(class_name: str, domain: str, *parts: Any) -> Any:
    """Derive a domain-separated ID and instantiate the model wrapper."""

    cls = getattr(_sem, class_name, RuntimeSemanticId)
    derive = getattr(cls, "derive", None)
    if derive is not None:
        return derive(domain, *parts)
    digest = hashlib.blake2b(
        _canonical_bytes((domain, *parts)),
        digest_size=16,
        person=b"RayOrchMGV3Run",
    ).digest()
    try:
        return cls(digest)
    except TypeError:
        return cls(raw=digest)


def _grain_id(kind: str, run: Any, node: Any, inputs: Any) -> Any:
    """Derive a logical grain ID independent of batch and attempt."""

    if kind == "map":
        return _sem.GrainId.map(run, node, tuple(inputs))
    if kind == "expand":
        parent = inputs[0] if isinstance(inputs, tuple) else inputs
        return _sem.GrainId.expand(run, node, parent)
    if kind == "source":
        return _sem.GrainId.source(run, node, int(inputs))
    return _derive_model_id("GrainId", f"{kind}-grain", run, node, inputs)


def _checked_mul(left: int, right: int) -> int:
    """Multiply non-negative counts with an explicit bounded-domain check."""

    if left < 0 or right < 0:
        raise CreditCapacityError("credit counts must be non-negative")
    result = left * right
    if result > 2**63 - 1:
        raise CreditCapacityError("credit multiplication overflow")
    return result


__all__ = [
    "BatchSelection",
    "ClosureReady",
    "ClosureView",
    "ControlStore",
    "CreditCapacityError",
    "CreditManager",
    "EventPlanner",
    "ExpansionCreditGranted",
    "InputLatch",
    "PlannerInvariantError",
    "ReadyScheduler",
    "ReceiptPublished",
    "ReceiptStore",
    "ScopeTracker",
]
