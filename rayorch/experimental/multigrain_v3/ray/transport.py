"""Ray actor lifecycle, ordered-generator polling, and physical lease cleanup."""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import ray

from ..model.semantics import (
    ActorId,
    BlockId,
    DispatchId,
    LeaseId,
    NodeId,
    RunId,
)
from .protocol import (
    PROTOCOL_VERSION,
    ActorLease,
    FailureManifest,
    Manifest,
    ProtocolValidationError,
    SuccessManifest,
    WorkerDispatch,
    WorkerErrorKind,
    header_for,
    validate_manifest,
    validate_worker_dispatch,
)
from .worker import Worker, WorkerContext

if TYPE_CHECKING:
    from ..runtime.dispatch import PendingDecision, PreparedDispatch


PINNED_RAY_VERSION = "2.50.0"
"""Ray build on which the ordered-generator commit gate is verified."""


class ActorPoolError(RuntimeError):
    """Report an invalid actor-slot or lease transition."""


class ActorCASMismatch(ActorPoolError):
    """Report a stale actor replacement callback without mutating the slot."""


class ReturnArityError(RuntimeError):
    """Report a short, long, or malformed ordered generator stream."""


class TransportClosedError(RuntimeError):
    """Report use of a Ray transport after physical shutdown."""


def ordered_generator_gate_supported(version: str | None = None) -> bool:
    """Return whether the installed Ray build passed the serialization gate."""

    candidate = ray.__version__ if version is None else version
    return candidate.split("+", 1)[0] == PINNED_RAY_VERSION


def _nonnegative_int(value: object, field_name: str) -> int:
    """Validate an integer field without accepting bool."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _actor_id(handle: object) -> ActorId:
    """Derive a backend-neutral ID from a Ray actor handle."""

    ray_id = getattr(handle, "_actor_id", None)
    if ray_id is None:
        return ActorId.new()
    hex_method = getattr(ray_id, "hex", None)
    raw = hex_method() if callable(hex_method) else str(ray_id)
    return ActorId.derive("ray-actor", raw)


@dataclass(slots=True)
class ActorSlot:
    """Track one current actor incarnation and its single pending owner."""

    handle: object
    actor_id: ActorId
    incarnation: int
    pending: int = 0
    active_lease: LeaseId | None = None

    def __post_init__(self) -> None:
        """Validate the single-pending actor-slot invariant."""

        if not isinstance(self.actor_id, ActorId):
            raise TypeError("ActorSlot.actor_id must be an ActorId")
        _nonnegative_int(self.incarnation, "ActorSlot.incarnation")
        _nonnegative_int(self.pending, "ActorSlot.pending")
        if self.pending not in (0, 1):
            raise ValueError("ActorSlot.pending must be zero or one")
        if self.pending == 0 and self.active_lease is not None:
            raise ValueError("an idle actor slot cannot have an active lease")
        if self.pending == 1 and not isinstance(self.active_lease, LeaseId):
            raise ValueError("a pending actor slot requires a LeaseId")


class ActorPool:
    """Own a fixed-size, per-MAP pool with CAS-safe actor replacement."""

    def __init__(
        self,
        node: NodeId,
        context: WorkerContext,
        udf_recipe: object,
        *,
        replicas: int = 1,
        ray_options: Mapping[str, object] | None = None,
        actor_factory: Callable[[], object] | None = None,
        exception_atomic: bool | None = None,
    ) -> None:
        """Configure a pool without admitting more than one task per actor."""

        if not isinstance(node, NodeId):
            raise TypeError("node must be a NodeId")
        if not isinstance(context, WorkerContext):
            raise TypeError("context must be a WorkerContext")
        if context.node != node:
            raise ValueError("WorkerContext.node must match ActorPool.node")
        if isinstance(replicas, bool) or not isinstance(replicas, int):
            raise TypeError("replicas must be an int")
        if replicas <= 0:
            raise ValueError("replicas must be positive")
        options = dict(ray_options or {})
        for key, required in (
            ("max_restarts", 0),
            ("max_task_retries", 0),
            ("max_concurrency", 1),
        ):
            if key in options and options[key] != required:
                raise ValueError(f"{key} must be {required}")
            options[key] = required

        self.node = node
        self.context = context
        self.udf_recipe = udf_recipe
        self.replicas = replicas
        self.ray_options = options
        self.actor_factory = actor_factory
        self.exception_atomic = (
            bool(getattr(udf_recipe, "exception_atomic", False))
            if exception_atomic is None
            else bool(exception_atomic)
        )
        self.slots: list[ActorSlot] = []
        self.next_slot = 0
        self._lease_owners: dict[LeaseId, ActorLease] = {}
        self._closed = False

    def _spawn(self) -> ActorSlot:
        """Create one hidden-restart-disabled actor incarnation."""

        if self.actor_factory is not None:
            handle = self.actor_factory()
        else:
            handle = Worker.options(**self.ray_options).remote(
                self.context,
                self.udf_recipe,
            )
        return ActorSlot(
            handle=handle,
            actor_id=_actor_id(handle),
            incarnation=0,
        )

    def start(self) -> None:
        """Idempotently create every configured actor slot."""

        if self._closed:
            raise ActorPoolError("closed actor pools cannot be restarted")
        while len(self.slots) < self.replicas:
            slot = self._spawn()
            slot.incarnation = 0
            self.slots.append(slot)

    def wait_ready(self, timeout_s: float | None = None) -> tuple[dict[str, int], ...]:
        """Wait until every actor has initialized its persistent UDF instance."""

        self.start()
        refs = [slot.handle.ready.remote() for slot in self.slots]
        if timeout_s is None:
            results = ray.get(refs)
        else:
            if timeout_s <= 0 or not math.isfinite(timeout_s):
                raise ValueError("timeout_s must be finite and positive")
            results = ray.get(refs, timeout=timeout_s)
        return tuple(results)

    def choose(self) -> tuple[int, ActorSlot] | None:
        """Choose an idle slot in round-robin order."""

        if self._closed:
            raise ActorPoolError("actor pool is closed")
        if not self.slots:
            self.start()
        slot_count = len(self.slots)
        for offset in range(slot_count):
            index = (self.next_slot + offset) % slot_count
            slot = self.slots[index]
            if slot.pending == 0:
                self.next_slot = (index + 1) % slot_count
                return index, slot
        return None

    def can_submit(self) -> bool:
        """Return whether at least one current actor incarnation is idle."""

        if self._closed:
            return False
        if not self.slots:
            self.start()
        return any(slot.pending == 0 for slot in self.slots)

    def acquire(self, slot_index: int, lease: ActorLease) -> None:
        """CAS-acquire one exact idle actor incarnation."""

        _nonnegative_int(slot_index, "slot_index")
        if not isinstance(lease, ActorLease):
            raise TypeError("lease must be an ActorLease")
        if slot_index >= len(self.slots):
            raise ActorPoolError("actor slot is outside the pool")
        slot = self.slots[slot_index]
        expected = (
            self.node,
            slot_index,
            slot.incarnation,
            slot.actor_id,
        )
        observed = (
            lease.node,
            lease.slot,
            lease.incarnation,
            lease.actor_id,
        )
        if observed != expected:
            raise ActorPoolError("actor lease does not match the current slot")
        existing = self._lease_owners.get(lease.lease_id)
        if existing is not None:
            if existing == lease:
                raise ActorPoolError("actor lease is already acquired")
            raise ActorPoolError("lease ID is owned by a different actor lease")
        if slot.pending != 0 or slot.active_lease is not None:
            raise ActorPoolError("actor incarnation already has a pending dispatch")
        slot.pending = 1
        slot.active_lease = lease.lease_id
        self._lease_owners[lease.lease_id] = lease

    def release(self, lease: ActorLease) -> bool:
        """Idempotently release an owner without touching a newer incarnation."""

        if not isinstance(lease, ActorLease):
            raise TypeError("lease must be an ActorLease")
        owned = self._lease_owners.get(lease.lease_id)
        if owned is None:
            return False
        if owned != lease:
            raise ActorPoolError("release does not exactly match the owned lease")
        del self._lease_owners[lease.lease_id]

        if lease.slot >= len(self.slots):
            return True
        slot = self.slots[lease.slot]
        current_owner = (
            slot.incarnation == lease.incarnation
            and slot.actor_id == lease.actor_id
            and slot.active_lease == lease.lease_id
        )
        if current_owner:
            slot.pending = 0
            slot.active_lease = None
        return True

    def replace(
        self,
        slot_index: int,
        expected_incarnation: int,
        expected_handle: object,
    ) -> ActorSlot:
        """Replace exactly the expected incarnation or reject the stale CAS."""

        _nonnegative_int(slot_index, "slot_index")
        _nonnegative_int(expected_incarnation, "expected_incarnation")
        if self._closed:
            raise ActorPoolError("actor pool is closed")
        if slot_index >= len(self.slots):
            raise ActorCASMismatch("actor slot no longer exists")
        current = self.slots[slot_index]
        if (
            current.incarnation != expected_incarnation
            or current.handle is not expected_handle
        ):
            raise ActorCASMismatch("actor incarnation changed before replacement")

        replacement = self._spawn()
        replacement.incarnation = expected_incarnation + 1
        # Recheck after spawning so a future concurrent pool cannot kill a newer
        # incarnation from a late callback.
        current = self.slots[slot_index]
        if (
            current.incarnation != expected_incarnation
            or current.handle is not expected_handle
        ):
            ray.kill(replacement.handle, no_restart=True)
            raise ActorCASMismatch("actor incarnation changed during replacement")
        ray.kill(expected_handle, no_restart=True)
        self.slots[slot_index] = replacement
        return replacement

    def close(self) -> None:
        """Idempotently kill every current actor incarnation."""

        if self._closed:
            return
        self._closed = True
        for slot in self.slots:
            try:
                ray.kill(slot.handle, no_restart=True)
            except Exception:
                pass
        self.slots.clear()
        self._lease_owners.clear()


class PendingFinalizeState(Enum):
    """Track resumable owner-keyed physical cleanup."""

    ACTIVE = "active"
    FINALIZING = "finalizing"
    FINALIZED = "finalized"


@dataclass(slots=True)
class PendingDispatch:
    """Hold all physical owners for one submitted dispatch."""

    prepared: Any
    wire: WorkerDispatch
    lease: ActorLease
    actor_handle: object
    actor_id: ActorId
    result_generator: object | None
    manifest_ref: ray.ObjectRef | None
    output_refs: list[ray.ObjectRef]
    input_blocks: tuple[BlockId, ...]
    submitted_at: float
    deadline_at: float | None
    decision: Any | None = None
    semantic_applied: bool = False
    finalize_state: PendingFinalizeState = PendingFinalizeState.ACTIVE
    generator_done_ref: ray.ObjectRef | None = None
    terminal: bool = False

    def __post_init__(self) -> None:
        """Validate immutable identity and deadline fields."""

        if not isinstance(self.wire, WorkerDispatch):
            raise TypeError("wire must be a WorkerDispatch")
        if not isinstance(self.lease, ActorLease):
            raise TypeError("lease must be an ActorLease")
        if self.wire.lease != self.lease:
            raise ValueError("wire and pending actor leases differ")
        if not isinstance(self.actor_id, ActorId) or self.actor_id != self.lease.actor_id:
            raise ValueError("pending actor_id must match the lease")

    @property
    def deadline(self) -> float | None:
        """Return the immutable absolute hard deadline, when configured."""

        return self.deadline_at
        if not isinstance(self.output_refs, list):
            raise TypeError("output_refs must be a list")
        if not isinstance(self.input_blocks, tuple):
            raise TypeError("input_blocks must be a tuple")
        if any(not isinstance(block, BlockId) for block in self.input_blocks):
            raise TypeError("input_blocks must contain BlockId records")
        if not math.isfinite(self.submitted_at):
            raise ValueError("submitted_at must be finite")
        if self.deadline_at is not None and not math.isfinite(self.deadline_at):
            raise ValueError("deadline_at must be finite")

    def freeze_decision(self, decision: "PendingDecision | Any") -> bool:
        """Install one semantic decision once, accepting idempotent replay."""

        if decision is None:
            raise TypeError("decision must not be None")
        if self.decision is None:
            self.decision = decision
            return True
        if self.decision != decision:
            raise RuntimeError("pending dispatch already has a different decision")
        return False

    def mark_semantic_applied(self) -> bool:
        """Record an at-most-once semantic apply after a frozen decision."""

        if self.decision is None:
            raise RuntimeError("freeze a decision before marking semantic apply")
        if self.semantic_applied:
            return False
        self.semantic_applied = True
        return True


class InfrastructureFailureKind(Enum):
    """Classify physical failures without selecting semantic retry policy."""

    ACTOR_DIED = "actor_died"
    NODE_DIED = "node_died"
    DEADLINE = "deadline"
    CANCELLED = "cancelled"
    INPUT_OBJECT_LOST = "input_object_lost"
    RETURN_SERIALIZATION = "return_serialization"
    RETURN_ARITY = "return_arity"


@dataclass(frozen=True, slots=True)
class InfrastructureFailure:
    """Describe a transport-observed failure and its original exception."""

    kind: InfrastructureFailureKind
    error_type: str
    message: str
    cause: BaseException | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Validate stable failure metadata."""

        if not isinstance(self.kind, InfrastructureFailureKind):
            raise TypeError("kind must be an InfrastructureFailureKind")
        if not isinstance(self.error_type, str) or not self.error_type:
            raise ValueError("error_type must be a non-empty string")
        if not isinstance(self.message, str):
            raise TypeError("message must be a string")


@dataclass(frozen=True, slots=True)
class RayCompletion:
    """Return either a validated manifest or one physical failure."""

    pending: PendingDispatch
    observed_at: float
    manifest: Manifest | None = None
    failure: InfrastructureFailure | None = None

    def __post_init__(self) -> None:
        """Require exactly one successful observation or failure."""

        if not isinstance(self.pending, PendingDispatch):
            raise TypeError("pending must be a PendingDispatch")
        if not math.isfinite(self.observed_at):
            raise ValueError("observed_at must be finite")
        if (self.manifest is None) == (self.failure is None):
            raise ValueError("RayCompletion requires exactly one terminal payload")


def _exception_name(exc: BaseException) -> str:
    """Return a stable qualified exception class name."""

    cls = type(exc)
    return f"{cls.__module__}.{cls.__qualname__}"


def classify_ray_failure(exc: BaseException) -> InfrastructureFailureKind:
    """Map Ray and local gate exceptions onto architecture failure kinds."""

    exceptions = ray.exceptions
    actor_error = getattr(exceptions, "RayActorError", ())
    node_error = getattr(exceptions, "NodeDiedError", ())
    cancelled_error = getattr(exceptions, "TaskCancelledError", ())
    object_lost_errors = tuple(
        candidate
        for candidate in (
            getattr(exceptions, "ObjectLostError", None),
            getattr(exceptions, "OwnerDiedError", None),
        )
        if isinstance(candidate, type)
    )
    candidates: list[BaseException] = [exc]
    for candidate in (
        getattr(exc, "cause", None),
        getattr(exc, "__cause__", None),
    ):
        if isinstance(candidate, BaseException) and candidate not in candidates:
            candidates.append(candidate)
    as_cause = getattr(exc, "as_instanceof_cause", None)
    if callable(as_cause):
        try:
            candidate = as_cause()
        except Exception:
            candidate = None
        if isinstance(candidate, BaseException) and candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        if isinstance(actor_error, type) and isinstance(candidate, actor_error):
            return InfrastructureFailureKind.ACTOR_DIED
        if isinstance(node_error, type) and isinstance(candidate, node_error):
            return InfrastructureFailureKind.NODE_DIED
        if isinstance(cancelled_error, type) and isinstance(
            candidate, cancelled_error
        ):
            return InfrastructureFailureKind.CANCELLED
        if object_lost_errors and isinstance(candidate, object_lost_errors):
            return InfrastructureFailureKind.INPUT_OBJECT_LOST
        if isinstance(candidate, ReturnArityError):
            return InfrastructureFailureKind.RETURN_ARITY
    # A worker generator is required to encode every UDF/contract exception in
    # a FailureManifest.  Any uncaught task error is therefore a failed return
    # stream (most importantly a data-yield serialization failure).
    return InfrastructureFailureKind.RETURN_SERIALIZATION


class RayTransport:
    """Expose narrow submit/poll/cancel/close operations over MAP actor pools."""

    def __init__(
        self,
        pools: Mapping[NodeId, ActorPool] | Iterable[ActorPool],
        *,
        resolve_input: Callable[[BlockId], ray.ObjectRef] | None = None,
        release_input: Callable[[DispatchId, BlockId], None] | None = None,
        release_reservation: Callable[[DispatchId], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        require_pinned_ray: bool = True,
    ) -> None:
        """Start pools and configure physical owner release callbacks."""

        if require_pinned_ray and not ordered_generator_gate_supported():
            raise RuntimeError(
                "ordered generator commit gate is unverified on Ray "
                f"{ray.__version__}; expected {PINNED_RAY_VERSION}"
            )
        if isinstance(pools, Mapping):
            pool_items = tuple(pools.items())
        else:
            pool_items = tuple((pool.node, pool) for pool in pools)
        self._pools: dict[NodeId, ActorPool] = {}
        for node, pool in pool_items:
            if not isinstance(node, NodeId) or not isinstance(pool, ActorPool):
                raise TypeError("pools must map NodeId to ActorPool")
            if node != pool.node:
                raise ValueError("pool mapping key does not match pool.node")
            if node in self._pools:
                raise ValueError("duplicate ActorPool for a MAP node")
            self._pools[node] = pool
        if not self._pools:
            raise ValueError("RayTransport requires at least one ActorPool")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._resolve_input = resolve_input
        self._release_input = release_input
        self._release_reservation = release_reservation
        self._clock = clock
        self._pending: dict[DispatchId, PendingDispatch] = {}
        self._submitted: set[DispatchId] = set()
        self._ready: deque[RayCompletion] = deque()
        self._input_owners: set[tuple[DispatchId, BlockId]] = set()
        self._actor_owners: set[LeaseId] = set()
        self._reservation_owners: set[DispatchId] = set()
        self._deadline_owners: set[DispatchId] = set()
        self._generator_owners: set[DispatchId] = set()
        self._closed = False
        for pool in self._pools.values():
            pool.start()

    def wait_ready(self, timeout_s: float | None = None) -> dict[NodeId, tuple[dict[str, int], ...]]:
        """Wait for all MAP pools to finish actor and UDF initialization."""

        self._ensure_open()
        started = float(self._clock())
        ready: dict[NodeId, tuple[dict[str, int], ...]] = {}
        for node, pool in self._pools.items():
            remaining = None
            if timeout_s is not None:
                elapsed = float(self._clock()) - started
                remaining = timeout_s - elapsed
                if remaining <= 0:
                    raise TimeoutError("RayTransport actor readiness timed out")
            ready[node] = pool.wait_ready(remaining)
        return ready

    def _ensure_open(self) -> None:
        """Reject operations after transport shutdown."""

        if self._closed:
            raise TransportClosedError("RayTransport is closed")

    def can_submit(self, node: NodeId) -> bool:
        """Return whether a MAP node has an idle actor without taking a lease."""

        self._ensure_open()
        pool = self._pools.get(node)
        return pool is not None and pool.can_submit()

    @property
    def pending_count(self) -> int:
        """Return the number of submitted dispatches awaiting finalization."""

        return len(self._pending)

    def has_pending(self) -> bool:
        """Return whether any dispatch still owns physical transport resources."""

        return bool(self._pending)

    def pending(
        self,
        run: RunId | None = None,
    ) -> tuple[PendingDispatch, ...]:
        """Return a stable snapshot of pending dispatches, optionally by run."""

        values = tuple(self._pending.values())
        if run is None:
            return values
        if not isinstance(run, RunId):
            raise TypeError("run must be a RunId or None")
        return tuple(pending for pending in values if pending.wire.run == run)

    def expire_deadlines(
        self,
        now: float | None = None,
    ) -> tuple[RayCompletion, ...]:
        """Expire due dispatches and return their newly queued observations."""

        self._ensure_open()
        observed = float(self._clock() if now is None else now)
        if not math.isfinite(observed):
            raise ValueError("deadline observation time must be finite")
        before = len(self._ready)
        self._expire_deadlines(observed)
        added = len(self._ready) - before
        completions = tuple(
            self._ready.pop() for _ in range(max(0, added))
        )
        return tuple(reversed(completions))

    def _inputs(
        self,
        prepared: Any,
        input_refs: Sequence[ray.ObjectRef] | None,
    ) -> tuple[ray.ObjectRef, ...]:
        """Resolve refs once in exact ``PreparedDispatch.block_ids`` order."""

        block_ids = tuple(getattr(prepared, "block_ids", ()))
        if any(not isinstance(block, BlockId) for block in block_ids):
            raise TypeError("PreparedDispatch.block_ids must contain BlockId records")
        if input_refs is None:
            if self._resolve_input is None:
                raise ValueError("input_refs or resolve_input is required")
            refs = tuple(self._resolve_input(block) for block in block_ids)
        else:
            refs = tuple(input_refs)
        if len(refs) != len(block_ids):
            raise ValueError("input ref count does not match ordered block_ids")
        if any(not isinstance(ref, ray.ObjectRef) for ref in refs):
            raise TypeError("input refs must contain Ray ObjectRef values")
        return refs

    def submit(
        self,
        prepared: "PreparedDispatch | Any",
        input_refs: Sequence[ray.ObjectRef] | None = None,
        *,
        deadline_at: float | None = None,
        timeout_s: float | None = None,
    ) -> PendingDispatch | None:
        """Lease one idle actor and submit a dispatch without blocking on UDF work."""

        self._ensure_open()
        dispatch_id = getattr(prepared, "id", None)
        run = getattr(prepared, "run", None)
        node = getattr(prepared, "node", None)
        entries = getattr(prepared, "wire_entries", None)
        attempts = tuple(getattr(prepared, "attempts", ()))
        if not isinstance(dispatch_id, DispatchId):
            raise TypeError("PreparedDispatch.id must be a DispatchId")
        if not isinstance(run, RunId):
            raise TypeError("PreparedDispatch.run must be a RunId")
        if not isinstance(node, NodeId):
            raise TypeError("PreparedDispatch.node must be a NodeId")
        if not isinstance(entries, tuple) or not entries:
            raise ValueError("PreparedDispatch.wire_entries must be non-empty")
        if tuple(entry.token for entry in entries) != attempts:
            raise ValueError("PreparedDispatch attempts and wire entries differ")
        if dispatch_id in self._submitted:
            raise ValueError("dispatch has already been submitted")
        pool = self._pools.get(node)
        if pool is None:
            raise ValueError("no ActorPool is configured for dispatch node")
        chosen = pool.choose()
        if chosen is None:
            return None

        now = float(self._clock())
        if not math.isfinite(now):
            raise ValueError("transport clock returned a non-finite value")
        if deadline_at is not None and timeout_s is not None:
            raise ValueError("specify deadline_at or timeout_s, not both")
        if timeout_s is not None:
            if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
                raise TypeError("timeout_s must be a number")
            if not math.isfinite(timeout_s) or timeout_s < 0:
                raise ValueError("timeout_s must be finite and non-negative")
            deadline_at = now + float(timeout_s)
        if deadline_at is not None:
            if isinstance(deadline_at, bool) or not isinstance(
                deadline_at, (int, float)
            ):
                raise TypeError("deadline_at must be a number")
            deadline_at = float(deadline_at)
            if not math.isfinite(deadline_at):
                raise ValueError("deadline_at must be finite")

        refs = self._inputs(prepared, input_refs)
        slot_index, slot = chosen
        lease = ActorLease(
            node=node,
            slot=slot_index,
            incarnation=slot.incarnation,
            actor_id=slot.actor_id,
            lease_id=LeaseId.new(),
        )
        wire = WorkerDispatch(
            protocol_version=PROTOCOL_VERSION,
            graph_fingerprint=pool.context.graph_fingerprint,
            run=run,
            dispatch=dispatch_id,
            node=node,
            lease=lease,
            entries=entries,
        )
        validate_worker_dispatch(
            wire,
            input_ref_count=len(refs),
            role_count=len(pool.context.call_schema.parameters),
        )

        pool.acquire(slot_index, lease)
        try:
            generator = slot.handle.execute.remote(wire, *refs)
            if not all(
                hasattr(generator, attribute)
                for attribute in ("next_ready", "is_finished", "completed")
            ):
                raise TypeError("Worker.execute did not return ObjectRefGenerator")
            done_ref = generator.completed()
            pending = PendingDispatch(
                prepared=prepared,
                wire=wire,
                lease=lease,
                actor_handle=slot.handle,
                actor_id=slot.actor_id,
                result_generator=generator,
                manifest_ref=None,
                output_refs=[],
                input_blocks=tuple(prepared.block_ids),
                submitted_at=now,
                deadline_at=deadline_at,
                generator_done_ref=done_ref,
            )
        except Exception:
            pool.release(lease)
            raise

        self._pending[dispatch_id] = pending
        self._submitted.add(dispatch_id)
        self._actor_owners.add(lease.lease_id)
        self._reservation_owners.add(dispatch_id)
        self._generator_owners.add(dispatch_id)
        if deadline_at is not None:
            self._deadline_owners.add(dispatch_id)
        for block in dict.fromkeys(pending.input_blocks):
            self._input_owners.add((dispatch_id, block))
        return pending

    def _replace_pending_actor(self, pending: PendingDispatch) -> None:
        """Quarantine only the incarnation captured by a pending lease."""

        pool = self._pools[pending.lease.node]
        try:
            pool.replace(
                pending.lease.slot,
                pending.lease.incarnation,
                pending.actor_handle,
            )
        except ActorCASMismatch:
            # A newer completion already replaced it.  The stale callback must
            # never kill the new incarnation.
            return

    def _infrastructure_completion(
        self,
        pending: PendingDispatch,
        kind: InfrastructureFailureKind,
        exc: BaseException,
        observed_at: float,
        *,
        replace_actor: bool = True,
        enqueue: bool = True,
    ) -> RayCompletion:
        """Freeze one physical failure and quarantine its exact actor if needed."""

        if pending.terminal:
            raise RuntimeError("pending dispatch is already terminal")
        if replace_actor:
            self._replace_pending_actor(pending)
        pending.terminal = True
        completion = RayCompletion(
            pending=pending,
            observed_at=observed_at,
            failure=InfrastructureFailure(
                kind=kind,
                error_type=_exception_name(exc),
                message=str(exc),
                cause=exc,
            ),
        )
        if enqueue:
            self._ready.append(completion)
        return completion

    def _consume_ready_refs(
        self,
        pending: PendingDispatch,
        output_count: int,
        observed_at: float,
    ) -> bool:
        """Consume every currently available generator ref without ray.get(data)."""

        generator = pending.result_generator
        if generator is None:
            return False
        try:
            while generator.next_ready():
                try:
                    ref = next(generator)
                except StopIteration:
                    break
                if not isinstance(ref, ray.ObjectRef):
                    raise ReturnArityError("generator yielded a non-ObjectRef handle")
                if len(pending.output_refs) < output_count:
                    pending.output_refs.append(ref)
                elif pending.manifest_ref is None:
                    pending.manifest_ref = ref
                else:
                    raise ReturnArityError(
                        "worker generator yielded more than P + 1 values"
                    )
        except BaseException as exc:
            kind = classify_ray_failure(exc)
            self._infrastructure_completion(
                pending,
                kind,
                exc,
                observed_at,
            )
            return True
        return pending.terminal

    def _progress(self, pending: PendingDispatch, observed_at: float) -> None:
        """Advance one stream through completion and manifest validation."""

        if pending.terminal:
            return
        pool = self._pools[pending.lease.node]
        output_count = len(pool.context.output_schema)
        if self._consume_ready_refs(pending, output_count, observed_at):
            return
        done_ref = pending.generator_done_ref
        if done_ref is None:
            exc = ReturnArityError("worker generator has no completion ref")
            self._infrastructure_completion(
                pending,
                InfrastructureFailureKind.RETURN_ARITY,
                exc,
                observed_at,
            )
            return
        ready, _ = ray.wait([done_ref], num_returns=1, timeout=0)
        if not ready:
            return
        try:
            ray.get(done_ref)
        except BaseException as exc:
            self._infrastructure_completion(
                pending,
                classify_ray_failure(exc),
                exc,
                observed_at,
            )
            return

        # Completion means every yield has been produced; drain again to detect
        # both short and overlong streams before observing the manifest.
        if self._consume_ready_refs(pending, output_count, observed_at):
            return
        generator = pending.result_generator
        if generator is None or not generator.is_finished():
            return
        if len(pending.output_refs) != output_count or pending.manifest_ref is None:
            exc = ReturnArityError(
                "worker generator ended before yielding P outputs and a manifest"
            )
            self._infrastructure_completion(
                pending,
                InfrastructureFailureKind.RETURN_ARITY,
                exc,
                observed_at,
            )
            return

        try:
            manifest = ray.get(pending.manifest_ref)
            expected_header = header_for(pending.wire)
            validate_manifest(
                manifest,
                max_bytes=pool.context.max_manifest_bytes,
                expected_header=expected_header,
                expected_output_count=output_count,
            )
        except ProtocolValidationError as exc:
            self._infrastructure_completion(
                pending,
                InfrastructureFailureKind.RETURN_ARITY,
                exc,
                observed_at,
            )
            return
        except BaseException as exc:
            self._infrastructure_completion(
                pending,
                classify_ray_failure(exc),
                exc,
                observed_at,
            )
            return

        if (
            isinstance(manifest, FailureManifest)
            and manifest.kind in (WorkerErrorKind.GENERIC_UDF, WorkerErrorKind.CONTRACT)
            and not pool.exception_atomic
        ):
            self._replace_pending_actor(pending)
        pending.terminal = True
        self._ready.append(
            RayCompletion(
                pending=pending,
                observed_at=observed_at,
                manifest=manifest,
            )
        )

    def _expire_deadlines(self, now: float) -> None:
        """Expire hard deadlines before any result can be committed."""

        for pending in tuple(self._pending.values()):
            if (
                not pending.terminal
                and pending.deadline_at is not None
                and now >= pending.deadline_at
            ):
                exc = TimeoutError(
                    f"dispatch {pending.wire.dispatch.hex()} exceeded hard deadline"
                )
                self._infrastructure_completion(
                    pending,
                    InfrastructureFailureKind.DEADLINE,
                    exc,
                    now,
                )

    def _poll_wait_budget(self, timeout: float, now: float) -> float:
        """Cap a poll wait by the nearest active hard deadline."""

        nearest = min(
            (
                pending.deadline_at
                for pending in self._pending.values()
                if not pending.terminal and pending.deadline_at is not None
            ),
            default=None,
        )
        if nearest is None:
            return timeout
        return max(0.0, min(timeout, nearest - now))

    def poll(
        self,
        timeout: float = 0.0,
        *,
        max_completions: int | None = None,
    ) -> tuple[RayCompletion, ...]:
        """Expire deadlines first, then return bounded terminal observations."""

        self._ensure_open()
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("timeout must be a number")
        timeout = float(timeout)
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and non-negative")
        if max_completions is not None:
            _nonnegative_int(max_completions, "max_completions")
            if max_completions == 0:
                return ()

        now = float(self._clock())
        self._expire_deadlines(now)
        for pending in tuple(self._pending.values()):
            self._progress(pending, now)

        if not self._ready and timeout > 0:
            active_done_refs = [
                pending.generator_done_ref
                for pending in self._pending.values()
                if not pending.terminal and pending.generator_done_ref is not None
            ]
            wait_budget = self._poll_wait_budget(timeout, now)
            if active_done_refs and wait_budget > 0:
                ray.wait(
                    active_done_refs,
                    num_returns=1,
                    timeout=wait_budget,
                )
            elif wait_budget > 0:
                time.sleep(wait_budget)
            now = float(self._clock())
            self._expire_deadlines(now)
            for pending in tuple(self._pending.values()):
                self._progress(pending, now)

        limit = len(self._ready) if max_completions is None else max_completions
        completions = tuple(self._ready.popleft() for _ in range(min(limit, len(self._ready))))
        return completions

    def cancel(
        self,
        target: RunId | DispatchId | PendingDispatch | None = None,
    ) -> tuple[RayCompletion, ...]:
        """Kill matching exact incarnations and return cancellation observations."""

        self._ensure_open()
        now = float(self._clock())
        selected: list[PendingDispatch] = []
        for pending in self._pending.values():
            if pending.terminal:
                continue
            if target is None:
                selected.append(pending)
            elif isinstance(target, PendingDispatch) and pending is target:
                selected.append(pending)
            elif isinstance(target, DispatchId) and pending.wire.dispatch == target:
                selected.append(pending)
            elif isinstance(target, RunId) and pending.wire.run == target:
                selected.append(pending)
        completions = []
        for pending in selected:
            exc = RuntimeError(f"dispatch {pending.wire.dispatch.hex()} cancelled")
            completions.append(
                self._infrastructure_completion(
                    pending,
                    InfrastructureFailureKind.CANCELLED,
                    exc,
                    now,
                    enqueue=False,
                )
            )
        return tuple(completions)

    def _may_finalize(self, pending: PendingDispatch, force: bool) -> bool:
        """Check the semantic-before-physical handoff invariant."""

        if force or pending.semantic_applied:
            return True
        decision_kind = getattr(getattr(pending.decision, "kind", None), "value", None)
        return decision_kind == "stale"

    def finalize_physical(
        self,
        pending: PendingDispatch,
        *,
        force: bool = False,
    ) -> None:
        """Idempotently release every owner key after semantic disposition."""

        if not isinstance(pending, PendingDispatch):
            raise TypeError("pending must be a PendingDispatch")
        if pending.finalize_state is PendingFinalizeState.FINALIZED:
            return
        if not pending.terminal and not force:
            raise RuntimeError("cannot finalize a non-terminal dispatch")
        if not self._may_finalize(pending, force):
            raise RuntimeError(
                "semantic disposition must be applied before physical finalization"
            )
        registered = self._pending.get(pending.wire.dispatch)
        if registered is not pending:
            if pending.finalize_state is PendingFinalizeState.FINALIZED:
                return
            raise RuntimeError("pending dispatch is not owned by this transport")
        pending.finalize_state = PendingFinalizeState.FINALIZING
        first_error: BaseException | None = None
        dispatch_id = pending.wire.dispatch

        for owner in tuple(self._input_owners):
            if owner[0] != dispatch_id:
                continue
            try:
                if self._release_input is not None:
                    self._release_input(*owner)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            else:
                self._input_owners.discard(owner)

        if dispatch_id in self._generator_owners:
            try:
                # Ray 2.50 exposes ``ObjectRefGenerator.close`` but raises
                # NotImplementedError.  Terminal tasks have either completed
                # or had their exact actor incarnation killed, so dropping all
                # local handles is the supported physical release operation.
                pending.result_generator = None
                pending.generator_done_ref = None
                pending.manifest_ref = None
                pending.output_refs.clear()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            else:
                self._generator_owners.discard(dispatch_id)

        if pending.lease.lease_id in self._actor_owners:
            try:
                self._pools[pending.lease.node].release(pending.lease)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            else:
                self._actor_owners.discard(pending.lease.lease_id)

        if dispatch_id in self._reservation_owners:
            try:
                if self._release_reservation is not None:
                    self._release_reservation(dispatch_id)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            else:
                self._reservation_owners.discard(dispatch_id)

        self._deadline_owners.discard(dispatch_id)
        if not any(owner[0] == dispatch_id for owner in self._input_owners) and all(
            (
                dispatch_id not in self._generator_owners,
                pending.lease.lease_id not in self._actor_owners,
                dispatch_id not in self._reservation_owners,
                dispatch_id not in self._deadline_owners,
            )
        ):
            pending.finalize_state = PendingFinalizeState.FINALIZED
            self._pending.pop(dispatch_id, None)
            self._submitted.discard(dispatch_id)
            self._ready = deque(
                completion
                for completion in self._ready
                if completion.pending is not pending
            )
        if first_error is not None:
            raise first_error

    def close(self) -> None:
        """Idempotently cancel work, release owners, and kill every actor pool."""

        if self._closed:
            return
        for pending in tuple(self._pending.values()):
            if not pending.terminal:
                now = float(self._clock())
                exc = RuntimeError(
                    f"dispatch {pending.wire.dispatch.hex()} cancelled by close"
                )
                self._infrastructure_completion(
                    pending,
                    InfrastructureFailureKind.CANCELLED,
                    exc,
                    now,
                    enqueue=False,
                )
            try:
                self.finalize_physical(pending, force=True)
            except Exception:
                # Continue best-effort cleanup; actor shutdown below is the final
                # physical safety boundary during session teardown.
                pass
        for pool in self._pools.values():
            pool.close()
        self._ready.clear()
        self._pending.clear()
        self._submitted.clear()
        self._input_owners.clear()
        self._actor_owners.clear()
        self._reservation_owners.clear()
        self._deadline_owners.clear()
        self._generator_owners.clear()
        self._closed = True
