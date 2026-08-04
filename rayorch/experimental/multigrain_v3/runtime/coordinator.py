"""Bounded run/session lifecycle with a backend-neutral transport protocol.

``InlineTransport`` is a deterministic local implementation used by unit
tests and environments without Ray.  A later ``RayTransport`` can implement
the same submission/polling boundary without importing coordinator state into
workers.
"""

from __future__ import annotations

import hashlib
import time
from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol

from ..model import semantics as _sem
from ..model import state as _state
from ..model.graph import CompiledGraph, MapOp, SourceOp, thaw_config_value
from .dispatch import (
    BatchSelection,
    CommitDelta,
    DispatchInvariantError,
    LocalSlotTake,
    LocalTransaction,
    LocalWireList,
    ManifestDeltaBuilder,
    PendingDecision,
    PendingDisposition,
    PreparedOutcome,
    RefInterner,
    prepare_dispatch,
    resolve_storage_gather,
)
from .planner import (
    CreditManager,
    EventPlanner,
    ReadyScheduler,
    ReceiptStore,
    ScopeTracker,
)


class CoordinatorInvariantError(RuntimeError):
    """Report a run-loop or transport contract violation."""


class RunAborted(RuntimeError):
    """Surface a run-scoped abort from the lazy ``RunStream``."""

    def __init__(self, cause: BaseException) -> None:
        """Retain the original failure while presenting a stable public type."""

        super().__init__(f"run aborted: {cause}")
        self.cause = cause


@dataclass(frozen=True, slots=True)
class RuntimeRunId:
    """Represent a local 128-bit run identity."""

    raw: bytes


@dataclass(frozen=True, slots=True)
class RuntimeRootId:
    """Represent a deterministic root identity within one run."""

    raw: bytes


@dataclass(frozen=True, slots=True)
class RuntimeBatchId:
    """Represent an owner key for one source-level inline batch."""

    raw: bytes


@dataclass(frozen=True, slots=True)
class _LocalRuntimeLimits:
    """Supply conservative coordinator limits when model limits are omitted."""

    max_active_roots: int = 16
    max_buffered_results: int = 16
    max_local_events_per_turn: int = 256
    max_completions_per_turn: int = 64


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """Pair one lazily consumed source payload with its monotonic sequence."""

    source_seq: int
    value: Any


@dataclass(frozen=True, slots=True)
class EndOfSource:
    """Mark that a ``SourceAdapter`` has permanently exhausted its iterator."""


END_OF_SOURCE = EndOfSource()


class SourceAdapter:
    """Consume an arbitrary iterable lazily and assign stable source ordinals."""

    def __init__(self, source: Iterable[Any]) -> None:
        """Wrap ``source`` without materializing it."""

        self.iterator = iter(source)
        self.next_seq = 0
        self.exhausted = False

    def next_record(self) -> SourceRecord | EndOfSource:
        """Consume at most one source element."""

        if self.exhausted:
            return END_OF_SOURCE
        try:
            value = next(self.iterator)
        except StopIteration:
            self.exhausted = True
            return END_OF_SOURCE
        record = SourceRecord(self.next_seq, value)
        self.next_seq += 1
        return record

    def close(self) -> None:
        """Close the underlying iterator when it exposes ``close``."""

        close = getattr(self.iterator, "close", None)
        if close is not None:
            close()
        self.exhausted = True


class RootPhase(Enum):
    """Describe the coordinator-visible lifecycle of one source root."""

    ACTIVE = "active"
    DELIVERABLE = "deliverable"
    RECLAIMED = "reclaimed"


@dataclass(slots=True)
class RootContext:
    """Index all run-loop state owned by one admitted source root."""

    id: Any
    source_seq: int
    phase: RootPhase
    source_record: SourceRecord
    grain_ids: set[Any] = field(default_factory=set)
    scope_ids: set[Any] = field(default_factory=set)
    item_ids: set[Any] = field(default_factory=set)
    latch_ids: set[Any] = field(default_factory=set)
    ready_count: int = 0
    inflight_count: int = 0
    pending_local_events: int = 0
    final_receipts: dict[str, Any | None] = field(default_factory=dict)
    submitted: bool = False
    result_enqueued: bool = False

    @property
    def quiescent(self) -> bool:
        """Return whether this local root has no remaining execution work."""

        return (
            self.submitted
            and self.ready_count == 0
            and self.inflight_count == 0
            and self.pending_local_events == 0
        )


class ResultLeafState(Enum):
    """Classify one named graph output at delivery time."""

    PRESENT = "present"
    ABSENT = "absent"
    FAILED = "failed"


class RootResultStatus(Enum):
    """Summarize all final leaves for one source root."""

    SUCCESS = "success"
    DROPPED = "dropped"
    FAILED = "failed"


@dataclass(slots=True)
class DetachedValue:
    """Own final refs and a wire gather independently of coordinator stores."""

    refs: tuple[Any, ...]
    tree: Any
    resolver: Callable[[tuple[Any, ...], Any], Any] | None = None
    closed: bool = False
    _local_value: Any = field(default=None, repr=False)
    _has_local_value: bool = field(default=False, repr=False)

    @classmethod
    def from_local(cls, value: Any) -> "DetachedValue":
        """Create a detached wrapper for an already local final value."""

        return cls(
            refs=(),
            tree=None,
            _local_value=value,
            _has_local_value=True,
        )

    def get(self) -> Any:
        """Materialize this final value only when explicitly requested."""

        if self.closed:
            raise RuntimeError("DetachedValue is closed")
        if self._has_local_value:
            return self._local_value
        if self.resolver is not None:
            return self.resolver(self.refs, self.tree)
        return _resolve_wire_tree(self.refs, self.tree)

    def close(self) -> None:
        """Release owned ref handles idempotently."""

        if self.closed:
            return
        self.refs = ()
        self._local_value = None
        self._has_local_value = False
        self.closed = True

    def __enter__(self) -> "DetachedValue":
        """Return this value for context-manager use."""

        if self.closed:
            raise RuntimeError("DetachedValue is closed")
        return self

    def __exit__(self, *_exc: Any) -> None:
        """Release refs when leaving a context manager."""

        self.close()

    def __del__(self) -> None:
        """Best-effort fallback release for abandoned detached values."""

        try:
            self.close()
        except BaseException:
            pass


@dataclass(frozen=True, slots=True)
class ResultLeaf:
    """Hold one final output's presence or independent failure summary."""

    state: ResultLeafState
    value: DetachedValue | None = None
    failure: Any | None = None


@dataclass(frozen=True, slots=True)
class RootResult:
    """Deliver all named outputs for one source sequence."""

    root: Any
    source_seq: int
    status: RootResultStatus
    outputs: Mapping[str, ResultLeaf]

    def unwrap(self) -> dict[str, Any]:
        """Materialize PRESENT leaves and map ABSENT leaves to ``None``."""

        result: dict[str, Any] = {}
        for name, leaf in self.outputs.items():
            if leaf.state is ResultLeafState.FAILED:
                raise RuntimeError(f"output {name!r} failed: {leaf.failure}")
            result[name] = None if leaf.value is None else leaf.value.get()
        return result


class ResultBuffer:
    """Bound undelivered ``RootResult`` objects and own their detached refs."""

    def __init__(self, max_results: int) -> None:
        """Create an empty hard-count result buffer."""

        if max_results <= 0:
            raise ValueError("max_results must be positive")
        self.max_results = max_results
        self._queue: deque[RootResult] = deque()
        self.closed = False

    def __len__(self) -> int:
        """Return the number of buffered, not-yet-transferred results."""

        return len(self._queue)

    @property
    def full(self) -> bool:
        """Return whether another result would violate the hard count."""

        return len(self._queue) >= self.max_results

    def put(self, result: RootResult) -> None:
        """Transfer one result's ownership into the buffer."""

        if self.closed:
            raise RuntimeError("ResultBuffer is closed")
        if self.full:
            raise BufferError("ResultBuffer is full")
        self._queue.append(result)

    def take(self) -> RootResult:
        """Atomically transfer the oldest result to the caller."""

        if not self._queue:
            raise IndexError("ResultBuffer is empty")
        return self._queue.popleft()

    def close(self) -> None:
        """Close all queued detached values and reject future puts."""

        if self.closed:
            return
        while self._queue:
            _close_root_result(self._queue.popleft())
        self.closed = True


@dataclass(frozen=True, slots=True)
class RootBatch:
    """Submit source records from multiple roots as one transport unit."""

    id: Any
    run: Any
    records: tuple[tuple[Any, SourceRecord], ...]


@dataclass(frozen=True, slots=True)
class TransportCompletion:
    """Return one completed root batch or one batch-wide exception."""

    batch: RootBatch
    outputs: tuple[Any, ...] = ()
    error: BaseException | None = None
    observed_at: float | None = None


class Transport(Protocol):
    """Define the coordinator-facing boundary for local or Ray execution."""

    def submit(self, dispatch: Any, *args: Any, **kwargs: Any) -> Any | None:
        """Submit one prepared unit, returning ``None`` for backpressure."""

    def poll(
        self,
        timeout: float = 0.0,
        *,
        max_completions: int | None = None,
    ) -> tuple[Any, ...]:
        """Return bounded completed units without performing semantic writes."""

    def close(self) -> None:
        """Release graph-bound physical resources idempotently."""


class InlineTransport:
    """Execute whole root batches synchronously without importing Ray."""

    def __init__(
        self,
        executor: Callable[[tuple[Any, ...]], Iterable[Any]] | None = None,
        *,
        max_pending: int = 64,
    ) -> None:
        """Create a deterministic local transport with bounded completions."""

        if max_pending <= 0:
            raise ValueError("max_pending must be positive")
        self.executor = executor
        self.max_pending = max_pending
        self.graph: Any | None = None
        self._completions: deque[TransportCompletion] = deque()
        self._closed = False
        self.submitted_batches: list[tuple[int, ...]] = []

    def start(self, graph: Any) -> None:
        """Bind one graph, allowing reuse across sequential session runs."""

        if self._closed:
            raise RuntimeError("InlineTransport is closed")
        if self.graph is not None and self.graph is not graph and self.graph != graph:
            raise CoordinatorInvariantError(
                "transport session cannot switch CompiledGraph"
            )
        self.graph = graph

    def submit(
        self,
        dispatch: Any,
        *_args: Any,
        **_kwargs: Any,
    ) -> bool:
        """Execute one ``RootBatch`` and queue a completion immediately."""

        if self._closed:
            raise RuntimeError("InlineTransport is closed")
        if not isinstance(dispatch, RootBatch):
            raise TypeError("InlineTransport expects RootBatch submissions")
        if len(self._completions) >= self.max_pending:
            return False
        self.submitted_batches.append(
            tuple(record.source_seq for _, record in dispatch.records)
        )
        values = tuple(record.value for _, record in dispatch.records)
        try:
            output = self._execute(values)
            outputs = self._normalize_outputs(output, len(values))
            completion = TransportCompletion(
                batch=dispatch,
                outputs=outputs,
                observed_at=time.monotonic(),
            )
        except BaseException as error:
            completion = TransportCompletion(
                batch=dispatch,
                error=error,
                observed_at=time.monotonic(),
            )
        self._completions.append(completion)
        return True

    def poll(
        self,
        timeout: float = 0.0,
        *,
        max_completions: int | None = None,
    ) -> tuple[TransportCompletion, ...]:
        """Return queued synchronous completions in submission order."""

        del timeout
        max_items = 1 if max_completions is None else max_completions
        if max_items < 0:
            raise ValueError("max_completions must be non-negative")
        return tuple(
            self._completions.popleft()
            for _ in range(min(max_items, len(self._completions)))
        )

    def expire_deadlines(self, now: float) -> tuple[TransportCompletion, ...]:
        """Return no expirations because inline calls finish synchronously."""

        del now
        return ()

    def has_pending(self) -> bool:
        """Return whether completed inline work still awaits polling."""

        return bool(self._completions)

    def reset_run(self) -> None:
        """Discard undelivered completions after a run closes or aborts."""

        self._completions.clear()

    def close(self) -> None:
        """Release local transport state idempotently."""

        if self._closed:
            return
        self._completions.clear()
        self._closed = True

    def _execute(self, values: tuple[Any, ...]) -> Any:
        """Invoke the configured batch executor or a graph-local fallback."""

        if self.executor is not None:
            return self.executor(values)
        for name in ("execute_local_batch", "execute_batch", "run_batch"):
            method = getattr(self.graph, name, None)
            if method is not None:
                return method(values)
        if callable(self.graph):
            return self.graph(values)
        return values

    def _normalize_outputs(self, output: Any, expected: int) -> tuple[Any, ...]:
        """Require exactly one transport result per submitted root."""

        if expected == 1 and not isinstance(output, (tuple, list)):
            return (output,)
        try:
            outputs = tuple(output)
        except TypeError as error:
            raise CoordinatorInvariantError(
                "inline executor must return one result per root"
            ) from error
        if len(outputs) != expected:
            raise CoordinatorInvariantError(
                "inline executor result count does not match root batch"
            )
        return outputs


class RunCoordinator:
    """Drive one bounded source run and preserve input-order result delivery."""

    def __init__(
        self,
        graph: Any,
        source: Iterable[Any] | SourceAdapter,
        *,
        transport: Transport,
        run: Any | None = None,
        limits: Any | None = None,
        batch_size: int = 1,
        planner: Any | None = None,
        scheduler: Any | None = None,
        transaction: Any | None = None,
        credits: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create run-local indexes without consuming the source iterable."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.graph = graph
        self.source = source if isinstance(source, SourceAdapter) else SourceAdapter(source)
        self.transport = transport
        self.run = run or _sem.RunId.new()
        self.limits = limits or _default_limits()
        self.batch_size = batch_size
        self.planner = planner
        self.scheduler = scheduler
        self.transaction = transaction
        self.credits = credits
        self.clock = clock
        self.roots: dict[Any, RootContext] = {}
        self.result_buffer = ResultBuffer(
            _limit(self.limits, "max_buffered_results", 16)
        )
        self.source_exhausted = False
        self.aborted: BaseException | None = None
        self._completed_by_seq: dict[int, RootResult] = {}
        self._root_by_seq: dict[int, Any] = {}
        self._next_delivery_seq = 0
        self._batch_counter = 0
        self._closed = False
        _start_transport(self.transport, graph)

    @property
    def complete(self) -> bool:
        """Return whether source, roots, events, and transport are all drained."""

        planner_events = bool(getattr(self.planner, "events", ()))
        expansions = bool(getattr(self.planner, "pending_expansions", ()))
        return (
            self.source_exhausted
            and not self.roots
            and not _transport_has_pending(self.transport)
            and not planner_events
            and not expansions
            and not self._completed_by_seq
        )

    def admit_available(self) -> int:
        """Lazily admit roots until active-root or result backpressure applies."""

        admitted = 0
        max_roots = _limit(self.limits, "max_active_roots", 16)
        while (
            not self.source_exhausted
            and len(self.roots) < max_roots
            and not self.result_buffer.full
        ):
            if self.credits is not None:
                can_admit = getattr(self.credits, "can_admit_root", None)
                if can_admit is not None and not can_admit():
                    break
            record = self.source.next_record()
            if isinstance(record, EndOfSource):
                self.source_exhausted = True
                break
            root_id = _root_id(self.run, record.source_seq)
            if root_id in self.roots:
                raise CoordinatorInvariantError("duplicate RootId admission")
            if self.credits is not None:
                admit = getattr(self.credits, "admit_root", None)
                if admit is not None and not admit(root_id):
                    raise CoordinatorInvariantError(
                        "root credit changed after source consumption"
                    )
            root = RootContext(
                id=root_id,
                source_seq=record.source_seq,
                phase=RootPhase.ACTIVE,
                source_record=record,
            )
            self.roots[root_id] = root
            self._root_by_seq[record.source_seq] = root_id
            admitted += 1
        return admitted

    def expire_deadlines(self) -> int:
        """Process transport hard-deadline completions before normal polling."""

        completions = _expire_transport(self.transport, self.clock())
        for completion in completions:
            self._accept_completion(completion)
        return len(completions)

    def drain_local_events(self) -> int:
        """Run a bounded planner turn when a semantic planner is attached."""

        if self.planner is None:
            return 0
        budget = _limit(self.limits, "max_local_events_per_turn", 256)
        return int(self.planner.drain(budget))

    def submit_available(self) -> int:
        """Submit cross-root batches while preserving per-batch source order."""

        candidates = sorted(
            (
                root
                for root in self.roots.values()
                if not root.submitted and root.phase is RootPhase.ACTIVE
            ),
            key=lambda root: root.source_seq,
        )
        if not candidates:
            return 0
        max_roots = _limit(self.limits, "max_active_roots", 16)
        flush_tail = self.source_exhausted or len(self.roots) >= max_roots
        if len(candidates) < self.batch_size and not flush_tail:
            return 0
        selected = tuple(candidates[: self.batch_size])
        batch = RootBatch(
            id=_batch_id(self.run, self._batch_counter),
            run=self.run,
            records=tuple(
                (root.id, root.source_record) for root in selected
            ),
        )
        if not self.transport.submit(batch):
            return 0
        self._batch_counter += 1
        for root in selected:
            root.submitted = True
            root.inflight_count += 1
        return 1

    def poll_ray(self) -> int:
        """Poll backend completions without assuming the backend is Ray."""

        max_items = _limit(self.limits, "max_completions_per_turn", 64)
        completions = _poll_transport(self.transport, max_items=max_items)
        for completion in completions:
            self._accept_completion(completion)
        return len(completions)

    def deliver_quiescent_roots(self) -> int:
        """Move contiguous completed roots into the bounded result buffer."""

        delivered = 0
        while not self.result_buffer.full:
            result = self._completed_by_seq.get(self._next_delivery_seq)
            if result is None:
                break
            self._completed_by_seq.pop(self._next_delivery_seq)
            root = self.roots[result.root]
            if not root.quiescent:
                raise CoordinatorInvariantError("completed root is not quiescent")
            root.phase = RootPhase.DELIVERABLE
            self.result_buffer.put(result)
            root.result_enqueued = True
            self._next_delivery_seq += 1
            delivered += 1
        return delivered

    def reclaim_delivered_roots(self) -> int:
        """Reclaim roots whose detached results are owned by the buffer."""

        reclaimed = 0
        for root_id, root in tuple(self.roots.items()):
            if not root.result_enqueued:
                continue
            root.phase = RootPhase.RECLAIMED
            self.roots.pop(root_id)
            self._root_by_seq.pop(root.source_seq, None)
            if self.credits is not None:
                release = getattr(self.credits, "release_root", None)
                if release is not None:
                    release(root_id)
            reclaimed += 1
        return reclaimed

    def drive_until_result_or_terminal(self) -> None:
        """Drive bounded turns until a result, terminal state, or abort."""

        if self._closed:
            return
        idle_turns = 0
        while not self.result_buffer and not self.complete:
            try:
                processed = sum(
                    (
                        self.expire_deadlines(),
                        self.admit_available(),
                        self.drain_local_events(),
                        self.submit_available(),
                        self.poll_ray(),
                        self.deliver_quiescent_roots(),
                        self.reclaim_delivered_roots(),
                    )
                )
            except BaseException as error:
                self.abort(error)
                break
            if processed == 0:
                idle_turns += 1
                if idle_turns > 1:
                    self._handle_stall()
            else:
                idle_turns = 0
        if self.aborted is not None:
            raise RunAborted(self.aborted) from self.aborted

    def abort(self, error: BaseException) -> None:
        """Abort only this run and close all undelivered detached values."""

        if self.aborted is None:
            self.aborted = error
        self.source.close()
        self.source_exhausted = True
        self.result_buffer.close()
        reset = getattr(self.transport, "reset_run", None)
        if reset is not None:
            reset()
        if self.credits is not None:
            release = getattr(self.credits, "release_root", None)
            if release is not None:
                for root_id in tuple(self.roots):
                    release(root_id)
        self.roots.clear()
        for result in self._completed_by_seq.values():
            _close_root_result(result)
        self._completed_by_seq.clear()
        self._root_by_seq.clear()

    def close(self) -> None:
        """Cancel remaining work and release run-owned buffered results."""

        if self._closed:
            return
        if not self.complete and self.aborted is None:
            self.abort(RuntimeError("run stream closed"))
        else:
            self.source.close()
            self.result_buffer.close()
            reset = getattr(self.transport, "reset_run", None)
            if reset is not None:
                reset()
        self._closed = True

    def _accept_completion(self, completion: Any) -> None:
        """Validate and translate one transport completion into root results."""

        if not isinstance(completion, TransportCompletion):
            raise CoordinatorInvariantError(
                "transport returned an unsupported completion record"
            )
        records = completion.batch.records
        if completion.error is not None:
            outputs: tuple[Any, ...] = tuple(
                _FailureOutput(completion.error) for _ in records
            )
        else:
            outputs = tuple(completion.outputs)
            if len(outputs) != len(records):
                raise CoordinatorInvariantError(
                    "transport completion cardinality does not match batch"
                )
        for (root_id, source_record), output in zip(records, outputs):
            root = self.roots.get(root_id)
            if root is None:
                continue
            if not root.submitted or root.inflight_count != 1:
                raise CoordinatorInvariantError(
                    "completion does not own one current root attempt"
                )
            root.inflight_count = 0
            result = self._root_result(root, source_record, output)
            existing = self._completed_by_seq.get(root.source_seq)
            if existing is not None and existing != result:
                raise CoordinatorInvariantError("conflicting duplicate completion")
            self._completed_by_seq[root.source_seq] = result

    def _root_result(
        self,
        root: RootContext,
        source: SourceRecord,
        output: Any,
    ) -> RootResult:
        """Detach one local transport output into named result leaves."""

        if isinstance(output, RootResult):
            if output.root != root.id or output.source_seq != source.source_seq:
                raise CoordinatorInvariantError(
                    "transport RootResult targets another root"
                )
            return output
        if isinstance(output, _FailureOutput):
            leaf = ResultLeaf(
                ResultLeafState.FAILED,
                failure=output.error,
            )
            return RootResult(
                root=root.id,
                source_seq=source.source_seq,
                status=RootResultStatus.FAILED,
                outputs=MappingProxyType({"output": leaf}),
            )
        mapping = output if isinstance(output, Mapping) else {"output": output}
        leaves = {
            str(name): self._result_leaf(value) for name, value in mapping.items()
        }
        status = _root_status(tuple(leaves.values()))
        return RootResult(
            root=root.id,
            source_seq=source.source_seq,
            status=status,
            outputs=MappingProxyType(leaves),
        )

    def _result_leaf(self, value: Any) -> ResultLeaf:
        """Normalize an inline result value to a detached result leaf."""

        if isinstance(value, ResultLeaf):
            return value
        if isinstance(value, DetachedValue):
            return ResultLeaf(ResultLeafState.PRESENT, value=value)
        if isinstance(value, _AbsentOutput):
            return ResultLeaf(ResultLeafState.ABSENT)
        if isinstance(value, _FailureOutput):
            return ResultLeaf(ResultLeafState.FAILED, failure=value.error)
        return ResultLeaf(
            ResultLeafState.PRESENT,
            value=DetachedValue.from_local(value),
        )

    def _progress_signature(self) -> tuple[Any, ...]:
        """Capture cheap monotonic state used to detect a stalled loop."""

        return (
            self.source.next_seq,
            self.source_exhausted,
            len(self.roots),
            len(self.result_buffer),
            len(self._completed_by_seq),
            _transport_has_pending(self.transport),
            len(getattr(self.planner, "events", ())),
        )

    def _handle_stall(self) -> None:
        """Fail deterministic local deadlocks instead of spinning forever."""

        if _transport_has_pending(self.transport):
            completions = _poll_transport(
                self.transport,
                max_items=1,
                timeout=0.01,
            )
            for completion in completions:
                self._accept_completion(completion)
            if completions:
                return
        raise CoordinatorInvariantError("run made no progress")


@dataclass(slots=True)
class _CompiledStores:
    """Aggregate one compiled run's single-writer semantic stores."""

    blocks: _state.BlockStore
    values: _state.ValueNodeStore
    controls: _state.ControlStore
    grains: _sem.GrainStore
    errors: _sem.ErrorStore
    receipts: ReceiptStore
    credits: CreditManager


class CompiledGraphRunCoordinator:
    """Execute a CompiledGraph through planner, scheduler, Ray, and transaction."""

    def __init__(
        self,
        graph: CompiledGraph,
        source: Iterable[Any] | SourceAdapter,
        *,
        transport: Transport,
        run: _sem.RunId | None = None,
        limits: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create all run-local stores without consuming the source iterable."""

        if not isinstance(graph, CompiledGraph):
            raise TypeError("CompiledGraphRunCoordinator requires CompiledGraph")
        self.graph = graph
        self.source = source if isinstance(source, SourceAdapter) else SourceAdapter(source)
        self.transport = transport
        self.run = run or _sem.RunId.new()
        self.limits = limits or _default_graph_limits()
        self.clock = clock
        blocks = _state.BlockStore()
        values = _state.ValueNodeStore(blocks)
        controls = _state.ControlStore()
        grains = _sem.GrainStore()
        errors = _sem.ErrorStore()
        receipts = ReceiptStore(values)
        credits = CreditManager(self.limits)
        self.stores = _CompiledStores(
            blocks, values, controls, grains, errors, receipts, credits
        )
        self.blocks = blocks
        self.values = values
        self.controls = controls
        self.grains = grains
        self.errors = errors
        self.receipts = receipts
        self.credits = credits
        self.scopes = ScopeTracker()
        self.scheduler = ReadyScheduler(graph, grains, clock=clock)
        self.planner = EventPlanner(
            graph,
            self.run,
            receipts=receipts,
            controls=controls,
            grains=grains,
            values=values,
            scopes=self.scopes,
            scheduler=self.scheduler,
            credits=credits,
            errors=errors,
        )
        self.transaction = LocalTransaction(
            self.stores,
            event_sink=self.planner,
            delta_builder=ManifestDeltaBuilder(graph, self.stores),
        )
        self.roots: dict[Any, RootContext] = {}
        self.result_buffer = ResultBuffer(
            _limit(self.limits, "max_buffered_results", 16)
        )
        self.source_exhausted = False
        self.aborted: BaseException | None = None
        self._next_delivery_seq = 0
        self._closed = False
        self.transport_destroyed = False
        self._metrics_started_at = self.clock()
        self._active_roots_high_watermark = 0
        self._dispatch_count_by_node: dict[Any, int] = {}
        self._grain_count_by_node: dict[Any, int] = {}
        self._batch_histogram_by_node: dict[Any, dict[int, int]] = {}
        self._worker_busy_s_by_node: dict[Any, float] = {}
        self._transport_backpressure_count = 0
        self._capacity_skip_count = 0

    @property
    def complete(self) -> bool:
        """Return whether source, semantic work, and physical work are drained."""

        return (
            self.source_exhausted
            and not self.roots
            and not self.scheduler
            and not self.planner.events
            and not self.planner.pending_expansions
            and not self.planner.waiting_expansions
            and not self.planner.waiting_grains
            and not _transport_has_pending(self.transport)
        )

    def admit_available(self) -> int:
        """Admit bounded roots and install source ObjectRef-backed facts."""

        import ray

        staged: list[RootContext] = []
        while (
            not self.source_exhausted
            and not self.result_buffer.full
            and self.credits.can_admit_root()
        ):
            record = self.source.next_record()
            if isinstance(record, EndOfSource):
                self.source_exhausted = True
                break
            root_id = _root_id(self.run, record.source_seq)
            if not self.credits.admit_root(root_id):
                raise CoordinatorInvariantError(
                    "root credit changed after source consumption"
                )
            root = RootContext(
                id=root_id,
                source_seq=record.source_seq,
                phase=RootPhase.ACTIVE,
                source_record=record,
                final_receipts={output.name: None for output in self.graph.outputs},
                submitted=True,
            )
            self.roots[root_id] = root
            staged.append(root)
        if not staged:
            return 0

        self._active_roots_high_watermark = max(
            self._active_roots_high_watermark,
            len(self.roots),
        )

        source_ref = ray.put([root.source_record.value for root in staged])
        block = self.blocks.install(source_ref, len(staged))
        try:
            for row, root in enumerate(staged):
                self._install_source(root, block, row)
        except BaseException:
            for root in staged:
                self._reclaim_root(root.id)
            if block in self.blocks:
                self.blocks.discard_unleased(block)
            raise
        return len(staged)

    def _install_source(
        self,
        root: RootContext,
        block: Any,
        row: int,
    ) -> None:
        """Publish one row from a coarse source ObjectRef block."""

        source_node = self.graph.producer(self.graph.source)
        if not isinstance(source_node.op, SourceOp):
            raise CoordinatorInvariantError("graph source port is not a SourceOp")
        value_node = self.values.create_scalar(block, row)
        entity = _sem.EntityId.source(self.run, root.source_seq)
        item = _sem.ItemRef(self.graph.source, entity)
        context = _sem.OccurrenceContext(root.id)
        grain_id = _sem.GrainId.source(self.run, source_node.id, root.source_seq)
        spec = _sem.GrainSpec(
            id=grain_id,
            run=self.run,
            root=root.id,
            node=source_node.id,
            context=context,
            inputs=(),
            output_slots=(item,),
        )
        self.values.bind_item(root.id, item, value_node)
        self.grains.ensure(spec)
        self.grains.seal(grain_id, _sem.Success())
        receipt = _sem.Receipt(
            item=item,
            context=context,
            state=_sem.ReceiptState.PRESENT,
            producer=grain_id,
        )
        self.planner.publish(receipt)
        root.grain_ids.add(grain_id)
        root.item_ids.add(item)

    def drain_local_events(self) -> int:
        """Retry transient expansion credit and drain one bounded planner turn."""

        budget = _limit(self.limits, "max_local_events_per_turn", 256)
        expanded = self.planner.retry_waiting_expansions(budget)
        remaining = max(0, budget - expanded)
        grains = self.planner.retry_waiting_grains(remaining)
        return expanded + grains + self.planner.drain(
            max(0, remaining - grains)
        )

    def submit_available(self) -> int:
        """Prepare and submit bounded same-MAP batches across causal roots."""

        submitted = 0
        budget = int(getattr(self.limits, "max_pending_dispatches", 64))
        can_submit = getattr(self.transport, "can_submit", None)
        eligible = can_submit if callable(can_submit) else None
        while submitted < budget:
            now = self.clock()
            selection = self.scheduler.select_batch(
                now,
                force=False,
                eligible=eligible,
            )
            if selection is None:
                if eligible is not None and self.scheduler.has_ready(now):
                    self._capacity_skip_count += 1
                break
            try:
                prepared = prepare_dispatch(
                    selection,
                    self.stores,
                    run=self.run,
                    limits=self.limits,
                )
            except BaseException:
                self._requeue_selection(selection)
                raise
            if not self._retain_prepared(prepared):
                self._rollback_prepared(prepared)
                break
            try:
                pending = self.transport.submit(prepared)
            except BaseException:
                self._rollback_prepared(prepared)
                raise
            if pending is None:
                self._transport_backpressure_count += 1
                self._rollback_prepared(prepared)
                break
            node = prepared.node
            size = len(prepared.attempts)
            self._dispatch_count_by_node[node] = (
                self._dispatch_count_by_node.get(node, 0) + 1
            )
            self._grain_count_by_node[node] = (
                self._grain_count_by_node.get(node, 0) + size
            )
            histogram = self._batch_histogram_by_node.setdefault(node, {})
            histogram[size] = histogram.get(size, 0) + 1
            submitted += 1
        return submitted

    def _retain_prepared(self, prepared: Any) -> bool:
        """Acquire dispatch credit and all unique input-block leases atomically."""

        if not self.credits.reserve_dispatch(prepared.id):
            return False
        retained: list[Any] = []
        try:
            for block in dict.fromkeys(prepared.block_ids):
                self.blocks.retain_dispatch(prepared.id, block)
                retained.append(block)
        except BaseException:
            for block in reversed(retained):
                self.blocks.release_dispatch(prepared.id, block)
            self.credits.release_dispatch(prepared.id)
            raise
        return True

    def _rollback_prepared(self, prepared: Any) -> None:
        """Undo reservation and input owners after transport backpressure."""

        for block in dict.fromkeys(prepared.block_ids):
            self.blocks.release_dispatch(prepared.id, block)
        self.credits.release_dispatch(prepared.id)
        for token in prepared.attempts:
            if self.grains.is_current(token):
                self.grains.retry(token)
            self.scheduler.enqueue(token.grain, node=prepared.node)

    def _requeue_selection(self, selection: BatchSelection) -> None:
        """Restore scheduler ownership when dispatch preparation itself fails."""

        for grain in selection.grains:
            state = self.grains.require(grain)
            if state.phase is _sem.GrainPhase.READY:
                self.scheduler.enqueue(grain, node=selection.node)

    def poll_ray(self, timeout: float = 0.0) -> int:
        """Apply bounded Ray completions on the single coordinator writer."""

        completions = _poll_transport(
            self.transport,
            max_items=_limit(self.limits, "max_completions_per_turn", 64),
            timeout=timeout,
        )
        for completion in completions:
            self._accept_ray_completion(completion)
        return len(completions)

    def _accept_ray_completion(self, completion: Any) -> None:
        """Commit success atomically or abort an attributable failed dispatch."""

        from ..ray.protocol import FailureManifest, SuccessManifest, WorkerErrorKind
        from ..ray.transport import RayCompletion

        if not isinstance(completion, RayCompletion):
            raise CoordinatorInvariantError("Ray transport returned another completion type")
        pending = completion.pending
        try:
            if completion.failure is not None:
                raise RuntimeError(
                    "MAP infrastructure failure "
                    f"{completion.failure.kind.value}: {completion.failure.message}"
                )
            manifest = completion.manifest
            if isinstance(manifest, SuccessManifest):
                self._worker_busy_s_by_node[pending.prepared.node] = (
                    self._worker_busy_s_by_node.get(pending.prepared.node, 0.0)
                    + manifest.worker_finished_at
                    - manifest.worker_started_at
                )
                decision = self.transaction.prepare(
                    pending, manifest, tuple(pending.output_refs)
                )
                self.transaction.apply_disposition(pending, decision)
                for token in pending.prepared.attempts:
                    self.credits.release_grain(token.grain)
            elif isinstance(manifest, FailureManifest):
                if manifest.kind is WorkerErrorKind.BAD_GRAIN:
                    self._apply_bad_grain(pending, manifest)
                else:
                    raise RuntimeError(
                        f"MAP worker failure {manifest.kind.value}: "
                        f"{manifest.error_type}: {manifest.message}"
                    )
            else:
                raise CoordinatorInvariantError("Ray completion has no valid manifest")
        except BaseException:
            if not getattr(pending, "semantic_applied", False):
                if getattr(pending, "decision", None) is None:
                    self.transaction.apply_disposition(
                        pending,
                        PendingDecision(PendingDisposition.ABORTED),
                    )
            self.transport.finalize_physical(pending, force=True)
            self._release_dispatch_owners(pending.prepared)
            raise
        self.transport.finalize_physical(pending)
        self._release_dispatch_owners(pending.prepared)

    def metrics_snapshot(self) -> dict[str, object]:
        """Return JSON-compatible run and per-MAP dispatch metrics."""

        nodes: dict[str, object] = {}
        for node_id, dispatch_count in self._dispatch_count_by_node.items():
            node = self.graph.node(node_id)
            batch_cap = int(node.op.execution.batch.max_size)
            grains = self._grain_count_by_node.get(node_id, 0)
            histogram = self._batch_histogram_by_node.get(node_id, {})
            nodes[node.name] = {
                "node_id": int(node_id),
                "dispatch_count": dispatch_count,
                "grain_count": grains,
                "batch_capacity": batch_cap,
                "average_batch_size": (
                    grains / dispatch_count if dispatch_count else 0.0
                ),
                "fill_ratio": (
                    grains / (dispatch_count * batch_cap)
                    if dispatch_count and batch_cap
                    else 0.0
                ),
                "batch_histogram": {
                    str(size): count for size, count in sorted(histogram.items())
                },
                "worker_busy_s": self._worker_busy_s_by_node.get(node_id, 0.0),
            }
        return {
            "elapsed_s": max(0.0, self.clock() - self._metrics_started_at),
            "active_roots_high_watermark": self._active_roots_high_watermark,
            "capacity_skip_count": self._capacity_skip_count,
            "transport_backpressure_count": self._transport_backpressure_count,
            "nodes": nodes,
        }

    def _apply_bad_grain(self, pending: Any, manifest: Any) -> None:
        """Fail one attributed root and return healthy batch peers to READY."""

        bad_index = manifest.bad_entry_index
        if bad_index is None:
            raise CoordinatorInvariantError("BAD_GRAIN has no entry attribution")
        attempts = tuple(pending.prepared.attempts)
        bad_token = attempts[bad_index]
        bad_state = self.grains.require(bad_token.grain)
        spec = bad_state.spec
        error_id = _sem.ErrorId.derive(
            "bad-grain-error",
            self.run,
            bad_token.grain,
            bad_token.generation,
        )
        error = _sem.ErrorRecord(
            id=error_id,
            root=spec.root,
            grain=spec.id,
            kind="BAD_GRAIN",
            message=(
                f"entry {bad_index} {manifest.error_type}: {manifest.message}"
            ),
            trace_digest=manifest.trace_digest,
        )
        receipts = tuple(
            _sem.Receipt(
                item=item,
                context=spec.context,
                state=_sem.ReceiptState.FAILED,
                producer=spec.id,
            )
            for item in tuple(spec.output_slots)
        )
        delta = CommitDelta(
            errors=(error,),
            grain_outcomes=(
                PreparedOutcome(
                    spec.id,
                    _sem.Failed(error_id),
                    token=bad_token,
                ),
            ),
            receipts=receipts,
        )
        self.transaction.apply(delta)
        self.credits.release_grain(bad_token.grain)
        for index, token in enumerate(attempts):
            if index == bad_index:
                continue
            if self.grains.is_current(token):
                self.grains.retry(token)
            self.scheduler.enqueue(token.grain, node=pending.prepared.node)
        pending.decision = PendingDecision(PendingDisposition.CONTROLLED_FAILURE)
        pending.semantic_applied = True

    def _release_dispatch_owners(self, prepared: Any) -> None:
        """Idempotently release coordinator-side physical owner ledgers."""

        for block in dict.fromkeys(prepared.block_ids):
            self.blocks.release_dispatch(prepared.id, block)
        self.credits.release_dispatch(prepared.id)

    def deliver_quiescent_roots(self) -> int:
        """Detach final refs for contiguous terminal roots in source order."""

        delivered = 0
        while not self.result_buffer.full:
            root = next(
                (
                    candidate
                    for candidate in self.roots.values()
                    if candidate.source_seq == self._next_delivery_seq
                ),
                None,
            )
            if root is None or not self._root_quiescent(root):
                break
            result = self._detach_root_result(root)
            root.phase = RootPhase.DELIVERABLE
            root.result_enqueued = True
            self.result_buffer.put(result)
            self._next_delivery_seq += 1
            delivered += 1
        return delivered

    def _root_quiescent(self, root: RootContext) -> bool:
        """Check terminal graph outputs and absence of root-owned execution work."""

        entity = _sem.EntityId.source(self.run, root.source_seq)
        for output in self.graph.outputs:
            receipt = self.receipts.get(_sem.ItemRef(output.port, entity))
            if receipt is None:
                return False
            root.final_receipts[output.name] = receipt
        if any(
            state.phase is not _sem.GrainPhase.TERMINAL
            for state in self.grains.states_for_root(root.id)
        ):
            return False
        if self.planner.events:
            return False
        for scope in (
            set(self.planner.pending_expansions)
            | set(self.planner.waiting_expansions)
        ):
            instance = self.scopes.instances.get(scope)
            if instance is not None and instance.root == root.id:
                return False
        if any(
            latch.context.root == root.id
            for latch in self.planner.waiting_grains.values()
        ):
            return False
        return not any(
            self.scopes.instances[scope].root == root.id
            for _, scope in self.scopes.closures
            if scope in self.scopes.instances
        )

    def _detach_root_result(self, root: RootContext) -> RootResult:
        """Detach every named graph output without fetching intermediate payloads."""

        leaves: dict[str, ResultLeaf] = {}
        for output in self.graph.outputs:
            receipt = root.final_receipts[output.name]
            if receipt.state is _sem.ReceiptState.PRESENT:
                try:
                    detached = self._detach_item(receipt.item)
                except DispatchInvariantError as error:
                    leaves[output.name] = ResultLeaf(
                        ResultLeafState.FAILED,
                        failure=_sem.DeliveryFailureSummary(
                            root=root.id,
                            kind="DETACH_LIMIT",
                            message=str(error),
                        ),
                    )
                else:
                    leaves[output.name] = ResultLeaf(
                        ResultLeafState.PRESENT,
                        value=detached,
                    )
            elif receipt.state is _sem.ReceiptState.NORMAL_ABSENCE:
                leaves[output.name] = ResultLeaf(ResultLeafState.ABSENT)
            else:
                leaves[output.name] = ResultLeaf(
                    ResultLeafState.FAILED,
                    failure=self.errors.summarize_root(root.id, self.grains),
                )
        frozen = MappingProxyType(leaves)
        return RootResult(
            root=root.id,
            source_seq=root.source_seq,
            status=_root_status(tuple(leaves.values())),
            outputs=frozen,
        )

    def _detach_item(self, item: Any) -> DetachedValue:
        """Copy one gather and its ObjectRef handles out of mutable stores."""

        from ..ray.protocol import SlotTake, WireList

        interner = RefInterner(
            max_refs=getattr(self.limits, "max_detached_result_refs", None)
        )
        tree = interner.bind(
            resolve_storage_gather(self.values, item),
            max_depth=getattr(self.limits, "max_gather_depth", None),
            max_nodes=getattr(self.limits, "max_detached_result_nodes", None),
            slot_factory=lambda slot, row: SlotTake(slot, row),
            list_factory=lambda children: WireList(children),
        )
        refs = tuple(self.blocks.resolve_ref(block) for block in interner.block_ids)
        return DetachedValue(refs=refs, tree=tree, resolver=_resolve_ray_tree)

    def reclaim_delivered_roots(self) -> int:
        """Reclaim semantic stores after detached results own their ref handles."""

        reclaimed = 0
        for root_id, root in tuple(self.roots.items()):
            if root.result_enqueued:
                self._reclaim_root(root_id)
                reclaimed += 1
        return reclaimed

    def _reclaim_root(self, root_id: Any) -> None:
        """Remove one root from every run-local owner index idempotently."""

        root = self.roots.pop(root_id, None)
        self.planner.remove_root(root_id)
        self.scopes.remove_root(root_id)
        self.receipts.remove_root(root_id)
        self.controls.remove_root(root_id)
        self.grains.remove_root(root_id)
        self.errors.remove_root(root_id)
        self.values.remove_root(root_id)
        self.credits.release_root_resources(root_id)
        if root is not None:
            root.phase = RootPhase.RECLAIMED

    def drive_until_result_or_terminal(self) -> None:
        """Drive bounded semantic and physical turns until delivery or terminal."""

        if self._closed:
            return
        idle_turns = 0
        while not self.result_buffer and not self.complete:
            try:
                processed = sum(
                    (
                        self.admit_available(),
                        self.drain_local_events(),
                        self.submit_available(),
                        self.poll_ray(),
                        self.deliver_quiescent_roots(),
                        self.reclaim_delivered_roots(),
                    )
                )
            except BaseException as error:
                self.abort(error)
                break
            if processed == 0:
                idle_turns += 1
                now = self.clock()
                flush_at = self.scheduler.next_flush_at(now)
                if _transport_has_pending(self.transport):
                    try:
                        wait_s = 0.05
                        if flush_at is not None:
                            wait_s = min(wait_s, max(0.0, flush_at - now))
                        waited = self.poll_ray(timeout=wait_s)
                    except BaseException as error:
                        self.abort(error)
                        break
                    if waited:
                        idle_turns = 0
                    elif flush_at is not None and flush_at <= self.clock():
                        idle_turns = 0
                elif flush_at is not None and flush_at > now:
                    time.sleep(min(0.05, flush_at - now))
                    idle_turns = 0
                elif idle_turns > 1:
                    self.abort(CoordinatorInvariantError("compiled run made no progress"))
            else:
                idle_turns = 0
        if self.aborted is not None:
            raise RunAborted(self.aborted) from self.aborted

    def abort(self, error: BaseException) -> None:
        """Cancel this run, finalize dispatches, and reclaim every root owner."""

        if self.aborted is None:
            self.aborted = error
        self.source.close()
        self.source_exhausted = True
        pending_method = getattr(self.transport, "pending", None)
        pending_items = () if pending_method is None else pending_method(self.run)
        cancel = getattr(self.transport, "cancel", None)
        if cancel is not None:
            try:
                cancel(self.run)
            except BaseException:
                pass
        for pending in pending_items:
            try:
                if not pending.semantic_applied and pending.decision is None:
                    self.transaction.apply_disposition(
                        pending,
                        PendingDecision(PendingDisposition.ABORTED),
                    )
                self.transport.finalize_physical(pending, force=True)
            except BaseException:
                # Run abort is a best-effort sweep; session close remains the
                # final actor/resource safety boundary.
                pass
            finally:
                self._release_dispatch_owners(pending.prepared)
        remaining = () if pending_method is None else pending_method(self.run)
        if remaining or (
            pending_method is None and _transport_has_pending(self.transport)
        ):
            self.transport.close()
            self.transport_destroyed = True
        self.result_buffer.close()
        for root_id in tuple(self.roots):
            self._reclaim_root(root_id)

    def close(self) -> None:
        """Close unread run work while preserving graph-bound actor pools."""

        if self._closed:
            return
        if not self.complete and self.aborted is None:
            self.abort(RuntimeError("run stream closed"))
        else:
            self.source.close()
            self.result_buffer.close()
        self._closed = True

    def _progress_signature(self) -> tuple[Any, ...]:
        """Capture bounded state used to diagnose a deterministic deadlock."""

        return (
            self.source.next_seq,
            self.source_exhausted,
            len(self.roots),
            len(self.result_buffer),
            len(self.scheduler),
            len(self.planner.events),
            len(self.planner.pending_expansions),
            len(self.planner.waiting_expansions),
            _transport_has_pending(self.transport),
            self.credits.live_occurrences,
            self.credits.live_structural_edges,
        )


@dataclass(frozen=True, slots=True)
class _AbsentOutput:
    """Represent local normal absence before it becomes a ResultLeaf."""


@dataclass(frozen=True, slots=True)
class _FailureOutput:
    """Represent local root delivery failure before result normalization."""

    error: BaseException


ABSENT_OUTPUT = _AbsentOutput()


class RunStream(Iterator[RootResult]):
    """Lazily drive one coordinator and transfer results to the caller."""

    def __init__(
        self,
        coordinator: RunCoordinator,
        *,
        on_close: Callable[["RunStream"], None] | None = None,
    ) -> None:
        """Wrap one active coordinator without driving it eagerly."""

        self.coordinator = coordinator
        self._on_close = on_close
        self._closed = False
        self._released = False

    def __iter__(self) -> "RunStream":
        """Return this single-pass result iterator."""

        return self

    def __next__(self) -> RootResult:
        """Drive until the next input-ordered result or terminal state."""

        if self._closed:
            raise StopIteration
        if not self.coordinator.result_buffer:
            try:
                self.coordinator.drive_until_result_or_terminal()
            except RunAborted:
                self._closed = True
                self._release_session()
                raise
        if self.coordinator.result_buffer:
            result = self.coordinator.result_buffer.take()
            if self.coordinator.complete and not self.coordinator.result_buffer:
                self._release_session()
            return result
        self._closed = True
        self._release_session()
        raise StopIteration

    def collect(self, order: str = "input") -> list[RootResult]:
        """Drain the stream; first edition supports deterministic input order."""

        if order not in {"input", "completion"}:
            raise ValueError("order must be 'input' or 'completion'")
        results = list(self)
        if order == "input":
            results.sort(key=lambda result: result.source_seq)
        return results

    def close(self) -> None:
        """Cancel unread work and close all still-buffered detached values."""

        if self._closed:
            self._release_session()
            return
        self.coordinator.close()
        self._closed = True
        self._release_session()

    def __enter__(self) -> "RunStream":
        """Return this active stream for context-manager use."""

        if self._closed:
            raise RuntimeError("RunStream is closed")
        return self

    def __exit__(self, *_exc: Any) -> None:
        """Close unread work on context-manager exit."""

        self.close()

    def __del__(self) -> None:
        """Best-effort fallback cancellation for an abandoned stream."""

        try:
            self.close()
        except BaseException:
            pass

    def _release_session(self) -> None:
        """Notify the owning session exactly once."""

        if self._released:
            return
        self._released = True
        if self._on_close is not None:
            self._on_close(self)


class ExecutorSession:
    """Bind one graph and transport across sequential, non-overlapping runs."""

    def __init__(
        self,
        graph: Any,
        *,
        transport: Transport | None = None,
        local_executor: Callable[[tuple[Any, ...]], Iterable[Any]] | None = None,
        limits: Any | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Create a graph-bound session without admitting source records."""

        self.graph = graph
        self._compiled = isinstance(graph, CompiledGraph)
        if self._compiled:
            _validate_supported_policies(graph)
        self.limits = limits or (
            _default_graph_limits() if self._compiled else _default_limits()
        )
        self._active_coordinator: Any | None = None
        self._transport_broken = False
        if transport is not None:
            self.transport = transport
        elif self._compiled:
            if local_executor is not None:
                raise ValueError(
                    "CompiledGraph execution does not accept root-batch local_executor"
                )
            self.transport = self._build_ray_transport()
        else:
            self.transport = InlineTransport(local_executor)
        self.batch_size = batch_size or _graph_batch_size(graph)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._active: RunStream | None = None
        self._closed = False
        _start_transport(self.transport, graph)

    @property
    def active_run(self) -> RunStream | None:
        """Return the one currently active stream, if any."""

        return self._active

    def wait_ready(self, timeout_s: float | None = None) -> object:
        """Wait for persistent MAP actors to finish UDF initialization."""

        if self._closed:
            raise RuntimeError("ExecutorSession is closed")
        wait = getattr(self.transport, "wait_ready", None)
        if wait is None:
            return None
        return wait(timeout_s)

    def run(self, source: Iterable[Any]) -> RunStream:
        """Start one lazy run and reject overlap within this session."""

        if self._closed:
            raise RuntimeError("ExecutorSession is closed")
        if self._transport_broken:
            raise RuntimeError(
                "ExecutorSession transport was destroyed during prior abort"
            )
        if self._active is not None:
            raise RuntimeError("ExecutorSession already has an active run")
        if self._compiled:
            coordinator: Any = CompiledGraphRunCoordinator(
                self.graph,
                source,
                transport=self.transport,
                limits=self.limits,
            )
        else:
            coordinator = RunCoordinator(
                self.graph,
                source,
                transport=self.transport,
                limits=self.limits,
                batch_size=self.batch_size,
            )
        self._active_coordinator = coordinator
        stream = RunStream(coordinator, on_close=self._stream_closed)
        self._active = stream
        return stream

    def execute(self, source: Iterable[Any]) -> RunStream:
        """Alias ``run`` for execution-oriented call sites."""

        return self.run(source)

    def close(self) -> None:
        """Close the active run and graph-bound transport idempotently."""

        if self._closed:
            return
        active = self._active
        if active is not None:
            active.close()
        self.transport.close()
        self._closed = True

    def __enter__(self) -> "ExecutorSession":
        """Return this open session for context-manager use."""

        if self._closed:
            raise RuntimeError("ExecutorSession is closed")
        return self

    def __exit__(self, *_exc: Any) -> None:
        """Close graph-bound resources on context-manager exit."""

        self.close()

    def _stream_closed(self, stream: RunStream) -> None:
        """Clear only the exact active stream that reached terminal state."""

        if self._active is stream:
            coordinator = self._active_coordinator
            if bool(getattr(coordinator, "transport_destroyed", False)):
                self._transport_broken = True
            self._active = None
            self._active_coordinator = None

    def _build_ray_transport(self) -> Any:
        """Build one persistent ActorPool per MAP node and a shared transport."""

        from ..ray.protocol import PROTOCOL_VERSION
        from ..ray.transport import ActorPool, RayTransport
        from ..ray.worker import WorkerContext

        pools = {}
        for node in self.graph.nodes:
            if not isinstance(node.op, MapOp):
                continue
            resources = node.op.execution.resources
            context = WorkerContext(
                protocol_version=PROTOCOL_VERSION,
                graph_fingerprint=self.graph.fingerprint,
                node=node.id,
                call_schema=node.op.call_schema,
                return_schema=node.op.return_schema,
                output_schema=node.op.physical_outputs,
                max_manifest_bytes=getattr(
                    self.limits, "max_manifest_bytes", 4 * 1024 * 1024
                ),
                max_error_message_bytes=getattr(
                    self.limits, "max_error_message_bytes", 64 * 1024
                ),
            )
            ray_options: dict[str, Any] = {
                "num_cpus": resources.num_cpus,
                "num_gpus": resources.num_gpus,
            }
            if resources.runtime_env:
                ray_options["runtime_env"] = thaw_config_value(
                    dict(resources.runtime_env)
                )
            pools[node.id] = ActorPool(
                node.id,
                context,
                node.op.udf,
                replicas=resources.replicas,
                ray_options=ray_options,
            )
        if not pools:
            raise ValueError("CompiledGraph execution requires at least one MAP node")
        return RayTransport(
            pools,
            resolve_input=self._resolve_input_ref,
            release_input=self._release_input_owner,
            release_reservation=self._release_dispatch_reservation,
        )

    def _resolve_input_ref(self, block: Any) -> Any:
        """Resolve one current run block to its opaque Ray ObjectRef."""

        if self._active_coordinator is None:
            raise RuntimeError("no active compiled run owns input blocks")
        return self._active_coordinator.blocks.resolve_ref(block)

    def _release_input_owner(self, dispatch: Any, block: Any) -> None:
        """Release one current run input-block owner idempotently."""

        if self._active_coordinator is not None:
            self._active_coordinator.blocks.release_dispatch(dispatch, block)

    def _release_dispatch_reservation(self, dispatch: Any) -> None:
        """Release one current run pending-dispatch credit idempotently."""

        if self._active_coordinator is not None:
            self._active_coordinator.credits.release_dispatch(dispatch)


def _start_transport(transport: Any, graph: Any) -> None:
    """Start graph-bound resources when a transport exposes that hook."""

    start = getattr(transport, "start", None)
    if start is not None:
        start(graph)


def _expire_transport(transport: Any, now: float) -> tuple[Any, ...]:
    """Read explicit deadline completions from transports that expose them."""

    expire = getattr(transport, "expire_deadlines", None)
    if expire is None:
        return ()
    return tuple(expire(now))


def _poll_transport(
    transport: Any,
    *,
    max_items: int,
    timeout: float = 0.0,
) -> tuple[Any, ...]:
    """Poll the shared Ray-compatible completion method shape."""

    return tuple(
        transport.poll(timeout, max_completions=max_items)
    )


def _transport_has_pending(transport: Any) -> bool:
    """Observe pending work through a public hook or adapter-owned mapping."""

    has_pending = getattr(transport, "has_pending", None)
    if has_pending is not None:
        return bool(has_pending())
    pending = getattr(transport, "_pending", None)
    return bool(pending)


def _root_id(run: Any, source_seq: int) -> Any:
    """Derive a root ID from run identity and source order."""

    if isinstance(run, _sem.RunId):
        return _sem.RootId.for_source(run, source_seq)
    return RuntimeRootId(_digest("root", _raw_id(run), source_seq))


def _batch_id(run: Any, sequence: int) -> Any:
    """Derive a run-local batch owner ID."""

    if isinstance(run, _sem.RunId):
        return _sem.DispatchId.derive("inline-batch", run, sequence)
    return RuntimeBatchId(_digest("inline-batch", _raw_id(run), sequence))


def _digest(domain: str, raw: bytes, sequence: int) -> bytes:
    """Hash a domain, run bytes, and non-negative sequence canonically."""

    if sequence < 0:
        raise ValueError("identity sequence must be non-negative")
    payload = (
        len(domain).to_bytes(2, "big")
        + domain.encode("utf-8")
        + len(raw).to_bytes(2, "big")
        + raw
        + sequence.to_bytes(8, "big")
    )
    return hashlib.blake2b(
        payload,
        digest_size=16,
        person=b"RayOrchMGV3Run",
    ).digest()


def _raw_id(value: Any) -> bytes:
    """Extract bytes from a model or local run identity."""

    if isinstance(value, bytes):
        return value
    raw = getattr(value, "raw", None)
    if isinstance(raw, bytes):
        return raw
    model_value = getattr(value, "value", None)
    if isinstance(model_value, bytes):
        return model_value
    raise TypeError("run identity must expose raw bytes")


def _limit(limits: Any, name: str, default: int) -> int:
    """Read one positive coordinator limit."""

    value = int(getattr(limits, name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _default_limits() -> Any:
    """Create conservative local defaults without depending on Ray/model state."""

    return _LocalRuntimeLimits()


def _default_graph_limits() -> _state.RuntimeLimits:
    """Create conservative complete limits for compiled graph execution."""

    return _state.RuntimeLimits(
        max_active_roots=32,
        max_live_grains=100_000,
        max_live_occurrences=100_000,
        max_occurrences_per_root=10_000,
        max_live_structural_edges=200_000,
        max_structural_edges_per_root=20_000,
        max_scope_width=10_000,
        max_pending_expansions=1_024,
        max_pending_dispatches=1_024,
        max_local_events=200_000,
        max_gather_depth=64,
        max_gather_nodes_per_entry=100_000,
        max_refs_per_dispatch=10_000,
        max_manifest_bytes=4 * 1024 * 1024,
        max_error_message_bytes=64 * 1024,
        max_buffered_results=32,
        max_detached_result_nodes=100_000,
        max_detached_result_refs=10_000,
    )


def default_runtime_limits(**overrides: int) -> _state.RuntimeLimits:
    """Return complete compiled-graph limits with validated integer overrides."""

    unknown = set(overrides).difference(_state.RuntimeLimits.__dataclass_fields__)
    if unknown:
        raise TypeError(
            "unknown runtime limit(s): " + ", ".join(sorted(unknown))
        )
    return replace(_default_graph_limits(), **overrides)


def _validate_supported_policies(graph: CompiledGraph) -> None:
    """Reject failure configurations whose retry/isolation semantics are absent."""

    unsupported: list[str] = []
    for node in graph.nodes:
        if not isinstance(node.op, MapOp):
            continue
        policy = node.op.execution.failures
        if policy.mode != "raise" or policy.infra_retries != 0:
            unsupported.append(
                f"{node.name}(mode={policy.mode}, "
                f"infra_retries={policy.infra_retries})"
            )
    if unsupported:
        raise NotImplementedError(
            "V3 currently supports only failure mode='raise' with "
            "infra_retries=0; unsupported: " + ", ".join(unsupported)
        )


def _graph_batch_size(graph: Any) -> int:
    """Use the first MAP batch policy, falling back to one root per batch."""

    for node in getattr(graph, "nodes", ()):
        op = getattr(node, "op", None)
        if type(op).__name__.lower() not in {"map", "mapop"}:
            continue
        execution = getattr(op, "execution", None)
        batch = getattr(execution, "batch", execution)
        size = getattr(batch, "max_size", None)
        if size is not None:
            return int(size)
    return 1


def _root_status(leaves: tuple[ResultLeaf, ...]) -> RootResultStatus:
    """Compute root status from final leaf states."""

    if any(leaf.state is ResultLeafState.FAILED for leaf in leaves):
        return RootResultStatus.FAILED
    if any(leaf.state is ResultLeafState.PRESENT for leaf in leaves):
        return RootResultStatus.SUCCESS
    return RootResultStatus.DROPPED


def _close_root_result(result: RootResult) -> None:
    """Close every detached leaf still owned by a buffered result."""

    for leaf in result.outputs.values():
        if leaf.value is not None:
            leaf.value.close()


def _resolve_wire_tree(refs: tuple[Any, ...], tree: Any) -> Any:
    """Resolve a detached local wire tree without importing Ray."""

    if isinstance(tree, LocalSlotTake) or (
        hasattr(tree, "ref_slot") and hasattr(tree, "row")
    ):
        block = refs[tree.ref_slot]
        return block[tree.row]
    if isinstance(tree, LocalWireList) or hasattr(tree, "children"):
        return [_resolve_wire_tree(refs, child) for child in tree.children]
    raise CoordinatorInvariantError("unsupported detached wire gather")


def _resolve_ray_tree(refs: tuple[Any, ...], tree: Any) -> Any:
    """Materialize detached final blocks and apply their immutable gather."""

    import ray

    blocks = tuple(ray.get(list(refs)))
    return _resolve_wire_tree(blocks, tree)


__all__ = [
    "ABSENT_OUTPUT",
    "CoordinatorInvariantError",
    "CompiledGraphRunCoordinator",
    "DetachedValue",
    "EndOfSource",
    "ExecutorSession",
    "InlineTransport",
    "ResultBuffer",
    "ResultLeaf",
    "ResultLeafState",
    "RootContext",
    "RootPhase",
    "RootResult",
    "RootResultStatus",
    "RunAborted",
    "RunCoordinator",
    "RunStream",
    "SourceAdapter",
    "SourceRecord",
    "Transport",
    "TransportCompletion",
    "default_runtime_limits",
]
