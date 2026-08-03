"""Versioned DTOs shared by the V3 Ray driver and persistent workers.

Only backend-neutral model records are allowed in this module.  In particular,
wire selectors never contain central ``BlockId`` values, runtime stores, actor
handles, or callbacks.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from ..model.graph import (
    BoolShape,
    OpaqueShape,
    PhysicalOutputSpec,
    StructuralListShape,
    ValueShape,
)
from ..model.semantics import (
    ActorId,
    AttemptToken,
    DispatchId,
    GraphFingerprint,
    LeaseId,
    NodeId,
    PortId,
    RunId,
)


PROTOCOL_VERSION = 3
"""Wire protocol implemented by the architecture v0.3 backend."""

EMPTY_BLOCK: tuple[object, ...] = ()
"""Fixed placeholder yielded before a controlled failure manifest."""


class ProtocolValidationError(ValueError):
    """Report a malformed or incompatible Ray wire record."""


def _plain_int(value: object, field: str, *, minimum: int = 0) -> int:
    """Validate an integer field without accepting ``bool`` as an integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an int")
    if value < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return value


def _tuple(value: object, field: str) -> tuple[object, ...]:
    """Require an immutable tuple at a wire boundary."""

    if not isinstance(value, tuple):
        raise TypeError(f"{field} must be a tuple")
    return value


def _shape(value: object, field: str = "shape") -> ValueShape:
    """Require one of the frozen model shape records."""

    if not isinstance(value, (OpaqueShape, BoolShape, StructuralListShape)):
        raise TypeError(f"{field} must be a ValueShape")
    return value


@dataclass(frozen=True, slots=True)
class SlotTake:
    """Select one row from an input ref's resolved coarse block."""

    ref_slot: int
    row: int

    def __post_init__(self) -> None:
        """Reject negative or non-integral selectors."""

        _plain_int(self.ref_slot, "SlotTake.ref_slot")
        _plain_int(self.row, "SlotTake.row")


@dataclass(frozen=True, slots=True)
class WireList:
    """Represent a structural list assembled from nested wire selectors."""

    children: tuple["WireGather", ...]

    def __post_init__(self) -> None:
        """Require immutable children containing only wire gather nodes."""

        _tuple(self.children, "WireList.children")
        for child in self.children:
            if not isinstance(child, (SlotTake, WireList)):
                raise TypeError("WireList.children must contain WireGather nodes")


WireGather: TypeAlias = SlotTake | WireList


@dataclass(frozen=True, slots=True)
class ActorLease:
    """Identify one exact pending owner on one actor incarnation."""

    node: NodeId
    slot: int
    incarnation: int
    actor_id: ActorId
    lease_id: LeaseId

    def __post_init__(self) -> None:
        """Validate every component used by actor-pool compare-and-swap."""

        if not isinstance(self.node, NodeId):
            raise TypeError("ActorLease.node must be a NodeId")
        _plain_int(self.slot, "ActorLease.slot")
        _plain_int(self.incarnation, "ActorLease.incarnation")
        if not isinstance(self.actor_id, ActorId):
            raise TypeError("ActorLease.actor_id must be an ActorId")
        if not isinstance(self.lease_id, LeaseId):
            raise TypeError("ActorLease.lease_id must be a LeaseId")


@dataclass(frozen=True, slots=True)
class WorkerEntry:
    """Carry one attempt authority and its parameter-ordered gather trees."""

    token: AttemptToken
    role_trees: tuple[WireGather, ...]

    def __post_init__(self) -> None:
        """Validate the attempt token and all top-level role selectors."""

        if not isinstance(self.token, AttemptToken):
            raise TypeError("WorkerEntry.token must be an AttemptToken")
        _tuple(self.role_trees, "WorkerEntry.role_trees")
        for tree in self.role_trees:
            if not isinstance(tree, (SlotTake, WireList)):
                raise TypeError("WorkerEntry.role_trees must contain WireGather nodes")


@dataclass(frozen=True, slots=True)
class WorkerDispatch:
    """Describe one immutable, versioned invocation sent to a MAP worker."""

    protocol_version: int
    graph_fingerprint: GraphFingerprint
    run: RunId
    dispatch: DispatchId
    node: NodeId
    lease: ActorLease
    entries: tuple[WorkerEntry, ...]

    def __post_init__(self) -> None:
        """Validate local field types and immutable entry structure."""

        _plain_int(self.protocol_version, "WorkerDispatch.protocol_version", minimum=1)
        if not isinstance(self.graph_fingerprint, GraphFingerprint):
            raise TypeError(
                "WorkerDispatch.graph_fingerprint must be a GraphFingerprint"
            )
        if not isinstance(self.run, RunId):
            raise TypeError("WorkerDispatch.run must be a RunId")
        if not isinstance(self.dispatch, DispatchId):
            raise TypeError("WorkerDispatch.dispatch must be a DispatchId")
        if not isinstance(self.node, NodeId):
            raise TypeError("WorkerDispatch.node must be a NodeId")
        if not isinstance(self.lease, ActorLease):
            raise TypeError("WorkerDispatch.lease must be an ActorLease")
        _tuple(self.entries, "WorkerDispatch.entries")
        if not self.entries:
            raise ValueError("WorkerDispatch.entries must not be empty")
        if any(not isinstance(entry, WorkerEntry) for entry in self.entries):
            raise TypeError("WorkerDispatch.entries must contain WorkerEntry records")
        if self.lease.node != self.node:
            raise ValueError("WorkerDispatch lease node must match dispatch node")
        tokens = tuple(entry.token for entry in self.entries)
        if any(token.run != self.run for token in tokens):
            raise ValueError("WorkerDispatch attempt runs must match dispatch run")
        if len(set(tokens)) != len(tokens):
            raise ValueError("WorkerDispatch attempts must be unique")


@dataclass(frozen=True, slots=True)
class ManifestHeader:
    """Echo the exact dispatch identity and publication authorities."""

    protocol_version: int
    graph_fingerprint: GraphFingerprint
    run: RunId
    dispatch: DispatchId
    node: NodeId
    lease: ActorLease
    attempts: tuple[AttemptToken, ...]

    def __post_init__(self) -> None:
        """Validate immutable identity fields and attempt alignment."""

        _plain_int(self.protocol_version, "ManifestHeader.protocol_version", minimum=1)
        if not isinstance(self.graph_fingerprint, GraphFingerprint):
            raise TypeError(
                "ManifestHeader.graph_fingerprint must be a GraphFingerprint"
            )
        if not isinstance(self.run, RunId):
            raise TypeError("ManifestHeader.run must be a RunId")
        if not isinstance(self.dispatch, DispatchId):
            raise TypeError("ManifestHeader.dispatch must be a DispatchId")
        if not isinstance(self.node, NodeId):
            raise TypeError("ManifestHeader.node must be a NodeId")
        if not isinstance(self.lease, ActorLease):
            raise TypeError("ManifestHeader.lease must be an ActorLease")
        _tuple(self.attempts, "ManifestHeader.attempts")
        if not self.attempts:
            raise ValueError("ManifestHeader.attempts must not be empty")
        if any(not isinstance(token, AttemptToken) for token in self.attempts):
            raise TypeError(
                "ManifestHeader.attempts must contain AttemptToken records"
            )
        if any(token.run != self.run for token in self.attempts):
            raise ValueError("manifest attempt runs must match the header run")
        if len(set(self.attempts)) != len(self.attempts):
            raise ValueError("manifest attempts must be unique")
        if self.lease.node != self.node:
            raise ValueError("manifest lease node must match the header node")


@dataclass(frozen=True, slots=True)
class OutputLayout:
    """Describe one return-slot block's logical-to-physical row mapping."""

    return_slot: int
    port: PortId
    shape: ValueShape
    logical_count: int
    row_count: int
    offsets: tuple[int, ...] | None
    control_bits: bytes | None
    estimated_bytes: int | None

    def __post_init__(self) -> None:
        """Validate scalar, list, and optional control-bit layouts completely."""

        _plain_int(self.return_slot, "OutputLayout.return_slot")
        if not isinstance(self.port, PortId):
            raise TypeError("OutputLayout.port must be a PortId")
        shape = _shape(self.shape, "OutputLayout.shape")
        logical_count = _plain_int(
            self.logical_count, "OutputLayout.logical_count"
        )
        row_count = _plain_int(self.row_count, "OutputLayout.row_count")
        if self.estimated_bytes is not None:
            _plain_int(self.estimated_bytes, "OutputLayout.estimated_bytes")

        if isinstance(shape, StructuralListShape):
            if self.offsets is None:
                raise ValueError("structural list layouts require offsets")
            _tuple(self.offsets, "OutputLayout.offsets")
            if len(self.offsets) != logical_count + 1:
                raise ValueError("list offsets must have logical_count + 1 entries")
            previous = -1
            for offset in self.offsets:
                current = _plain_int(offset, "OutputLayout.offsets entry")
                if current < previous:
                    raise ValueError("list offsets must be monotonic")
                previous = current
            if not self.offsets or self.offsets[0] != 0:
                raise ValueError("list offsets must start at zero")
            if self.offsets[-1] != row_count:
                raise ValueError("last list offset must equal row_count")
            if self.control_bits is not None:
                raise ValueError("structural list layouts cannot carry control bits")
            return

        if self.offsets is not None:
            raise ValueError("scalar layouts must not carry offsets")
        if row_count != logical_count:
            raise ValueError("scalar row_count must equal logical_count")
        if self.control_bits is None:
            return
        if not isinstance(shape, BoolShape):
            raise ValueError("only BoolShape layouts may carry control bits")
        if not isinstance(self.control_bits, bytes):
            raise TypeError("OutputLayout.control_bits must be bytes")
        expected_bytes = (logical_count + 7) // 8
        if len(self.control_bits) != expected_bytes:
            raise ValueError("control bitset byte length does not match logical_count")
        remainder = logical_count % 8
        if remainder and self.control_bits:
            padding_mask = 0xFF ^ ((1 << remainder) - 1)
            if self.control_bits[-1] & padding_mask:
                raise ValueError("control bitset has non-zero padding bits")


class WorkerErrorKind(Enum):
    """Classify controlled worker failures without choosing a disposition."""

    BAD_GRAIN = "bad_grain"
    GENERIC_UDF = "generic_udf"
    CONTRACT = "contract"


@dataclass(frozen=True, slots=True)
class SuccessManifest:
    """Commit-gate manifest for a completely normalized MAP result."""

    header: ManifestHeader
    outputs: tuple[OutputLayout, ...]
    worker_started_at: float
    worker_finished_at: float
    worker_rss_bytes: int | None

    def __post_init__(self) -> None:
        """Validate output ordering, batch cardinality, timing, and RSS."""

        if not isinstance(self.header, ManifestHeader):
            raise TypeError("SuccessManifest.header must be a ManifestHeader")
        _tuple(self.outputs, "SuccessManifest.outputs")
        if not self.outputs:
            raise ValueError("SuccessManifest.outputs must not be empty")
        if any(not isinstance(layout, OutputLayout) for layout in self.outputs):
            raise TypeError(
                "SuccessManifest.outputs must contain OutputLayout records"
            )
        for slot, layout in enumerate(self.outputs):
            if layout.return_slot != slot:
                raise ValueError("success outputs must be in contiguous return order")
            if layout.logical_count != len(self.header.attempts):
                raise ValueError(
                    "each output logical_count must equal the attempt count"
                )
        if len({layout.port for layout in self.outputs}) != len(self.outputs):
            raise ValueError("success output ports must be unique")
        for value, field in (
            (self.worker_started_at, "SuccessManifest.worker_started_at"),
            (self.worker_finished_at, "SuccessManifest.worker_finished_at"),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field} must be a finite number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{field} must be finite")
        if self.worker_finished_at < self.worker_started_at:
            raise ValueError("worker_finished_at must not precede worker_started_at")
        if self.worker_rss_bytes is not None:
            _plain_int(self.worker_rss_bytes, "SuccessManifest.worker_rss_bytes")


@dataclass(frozen=True, slots=True)
class FailureManifest:
    """Commit-gate manifest for a controlled worker-side failure."""

    header: ManifestHeader
    kind: WorkerErrorKind
    bad_entry_index: int | None
    error_type: str
    message: str
    trace_digest: bytes | None

    def __post_init__(self) -> None:
        """Validate failure classification and any entry-local attribution."""

        if not isinstance(self.header, ManifestHeader):
            raise TypeError("FailureManifest.header must be a ManifestHeader")
        if not isinstance(self.kind, WorkerErrorKind):
            raise TypeError("FailureManifest.kind must be a WorkerErrorKind")
        if not isinstance(self.error_type, str) or not self.error_type:
            raise ValueError("FailureManifest.error_type must be a non-empty string")
        if not isinstance(self.message, str):
            raise TypeError("FailureManifest.message must be a string")
        if self.trace_digest is not None and not isinstance(self.trace_digest, bytes):
            raise TypeError("FailureManifest.trace_digest must be bytes")
        if self.kind is WorkerErrorKind.BAD_GRAIN:
            if self.bad_entry_index is None:
                raise ValueError("BAD_GRAIN requires bad_entry_index")
            index = _plain_int(
                self.bad_entry_index, "FailureManifest.bad_entry_index"
            )
            if index >= len(self.header.attempts):
                raise ValueError("bad_entry_index is outside the dispatch batch")
        elif self.bad_entry_index is not None:
            raise ValueError("only BAD_GRAIN may carry bad_entry_index")


Manifest: TypeAlias = SuccessManifest | FailureManifest


def validate_wire_gather(
    gather: WireGather,
    *,
    ref_count: int | None = None,
    max_depth: int | None = None,
    max_nodes: int | None = None,
) -> tuple[int, int]:
    """Iteratively validate a gather and return ``(node_count, max_depth)``."""

    if ref_count is not None:
        _plain_int(ref_count, "ref_count")
    if max_depth is not None:
        _plain_int(max_depth, "max_depth")
    if max_nodes is not None:
        _plain_int(max_nodes, "max_nodes")
    if not isinstance(gather, (SlotTake, WireList)):
        raise TypeError("gather must be a WireGather")

    count = 0
    observed_depth = 0
    stack: list[tuple[WireGather, int]] = [(gather, 0)]
    while stack:
        node, depth = stack.pop()
        count += 1
        observed_depth = max(observed_depth, depth)
        if max_nodes is not None and count > max_nodes:
            raise ProtocolValidationError("gather exceeds max_nodes")
        if max_depth is not None and depth > max_depth:
            raise ProtocolValidationError("gather exceeds max_depth")
        if isinstance(node, SlotTake):
            if ref_count is not None and node.ref_slot >= ref_count:
                raise ProtocolValidationError(
                    "gather ref_slot is outside the input ref table"
                )
            continue
        for child in reversed(node.children):
            stack.append((child, depth + 1))
    return count, observed_depth


def validate_worker_dispatch(
    dispatch: WorkerDispatch,
    *,
    expected_version: int = PROTOCOL_VERSION,
    input_ref_count: int | None = None,
    role_count: int | None = None,
    max_depth: int | None = None,
    max_nodes_per_entry: int | None = None,
) -> None:
    """Validate version, identity alignment, arity, and every gather tree."""

    if not isinstance(dispatch, WorkerDispatch):
        raise TypeError("dispatch must be a WorkerDispatch")
    _plain_int(expected_version, "expected_version", minimum=1)
    if dispatch.protocol_version != expected_version:
        raise ProtocolValidationError(
            f"unsupported protocol version {dispatch.protocol_version}; "
            f"expected {expected_version}"
        )
    if dispatch.lease.node != dispatch.node:
        raise ProtocolValidationError("actor lease node does not match dispatch node")
    if role_count is not None:
        _plain_int(role_count, "role_count")
    tokens = tuple(entry.token for entry in dispatch.entries)
    if len(set(tokens)) != len(tokens):
        raise ProtocolValidationError("worker dispatch attempts must be unique")
    for entry in dispatch.entries:
        if entry.token.run != dispatch.run:
            raise ProtocolValidationError("attempt token run does not match dispatch")
        if role_count is not None and len(entry.role_trees) != role_count:
            raise ProtocolValidationError(
                "worker entry role arity does not match CallSchema"
            )
        entry_nodes = 0
        for tree in entry.role_trees:
            node_count, _ = validate_wire_gather(
                tree,
                ref_count=input_ref_count,
                max_depth=max_depth,
            )
            entry_nodes += node_count
            if (
                max_nodes_per_entry is not None
                and entry_nodes > max_nodes_per_entry
            ):
                raise ProtocolValidationError(
                    "entry gathers exceed max_nodes_per_entry"
                )


def header_for(dispatch: WorkerDispatch) -> ManifestHeader:
    """Build the exact manifest header echoed by a worker."""

    return ManifestHeader(
        protocol_version=dispatch.protocol_version,
        graph_fingerprint=dispatch.graph_fingerprint,
        run=dispatch.run,
        dispatch=dispatch.dispatch,
        node=dispatch.node,
        lease=dispatch.lease,
        attempts=tuple(entry.token for entry in dispatch.entries),
    )


def manifest_size_bytes(manifest: Manifest) -> int:
    """Return the canonical defensive pickle size of a manifest."""

    if not isinstance(manifest, (SuccessManifest, FailureManifest)):
        raise TypeError("manifest must be a SuccessManifest or FailureManifest")
    try:
        return len(pickle.dumps(manifest, protocol=5))
    except Exception as exc:  # pragma: no cover - frozen DTOs should be picklable
        raise ProtocolValidationError("manifest is not serializable") from exc


def validate_output_layouts(
    layouts: tuple[OutputLayout, ...],
    expected: tuple[PhysicalOutputSpec, ...],
) -> None:
    """Match manifest layouts exactly to compiled physical output contracts."""

    if not isinstance(layouts, tuple):
        raise TypeError("layouts must be a tuple")
    if not isinstance(expected, tuple) or not expected:
        raise TypeError("expected output schema must be a non-empty tuple")
    if any(not isinstance(layout, OutputLayout) for layout in layouts):
        raise TypeError("layouts must contain OutputLayout records")
    if any(not isinstance(spec, PhysicalOutputSpec) for spec in expected):
        raise TypeError(
            "expected output schema must contain PhysicalOutputSpec records"
        )
    if len(layouts) != len(expected):
        raise ProtocolValidationError(
            "manifest output count does not match compiled output schema"
        )
    for layout, spec in zip(layouts, expected):
        if layout.return_slot != spec.return_slot:
            raise ProtocolValidationError(
                "manifest return_slot does not match compiled output schema"
            )
        if layout.port != spec.port:
            raise ProtocolValidationError(
                "manifest port does not match compiled output schema"
            )
        if layout.shape != spec.shape:
            raise ProtocolValidationError(
                "manifest shape does not match compiled output schema"
            )
        has_control_bits = layout.control_bits is not None
        if has_control_bits != spec.emit_control_bits:
            raise ProtocolValidationError(
                "manifest control_bits presence does not match compiled output schema"
            )


def validate_manifest(
    manifest: Manifest,
    *,
    max_bytes: int,
    expected_header: ManifestHeader | None = None,
    expected_output_count: int | None = None,
    expected_output_schema: tuple[PhysicalOutputSpec, ...] | None = None,
) -> int:
    """Defensively validate a manifest against pending and compiled facts."""

    _plain_int(max_bytes, "max_bytes", minimum=1)
    if not isinstance(manifest, (SuccessManifest, FailureManifest)):
        raise ProtocolValidationError("generator tail is not a manifest")
    if manifest.header.protocol_version != PROTOCOL_VERSION:
        raise ProtocolValidationError("manifest protocol version is unsupported")
    if expected_header is not None and manifest.header != expected_header:
        raise ProtocolValidationError("manifest header does not match pending dispatch")
    if expected_output_schema is not None:
        if expected_output_count is not None and (
            expected_output_count != len(expected_output_schema)
        ):
            raise ProtocolValidationError(
                "expected output count conflicts with compiled output schema"
            )
        if isinstance(manifest, SuccessManifest):
            validate_output_layouts(manifest.outputs, expected_output_schema)
        elif not expected_output_schema:
            raise ProtocolValidationError(
                "controlled failures require compiled output slots"
            )
    if expected_output_count is not None:
        _plain_int(expected_output_count, "expected_output_count", minimum=1)
        if isinstance(manifest, SuccessManifest):
            if len(manifest.outputs) != expected_output_count:
                raise ProtocolValidationError(
                    "manifest output count does not match pending dispatch"
                )
        elif expected_output_count < 1:
            raise ProtocolValidationError("controlled failures require output slots")
    size = manifest_size_bytes(manifest)
    if size > max_bytes:
        raise ProtocolValidationError(
            f"manifest size {size} exceeds configured limit {max_bytes}"
        )
    return size


def truncate_utf8(message: str, max_bytes: int) -> str:
    """Truncate text to a byte limit without emitting invalid UTF-8."""

    if not isinstance(message, str):
        raise TypeError("message must be a string")
    _plain_int(max_bytes, "max_bytes")
    encoded = message.encode("utf-8")
    if len(encoded) <= max_bytes:
        return message
    return encoded[:max_bytes].decode("utf-8", errors="ignore")
