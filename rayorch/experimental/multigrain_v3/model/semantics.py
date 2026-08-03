"""Backend-independent identities and terminal semantic facts for V3.

The records in this module deliberately contain no Ray handles and no scheduler
state.  They are safe to serialize, compare, and use as the authoritative
logical identity layer of a run.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import secrets
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping, Set
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, TypeAlias, TypeVar


class InvariantViolation(RuntimeError):
    """Raised when two observations contradict an already frozen V3 fact."""


@dataclass(frozen=True, slots=True, order=True)
class _IndexId:
    """Base for compact, non-negative IDs used inside a compiled graph/session."""

    value: int

    def __post_init__(self) -> None:
        """Validate the compact non-negative integer representation."""

        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise TypeError(f"{type(self).__name__} value must be an int")
        if self.value < 0:
            raise ValueError(f"{type(self).__name__} value must be non-negative")

    def __int__(self) -> int:
        """Return the compact integer representation."""

        return self.value


class NodeId(_IndexId):
    """Stable topological node index in one compiled graph."""

    __slots__ = ()


class PortId(_IndexId):
    """Stable logical port index in one compiled graph."""

    __slots__ = ()


class ScopeDefId(_IndexId):
    """Static scope definition allocated to an EXPAND node."""

    __slots__ = ()


class OccurrenceDomainId(_IndexId):
    """Static identity of a set of aligned occurrence coordinates."""

    __slots__ = ()


class ValueNodeId(_IndexId):
    """Session-local identifier for a node in the value DAG."""

    __slots__ = ()


class BlockId(_IndexId):
    """Session-local identifier for an opaque physical value block."""

    __slots__ = ()


@dataclass(frozen=True, slots=True, order=True)
class _DigestId:
    """Base for fixed-width, content-addressed or random logical IDs."""

    value: bytes
    WIDTH: ClassVar[int] = 16

    def __post_init__(self) -> None:
        """Validate exact byte width without accepting mutable bytearrays."""

        if not isinstance(self.value, bytes):
            raise TypeError(f"{type(self).__name__} value must be bytes")
        if len(self.value) != self.WIDTH:
            raise ValueError(
                f"{type(self).__name__} must contain exactly {self.WIDTH} bytes"
            )

    @classmethod
    def new(cls):
        """Create a cryptographically random identifier of this ID type."""

        return cls(secrets.token_bytes(cls.WIDTH))

    @classmethod
    def derive(cls, domain: str, *parts: object):
        """Derive a domain-separated ID from canonical, representation-free data."""

        if not domain or not isinstance(domain, str):
            raise ValueError("identity domain must be a non-empty string")
        preimage = (
            b"rayorch.multigrain.v3\0"
            + cls.__name__.encode("ascii")
            + b"\0"
            + domain.encode("utf-8")
            + b"\0"
            + b"".join(_encode_part(part) for part in parts)
        )
        digest = hashlib.blake2b(preimage, digest_size=cls.WIDTH).digest()
        _record_preimage(cls, digest, preimage)
        return cls(digest)

    def hex(self) -> str:
        """Return the lower-case hexadecimal form."""

        return self.value.hex()

    def __bytes__(self) -> bytes:
        """Return the raw fixed-width representation."""

        return self.value


class GraphFingerprint(_DigestId):
    """A 256-bit digest of the canonical frozen graph."""

    __slots__ = ()
    WIDTH = 32


class RunId(_DigestId):
    """Random identity of one execution of a compiled graph."""

    __slots__ = ()


class RootId(_DigestId):
    """Identity of one source occurrence and its causal descendants."""

    __slots__ = ()

    @classmethod
    def for_source(cls, run: "RunId", source_seq: int) -> "RootId":
        """Derive the root identity for a source sequence number."""

        return cls.derive("root", run, source_seq)


class EntityId(_DigestId):
    """Logical occurrence identity shared across aligned ports."""

    __slots__ = ()

    @classmethod
    def source(cls, run: RunId, source_seq: int) -> "EntityId":
        """Derive a root source entity independently of physical batching."""

        return cls.derive("source-entity", run, source_seq)

    @classmethod
    def expand(
        cls,
        run: RunId,
        node: NodeId,
        parent: "ItemRef",
        ordinal: int,
    ) -> "EntityId":
        """Derive a child entity at one EXPAND ordinal."""

        return cls.derive("expand-entity", run, node, parent, ordinal)


class GrainId(_DigestId):
    """Identity of one semantic primitive invocation."""

    __slots__ = ()

    @classmethod
    def source(cls, run: RunId, node: NodeId, source_seq: int) -> "GrainId":
        """Derive a SOURCE grain identity."""

        return cls.derive("source-grain", run, node, source_seq)

    @classmethod
    def map(
        cls,
        run: RunId,
        node: NodeId,
        ordered_inputs: tuple["ItemRef", ...],
    ) -> "GrainId":
        """Derive a MAP grain from compiled-role ordered input coordinates."""

        return cls.derive("map-grain", run, node, ordered_inputs)

    @classmethod
    def filter(
        cls,
        run: RunId,
        node: NodeId,
        mask: "ItemRef",
        target: "ItemRef",
    ) -> "GrainId":
        """Derive a FILTER grain from mask and target coordinates."""

        return cls.derive("filter-grain", run, node, mask, target)

    @classmethod
    def expand(
        cls,
        run: RunId,
        node: NodeId,
        parent: "ItemRef",
    ) -> "GrainId":
        """Derive an EXPAND grain from its structural parent coordinate."""

        return cls.derive("expand-grain", run, node, parent)

    @classmethod
    def reduce(
        cls,
        run: RunId,
        node: NodeId,
        scope: "ScopeInstanceId",
    ) -> "GrainId":
        """Derive a REDUCE grain from the closed dynamic scope."""

        return cls.derive("reduce-grain", run, node, scope)


class ScopeInstanceId(_DigestId):
    """Identity of one dynamic scope opened by EXPAND."""

    __slots__ = ()

    @classmethod
    def for_expand(
        cls,
        run: RunId,
        expand_node: NodeId,
        parent: "ItemRef",
    ) -> "ScopeInstanceId":
        """Derive a scope instance from the exact EXPAND parent coordinate."""

        return cls.derive("scope-instance", run, expand_node, parent)


class DispatchId(_DigestId):
    """Identity of one physical dispatch reservation."""

    __slots__ = ()


class LeaseId(_DigestId):
    """Identity of one physical actor lease."""

    __slots__ = ()


class StructuralLeaseId(_DigestId):
    """Identity of one structural fan-out allocation."""

    __slots__ = ()


class CreditReservationId(_DigestId):
    """Identity of one hard-count credit reservation."""

    __slots__ = ()


class ActorId(_DigestId):
    """Backend-neutral identity of an actor incarnation."""

    __slots__ = ()


class ErrorId(_DigestId):
    """Stable identity of a semantic error record."""

    __slots__ = ()

    @classmethod
    def for_grain(
        cls,
        run: RunId,
        grain: GrainId,
        kind: str,
    ) -> "ErrorId":
        """Derive an error identity without embedding message formatting."""

        return cls.derive("grain-error", run, grain, kind)


_I = TypeVar("_I", bound=_DigestId)
_PREIMAGE_LOCK = threading.Lock()
PREIMAGE_DEBUG_LEDGER_LIMIT = 1024
"""Maximum recent digest preimages retained for collision diagnostics."""

_DIGEST_PREIMAGES: OrderedDict[
    tuple[type[_DigestId], bytes],
    bytes,
] = OrderedDict()


def _record_preimage(
    identity_type: type[_I],
    digest: bytes,
    preimage: bytes,
) -> None:
    """Detect recent collisions without retaining unbounded run history."""

    if PREIMAGE_DEBUG_LEDGER_LIMIT <= 0:
        return
    key = (identity_type, digest)
    with _PREIMAGE_LOCK:
        previous = _DIGEST_PREIMAGES.get(key)
        if previous is None:
            _DIGEST_PREIMAGES[key] = preimage
            while len(_DIGEST_PREIMAGES) > PREIMAGE_DEBUG_LEDGER_LIMIT:
                _DIGEST_PREIMAGES.popitem(last=False)
            return
        if previous != preimage:
            raise InvariantViolation(
                f"digest collision detected for {identity_type.__name__}"
            )
        _DIGEST_PREIMAGES.move_to_end(key)


def preimage_debug_ledger_size() -> int:
    """Return the bounded recent-preimage count for tests and diagnostics."""

    with _PREIMAGE_LOCK:
        return len(_DIGEST_PREIMAGES)


def canonical_identity_bytes(value: object) -> bytes:
    """Expose the canonical encoder for frozen configuration ordering."""

    return _encode_part(value)


def _encode_part(value: object) -> bytes:
    """Encode supported identity inputs without ``repr`` or Python ``hash``."""

    if value is None:
        return b"n;"
    if value is True:
        return b"b1;"
    if value is False:
        return b"b0;"
    if isinstance(value, int) and not isinstance(value, bool):
        payload = str(value).encode("ascii")
        return b"i" + str(len(payload)).encode("ascii") + b":" + payload
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floats are not canonical identity inputs")
        payload = value.hex().encode("ascii")
        return b"f" + str(len(payload)).encode("ascii") + b":" + payload
    if isinstance(value, str):
        payload = value.encode("utf-8")
        return b"s" + str(len(payload)).encode("ascii") + b":" + payload
    if isinstance(value, bytes):
        return b"y" + str(len(value)).encode("ascii") + b":" + value
    if isinstance(value, _IndexId):
        return (
            b"x"
            + type(value).__name__.encode("ascii")
            + b":"
            + _encode_part(value.value)
        )
    if isinstance(value, _DigestId):
        return (
            b"d"
            + type(value).__name__.encode("ascii")
            + b":"
            + _encode_part(value.value)
        )
    if isinstance(value, Enum):
        return (
            b"e"
            + type(value).__module__.encode("utf-8")
            + b"."
            + type(value).__qualname__.encode("utf-8")
            + b":"
            + _encode_part(value.value)
        )
    if isinstance(value, Mapping):
        encoded = [
            (_encode_part(key), _encode_part(item))
            for key, item in value.items()
        ]
        encoded.sort(key=lambda pair: pair[0])
        body = b"".join(key + item for key, item in encoded)
        return b"m" + str(len(encoded)).encode("ascii") + b":" + body
    if isinstance(value, Set) and not isinstance(value, (str, bytes)):
        encoded_items = sorted(_encode_part(item) for item in value)
        body = b"".join(encoded_items)
        return b"q" + str(len(encoded_items)).encode("ascii") + b":" + body
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        body = b"".join(
            _encode_part(data_field.name)
            + _encode_part(getattr(value, data_field.name))
            for data_field in dataclasses.fields(value)
        )
        type_name = (
            type(value).__module__ + "." + type(value).__qualname__
        ).encode("utf-8")
        return b"c" + _encode_part(type_name) + _encode_part(body)
    if isinstance(value, (tuple, list)):
        body = b"".join(_encode_part(item) for item in value)
        prefix = b"t" if isinstance(value, tuple) else b"l"
        return prefix + str(len(value)).encode("ascii") + b":" + body
    raise TypeError(
        f"{type(value).__module__}.{type(value).__qualname__} "
        "is not a canonical identity input"
    )


@dataclass(frozen=True, slots=True)
class ItemRef:
    """A logical output coordinate, whether or not a value is present."""

    port: PortId
    entity: EntityId


@dataclass(frozen=True, slots=True)
class ScopePosition:
    """One dynamic scope instance and the child ordinal within it."""

    instance: ScopeInstanceId
    ordinal: int

    def __post_init__(self) -> None:
        """Validate a non-negative dynamic child ordinal."""

        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int):
            raise TypeError("scope ordinal must be an int")
        if self.ordinal < 0:
            raise ValueError("scope ordinal must be non-negative")


@dataclass(frozen=True, slots=True)
class OccurrenceContext:
    """Root plus the full outer-to-inner dynamic scope position stack."""

    root: RootId
    positions: tuple[ScopePosition, ...] = ()

    def push(
        self,
        instance: ScopeInstanceId,
        ordinal: int,
    ) -> "OccurrenceContext":
        """Append exactly one EXPAND position and return a new context."""

        return OccurrenceContext(
            self.root,
            self.positions + (ScopePosition(instance, ordinal),),
        )

    def pop(self) -> tuple["OccurrenceContext", ScopePosition]:
        """Remove exactly one innermost REDUCE position."""

        if not self.positions:
            raise InvariantViolation("cannot pop the root occurrence context")
        return (
            OccurrenceContext(self.root, self.positions[:-1]),
            self.positions[-1],
        )

    def validate_scope_path(
        self,
        scope_path: tuple[ScopeDefId, ...],
        definition_of: Callable[[ScopeInstanceId], ScopeDefId],
    ) -> None:
        """Validate the dynamic stack against a frozen port scope path."""

        definitions = tuple(
            definition_of(position.instance) for position in self.positions
        )
        if definitions != scope_path:
            raise InvariantViolation(
                "occurrence positions do not match the port scope path"
            )


class ReceiptState(Enum):
    """Terminal state of one logical output coordinate."""

    PRESENT = "present"
    NORMAL_ABSENCE = "normal_absence"
    FAILED = "failed"
    SUPPRESSED = "suppressed"

    @property
    def is_failure(self) -> bool:
        """Whether the state belongs to the failure lineage."""

        return self in {ReceiptState.FAILED, ReceiptState.SUPPRESSED}

    @property
    def is_absent(self) -> bool:
        """Whether the state is normal data absence."""

        return self is ReceiptState.NORMAL_ABSENCE


@dataclass(frozen=True, slots=True)
class Receipt:
    """Authoritative terminal fact for one :class:`ItemRef` coordinate."""

    item: ItemRef
    context: OccurrenceContext
    state: ReceiptState
    producer: GrainId


@dataclass(slots=True)
class ReceiptStore:
    """Publish-once receipt index with root-local reclamation."""

    binding_exists: Callable[[ItemRef], bool] | None = None
    _receipts: dict[ItemRef, Receipt] = field(default_factory=dict, init=False)
    _by_root: dict[RootId, set[ItemRef]] = field(
        default_factory=dict,
        init=False,
    )

    def publish_once(self, receipt: Receipt) -> bool:
        """Publish a terminal receipt, returning false for an exact duplicate."""

        previous = self._receipts.get(receipt.item)
        if previous is not None:
            if previous == receipt:
                return False
            raise InvariantViolation(
                f"conflicting receipt for coordinate {receipt.item}"
            )
        if self.binding_exists is not None:
            has_binding = self.binding_exists(receipt.item)
            if receipt.state is ReceiptState.PRESENT and not has_binding:
                raise InvariantViolation(
                    "PRESENT receipt requires an installed value binding"
                )
            if receipt.state is not ReceiptState.PRESENT and has_binding:
                raise InvariantViolation(
                    "non-PRESENT receipt cannot have a value binding"
                )
        self._receipts[receipt.item] = receipt
        self._by_root.setdefault(receipt.context.root, set()).add(receipt.item)
        return True

    def get(self, item: ItemRef) -> Receipt | None:
        """Return the receipt for a coordinate, if terminal."""

        return self._receipts.get(item)

    def require(self, item: ItemRef) -> Receipt:
        """Return a terminal receipt or raise for a missing coordinate."""

        try:
            return self._receipts[item]
        except KeyError as error:
            raise KeyError(f"no terminal receipt for {item}") from error

    def remove_root(self, root: RootId) -> None:
        """Remove all receipts indexed to one reclaimed root."""

        for item in self._by_root.pop(root, set()):
            self._receipts.pop(item, None)

    def __len__(self) -> int:
        """Return the number of live terminal coordinates."""

        return len(self._receipts)


@dataclass(frozen=True, slots=True)
class RoleBinding:
    """Compiled input role and its ordered logical coordinates."""

    role: str
    items: tuple[ItemRef, ...]

    def __post_init__(self) -> None:
        """Validate the stable non-empty compiled role name."""

        if not self.role:
            raise ValueError("role name must be non-empty")


@dataclass(frozen=True, slots=True)
class ExpandOutputRange:
    """Compact deterministic coordinate range produced by EXPAND."""

    port: PortId
    parent: ItemRef
    count: int

    def __post_init__(self) -> None:
        """Validate the deterministic non-negative fan-out count."""

        if isinstance(self.count, bool) or not isinstance(self.count, int):
            raise TypeError("expand output count must be an int")
        if self.count < 0:
            raise ValueError("expand output count must be non-negative")


OutputSlotSpec: TypeAlias = tuple[ItemRef, ...] | ExpandOutputRange


@dataclass(frozen=True, slots=True)
class GrainSpec:
    """Immutable declaration of one logical primitive invocation."""

    id: GrainId
    run: RunId
    root: RootId
    node: NodeId
    context: OccurrenceContext
    inputs: tuple[RoleBinding, ...]
    output_slots: OutputSlotSpec
    reduce_binding: object | None = None

    def __post_init__(self) -> None:
        """Validate root consistency, role uniqueness, and output cardinality."""

        if self.context.root != self.root:
            raise ValueError("grain context root must match GrainSpec.root")
        roles = tuple(binding.role for binding in self.inputs)
        if len(roles) != len(set(roles)):
            raise ValueError("grain input roles must be unique")
        if isinstance(self.output_slots, tuple) and not self.output_slots:
            raise ValueError("fixed grain output slots must be non-empty")


@dataclass(frozen=True, slots=True)
class Success:
    """Outcome of a grain whose semantic operation completed normally."""


@dataclass(frozen=True, slots=True)
class Skipped:
    """Outcome of a grain skipped because required coordinates were absent."""

    reasons: tuple[ItemRef, ...]

    def __post_init__(self) -> None:
        """Require canonical non-empty absent-input reasons."""

        if not self.reasons:
            raise ValueError("Skipped requires at least one absent input")
        if len(self.reasons) != len(set(self.reasons)):
            raise ValueError("Skipped reasons must be unique and canonical")


@dataclass(frozen=True, slots=True)
class Failed:
    """Outcome of a grain that directly produced a semantic error."""

    error: ErrorId


@dataclass(frozen=True, slots=True)
class Suppressed:
    """Outcome blocked by canonical direct predecessor failures."""

    causes: tuple[GrainId, ...]

    def __post_init__(self) -> None:
        """Require canonical non-empty direct predecessor causes."""

        if not self.causes:
            raise ValueError("Suppressed requires at least one direct cause")
        if len(self.causes) != len(set(self.causes)):
            raise ValueError("Suppressed causes must be unique and canonical")


GrainOutcome: TypeAlias = Success | Skipped | Failed | Suppressed


class GrainPhase(Enum):
    """Mutable scheduling phase of a logical grain."""

    READY = "ready"
    IN_FLIGHT = "in_flight"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class AttemptToken:
    """Exact publication authority for one MAP retry generation."""

    run: RunId
    grain: GrainId
    generation: int

    def __post_init__(self) -> None:
        """Require a positive retry generation."""

        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
        ):
            raise TypeError("attempt generation must be an int")
        if self.generation <= 0:
            raise ValueError("attempt generation must be positive")


@dataclass(slots=True)
class GrainState:
    """Mutable phase and outcome attached to an immutable GrainSpec."""

    spec: GrainSpec
    phase: GrainPhase
    generation: int = 0
    active_attempt: AttemptToken | None = None
    outcome: GrainOutcome | None = None

    def __post_init__(self) -> None:
        """Validate phase, generation, attempt authority, and outcome coupling."""

        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
        ):
            raise ValueError("grain generation must be non-negative")
        if self.phase is GrainPhase.TERMINAL and self.outcome is None:
            raise ValueError("terminal grain state requires an outcome")
        if self.phase is not GrainPhase.TERMINAL and self.outcome is not None:
            raise ValueError("non-terminal grain state cannot have an outcome")
        if (
            self.phase is GrainPhase.IN_FLIGHT
            and self.active_attempt is None
        ):
            raise ValueError("in-flight grain requires an active attempt")
        if (
            self.phase is not GrainPhase.IN_FLIGHT
            and self.active_attempt is not None
        ):
            raise ValueError("only in-flight grains may retain an attempt")


@dataclass(slots=True)
class GrainStore:
    """Single-writer grain lifecycle store with stale-attempt rejection."""

    _states: dict[GrainId, GrainState] = field(
        default_factory=dict,
        init=False,
    )
    _by_root: dict[RootId, set[GrainId]] = field(
        default_factory=dict,
        init=False,
    )

    def ensure(
        self,
        spec: GrainSpec,
        *,
        phase: GrainPhase = GrainPhase.READY,
    ) -> GrainState:
        """Install a grain exactly once or return its identical existing state."""

        if phase is not GrainPhase.READY:
            raise ValueError("ensure() creates READY grains; use reserve/seal")
        previous = self._states.get(spec.id)
        if previous is not None:
            if previous.spec != spec:
                raise InvariantViolation(
                    f"conflicting GrainSpec for {spec.id.hex()}"
                )
            return previous
        state = GrainState(spec=spec, phase=phase)
        self._states[spec.id] = state
        self._by_root.setdefault(spec.root, set()).add(spec.id)
        return state

    def get(self, grain: GrainId) -> GrainState | None:
        """Return a live grain state, if known."""

        return self._states.get(grain)

    def require(self, grain: GrainId) -> GrainState:
        """Return a grain state or raise for an unknown ID."""

        try:
            return self._states[grain]
        except KeyError as error:
            raise KeyError(f"unknown grain {grain.hex()}") from error

    def reserve(self, grain: GrainId) -> AttemptToken:
        """Move a READY MAP grain in flight with a fresh generation token."""

        state = self.require(grain)
        if state.phase is not GrainPhase.READY:
            raise InvariantViolation("only READY grains can be reserved")
        state.generation += 1
        token = AttemptToken(state.spec.run, grain, state.generation)
        state.active_attempt = token
        state.phase = GrainPhase.IN_FLIGHT
        return token

    def is_current(self, token: AttemptToken) -> bool:
        """Whether a token is the exact active publication authority."""

        state = self._states.get(token.grain)
        return bool(
            state is not None
            and state.spec.run == token.run
            and state.phase is GrainPhase.IN_FLIGHT
            and state.active_attempt == token
        )

    def retry(self, token: AttemptToken) -> None:
        """Return the current in-flight grain to READY without reusing a token."""

        if not self.is_current(token):
            raise InvariantViolation("cannot retry a stale attempt token")
        state = self._states[token.grain]
        state.phase = GrainPhase.READY
        state.active_attempt = None

    def seal(
        self,
        authority: AttemptToken | GrainId,
        outcome: GrainOutcome,
    ) -> bool:
        """Seal one outcome; return false for an exact terminal duplicate."""

        grain = (
            authority.grain
            if isinstance(authority, AttemptToken)
            else authority
        )
        state = self.require(grain)
        if state.phase is GrainPhase.TERMINAL:
            if state.outcome == outcome:
                return False
            raise InvariantViolation("terminal GrainOutcome cannot be changed")
        if isinstance(authority, AttemptToken):
            if not self.is_current(authority):
                raise InvariantViolation("stale attempt has no publication right")
        elif state.phase is GrainPhase.IN_FLIGHT:
            raise InvariantViolation(
                "in-flight MAP grain requires its current AttemptToken"
            )
        self._validate_outcome(state, outcome)
        state.phase = GrainPhase.TERMINAL
        state.active_attempt = None
        state.outcome = outcome
        return True

    def _validate_outcome(
        self,
        state: GrainState,
        outcome: GrainOutcome,
    ) -> None:
        """Validate lineage facts that require access to predecessor grains."""

        if isinstance(outcome, Skipped):
            expected_inputs = {
                item
                for binding in state.spec.inputs
                for item in binding.items
            }
            reduce_binding = state.spec.reduce_binding
            origin_receipt = getattr(
                reduce_binding,
                "origin_receipt",
                None,
            )
            if isinstance(origin_receipt, Receipt):
                expected_inputs.add(origin_receipt.item)
            if any(reason not in expected_inputs for reason in outcome.reasons):
                raise InvariantViolation(
                    "Skipped reason is not an input of the sealed grain"
                )
        if not isinstance(outcome, Suppressed):
            return
        reduce_binding = state.spec.reduce_binding
        compact_causes = {
            cause
            for cause in (
                getattr(reduce_binding, "origin_failure", None),
                *(
                    failure
                    for _, failure in getattr(
                        reduce_binding,
                        "direct_failures",
                        (),
                    )
                ),
            )
            if isinstance(cause, GrainId)
        }
        for cause in outcome.causes:
            if cause == state.spec.id:
                raise InvariantViolation("grain cannot suppress itself")
            predecessor = self._states.get(cause)
            if predecessor is None:
                if cause in compact_causes:
                    continue
                raise InvariantViolation(
                    "Suppressed cause must name a known predecessor grain"
                )
            if predecessor.spec.root != state.spec.root:
                raise InvariantViolation(
                    "Suppressed causes must belong to the same root"
                )
            if predecessor.phase is not GrainPhase.TERMINAL:
                raise InvariantViolation(
                    "Suppressed cause must already be terminal"
                )
            if self._cause_reaches(cause, state.spec.id):
                raise InvariantViolation("Suppressed cause DAG contains a cycle")

    def _cause_reaches(self, start: GrainId, target: GrainId) -> bool:
        """Return whether an existing suppressed-cause DAG reaches a grain."""

        pending = [start]
        visited: set[GrainId] = set()
        while pending:
            current = pending.pop()
            if current == target:
                return True
            if current in visited:
                continue
            visited.add(current)
            state = self._states.get(current)
            if state is not None and isinstance(state.outcome, Suppressed):
                pending.extend(state.outcome.causes)
        return False

    def states_for_root(self, root: RootId) -> tuple[GrainState, ...]:
        """Return root grains in deterministic GrainId byte order."""

        ids = sorted(self._by_root.get(root, ()), key=bytes)
        return tuple(self._states[grain] for grain in ids)

    def remove_root(self, root: RootId) -> None:
        """Remove all detailed grain state for a reclaimed root."""

        for grain in self._by_root.pop(root, set()):
            self._states.pop(grain, None)

    def __len__(self) -> int:
        """Return the number of live grain records."""

        return len(self._states)


@dataclass(frozen=True, slots=True)
class ErrorRecord:
    """Detached description of one directly failed grain."""

    id: ErrorId
    root: RootId
    grain: GrainId
    kind: str
    message: str
    trace_digest: bytes | None = None

    def __post_init__(self) -> None:
        """Validate detached and bounded error metadata types."""

        if not self.kind:
            raise ValueError("error kind must be non-empty")
        if not isinstance(self.message, str):
            raise TypeError("error message must be a string")
        if self.trace_digest is not None and not isinstance(
            self.trace_digest,
            bytes,
        ):
            raise TypeError("trace_digest must be bytes or None")


@dataclass(frozen=True, slots=True)
class SemanticFailureSummary:
    """Self-contained semantic failure data retained after root reclamation."""

    root: RootId
    errors: tuple[ErrorRecord, ...]
    failed_grains: tuple[GrainId, ...]
    suppressed_grains: tuple[tuple[GrainId, tuple[GrainId, ...]], ...]


@dataclass(frozen=True, slots=True)
class DeliveryFailureSummary:
    """Failure while detaching or delivering an otherwise terminal result."""

    root: RootId
    kind: str
    message: str


FailureSummary: TypeAlias = SemanticFailureSummary | DeliveryFailureSummary


@dataclass(slots=True)
class ErrorStore:
    """Publish-once semantic errors with root-local indexing and summaries."""

    _errors: dict[ErrorId, ErrorRecord] = field(
        default_factory=dict,
        init=False,
    )
    _by_root: dict[RootId, set[ErrorId]] = field(
        default_factory=dict,
        init=False,
    )

    def put(self, record: ErrorRecord) -> bool:
        """Install an error, returning false for an exact duplicate."""

        previous = self._errors.get(record.id)
        if previous is not None:
            if previous == record:
                return False
            raise InvariantViolation(
                f"conflicting ErrorRecord for {record.id.hex()}"
            )
        self._errors[record.id] = record
        self._by_root.setdefault(record.root, set()).add(record.id)
        return True

    def get(self, error: ErrorId) -> ErrorRecord:
        """Return an error record by ID."""

        try:
            return self._errors[error]
        except KeyError as missing:
            raise KeyError(f"unknown error {error.hex()}") from missing

    def summarize_root(
        self,
        root: RootId,
        grains: GrainStore | None = None,
    ) -> SemanticFailureSummary:
        """Materialize a summary independent of live stores."""

        errors = tuple(
            sorted(
                (
                    self._errors[error]
                    for error in self._by_root.get(root, ())
                ),
                key=lambda record: bytes(record.id),
            )
        )
        failed: list[GrainId] = []
        suppressed: list[tuple[GrainId, tuple[GrainId, ...]]] = []
        if grains is not None:
            for state in grains.states_for_root(root):
                if isinstance(state.outcome, Failed):
                    failed.append(state.spec.id)
                elif isinstance(state.outcome, Suppressed):
                    suppressed.append(
                        (state.spec.id, state.outcome.causes)
                    )
        return SemanticFailureSummary(
            root=root,
            errors=errors,
            failed_grains=tuple(failed),
            suppressed_grains=tuple(suppressed),
        )

    def remove_root(self, root: RootId) -> None:
        """Remove detailed errors after a detached summary has been built."""

        for error in self._by_root.pop(root, set()):
            self._errors.pop(error, None)

    def __len__(self) -> int:
        """Return the number of live error records."""

        return len(self._errors)


__all__ = [
    "ActorId",
    "AttemptToken",
    "BlockId",
    "CreditReservationId",
    "DeliveryFailureSummary",
    "DispatchId",
    "EntityId",
    "ErrorId",
    "ErrorRecord",
    "ErrorStore",
    "ExpandOutputRange",
    "Failed",
    "FailureSummary",
    "GrainId",
    "GrainOutcome",
    "GrainPhase",
    "GrainSpec",
    "GrainState",
    "GrainStore",
    "GraphFingerprint",
    "InvariantViolation",
    "ItemRef",
    "LeaseId",
    "NodeId",
    "OccurrenceContext",
    "OccurrenceDomainId",
    "OutputSlotSpec",
    "PortId",
    "Receipt",
    "ReceiptState",
    "ReceiptStore",
    "RoleBinding",
    "RootId",
    "RunId",
    "ScopeDefId",
    "ScopeInstanceId",
    "ScopePosition",
    "SemanticFailureSummary",
    "Skipped",
    "StructuralLeaseId",
    "Success",
    "Suppressed",
    "ValueNodeId",
]
