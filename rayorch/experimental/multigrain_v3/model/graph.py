"""Authoring trace, strict UDF schema inference, and frozen V3 graph IR."""

from __future__ import annotations

import contextvars
import inspect
import math
import pickle
import sys
import types
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import (
    Annotated,
    Any,
    ClassVar,
    TypeAlias,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

from .semantics import (
    GraphFingerprint,
    NodeId,
    OccurrenceDomainId,
    PortId,
    ScopeDefId,
    canonical_identity_bytes,
)


class CompileError(ValueError):
    """A stable, categorized rejection of an invalid authoring graph."""

    def __init__(self, code: str, detail: str | None = None) -> None:
        """Initialize a machine-readable category and human-readable detail."""

        if not code:
            raise ValueError("CompileError code must be non-empty")
        self.code = code
        self.detail = detail or code
        super().__init__(
            f"{code}: {self.detail}" if detail is not None else self.detail
        )


class OpaqueValue:
    """``Annotated`` marker preventing a Python list item from being EXPANDable."""


@dataclass(frozen=True, slots=True)
class TypeRef:
    """Canonical importable Python type reference used in graph fingerprints."""

    module: str
    qualname: str
    args: tuple["TypeRef", ...] = ()

    def __post_init__(self) -> None:
        """Validate an importable, non-local canonical reference."""

        if not self.module or not self.qualname:
            raise ValueError("TypeRef module and qualname must be non-empty")
        if "<locals>" in self.qualname:
            raise ValueError("TypeRef cannot name a local class")

    def resolve(self) -> object:
        """Resolve the unparameterized referenced object from loaded modules."""

        if self.module == "typing" and self.qualname == "Any":
            return Any
        if self.module == "builtins" and self.qualname == "ellipsis":
            return Ellipsis
        try:
            value: object = __import__(self.module, fromlist=["*"])
            for component in self.qualname.split("."):
                value = getattr(value, component)
            return value
        except (ImportError, AttributeError) as error:
            raise LookupError(
                f"cannot resolve {self.module}.{self.qualname}"
            ) from error


@dataclass(frozen=True, slots=True)
class OpaqueShape:
    """Opaque per-grain payload; ``None`` is the compile-time Any wildcard."""

    type_ref: TypeRef | None


@dataclass(frozen=True, slots=True)
class BoolShape:
    """Strict bool control/payload value shape."""


@dataclass(frozen=True, slots=True)
class StructuralListShape:
    """Exactly one explicitly represented outer structural list layer."""

    element: "ValueShape"


ValueShape: TypeAlias = OpaqueShape | BoolShape | StructuralListShape


@dataclass(frozen=True, slots=True)
class _FrozenConfigMapping(Mapping[object, object]):
    """Canonical immutable mapping used inside graph configuration records."""

    entries: tuple[tuple[object, object], ...]

    def __getitem__(self, key: object) -> object:
        """Return one frozen value by its frozen key."""

        for candidate, value in self.entries:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        """Iterate frozen keys in canonical byte order."""

        return (key for key, _ in self.entries)

    def __len__(self) -> int:
        """Return the number of frozen mapping entries."""

        return len(self.entries)


@dataclass(frozen=True, slots=True)
class _FrozenConfigList:
    """Immutable encoding of a mutable Python list."""

    items: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _FrozenConfigSet:
    """Canonical immutable encoding of a set or frozenset."""

    items: tuple[object, ...]
    restore_frozen: bool

    def __contains__(self, value: object) -> bool:
        """Return whether a frozen item is present."""

        return value in self.items

    def __iter__(self):
        """Iterate items in canonical byte order."""

        return iter(self.items)

    def __len__(self) -> int:
        """Return the number of frozen set items."""

        return len(self.items)


@dataclass(frozen=True, slots=True)
class _FrozenPickleValue:
    """Immutable serialized fallback for non-container constructor leaves."""

    payload: bytes


def freeze_config_value(value: object) -> object:
    """Canonicalize nested config into immutable, pickle-safe graph values.

    Dict/list/set containers are represented by frozen DTOs; tuples retain
    tuple shape with recursively frozen children. Other serializable leaves are
    captured as pickle bytes so later mutation cannot affect graph identity.
    Cyclic containers are rejected because they have no finite canonical form.
    """

    return _freeze_config_value(value, {}, set())


def _freeze_config_value(
    value: object,
    memo: dict[int, object],
    active: set[int],
) -> object:
    """Recursive implementation preserving aliases between acyclic containers."""

    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(
        value,
        (
            _FrozenConfigMapping,
            _FrozenConfigList,
            _FrozenConfigSet,
            _FrozenPickleValue,
        ),
    ):
        return value
    if isinstance(value, tuple):
        identity = id(value)
        if identity in active:
            raise TypeError("cyclic configuration values are not supported")
        if identity in memo:
            return memo[identity]
        active.add(identity)
        frozen = tuple(
            _freeze_config_value(item, memo, active) for item in value
        )
        active.remove(identity)
        memo[identity] = frozen
        return frozen
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise TypeError("cyclic configuration values are not supported")
        if identity in memo:
            return memo[identity]
        active.add(identity)
        entries = tuple(
            (
                _freeze_config_value(key, memo, active),
                _freeze_config_value(item, memo, active),
            )
            for key, item in value.items()
        )
        active.remove(identity)
        entries = tuple(
            sorted(entries, key=lambda pair: canonical_identity_bytes(pair[0]))
        )
        frozen_mapping = _FrozenConfigMapping(entries)
        memo[identity] = frozen_mapping
        return frozen_mapping
    if isinstance(value, list):
        identity = id(value)
        if identity in active:
            raise TypeError("cyclic configuration values are not supported")
        if identity in memo:
            return memo[identity]
        active.add(identity)
        items = tuple(
            _freeze_config_value(item, memo, active) for item in value
        )
        active.remove(identity)
        frozen_list = _FrozenConfigList(items)
        memo[identity] = frozen_list
        return frozen_list
    if isinstance(value, (set, frozenset)):
        identity = id(value)
        if identity in active:
            raise TypeError("cyclic configuration values are not supported")
        if identity in memo:
            return memo[identity]
        active.add(identity)
        items = tuple(
            _freeze_config_value(item, memo, active) for item in value
        )
        active.remove(identity)
        items = tuple(sorted(items, key=canonical_identity_bytes))
        frozen_set = _FrozenConfigSet(
            items,
            restore_frozen=isinstance(value, frozenset),
        )
        memo[identity] = frozen_set
        return frozen_set
    try:
        payload = pickle.dumps(value, protocol=5)
    except Exception as error:
        raise TypeError(
            f"{type(value).__module__}.{type(value).__qualname__} "
            "is not a serializable configuration value"
        ) from error
    return _FrozenPickleValue(payload)


def thaw_config_value(value: object) -> object:
    """Restore ordinary Python containers from a canonical frozen config."""

    return _thaw_config_value(value, {})


def _thaw_config_value(
    value: object,
    memo: dict[int, object],
) -> object:
    """Recursive thaw implementation preserving frozen-container aliases."""

    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if isinstance(value, _FrozenPickleValue):
        restored = pickle.loads(value.payload)
        memo[identity] = restored
        return restored
    if isinstance(value, _FrozenConfigList):
        restored_list: list[object] = []
        memo[identity] = restored_list
        restored_list.extend(
            _thaw_config_value(item, memo) for item in value.items
        )
        return restored_list
    if isinstance(value, _FrozenConfigMapping):
        restored_mapping: dict[object, object] = {}
        memo[identity] = restored_mapping
        for key, item in value.entries:
            restored_mapping[
                _thaw_config_value(key, memo)
            ] = _thaw_config_value(item, memo)
        return restored_mapping
    if isinstance(value, _FrozenConfigSet):
        restored_items = {
            _thaw_config_value(item, memo) for item in value.items
        }
        restored_set: object = (
            frozenset(restored_items)
            if value.restore_frozen
            else restored_items
        )
        memo[identity] = restored_set
        return restored_set
    if isinstance(value, tuple):
        restored_tuple = tuple(
            _thaw_config_value(item, memo) for item in value
        )
        memo[identity] = restored_tuple
        return restored_tuple
    if isinstance(value, Mapping):
        restored_mapping = {
            _thaw_config_value(key, memo):
                _thaw_config_value(item, memo)
            for key, item in value.items()
        }
        memo[identity] = restored_mapping
        return restored_mapping
    if isinstance(value, list):
        restored_list = [
            _thaw_config_value(item, memo) for item in value
        ]
        memo[identity] = restored_list
        return restored_list
    if isinstance(value, (set, frozenset)):
        restored_items = {
            _thaw_config_value(item, memo) for item in value
        }
        restored_set = (
            frozenset(restored_items)
            if isinstance(value, frozenset)
            else restored_items
        )
        memo[identity] = restored_set
        return restored_set
    return value


def freeze_runtime_env(
    runtime_env: Mapping[str, object] | None,
) -> tuple[tuple[str, object], ...]:
    """Freeze a Ray runtime environment with canonical top-level key order."""

    if runtime_env is None:
        return ()
    if not isinstance(runtime_env, Mapping):
        raise TypeError("runtime_env must be a mapping or None")
    if any(
        not isinstance(key, str) or not key
        for key in runtime_env
    ):
        raise ValueError("runtime_env keys must be non-empty strings")
    frozen = freeze_config_value(dict(runtime_env))
    assert isinstance(frozen, _FrozenConfigMapping)
    return tuple(
        (typing.cast(str, key), item) for key, item in frozen.entries
    )


def freeze_constructor_arguments(
    init_args: tuple[object, ...],
    init_kwargs: tuple[tuple[str, object], ...],
) -> tuple[tuple[object, ...], tuple[tuple[str, object], ...]]:
    """Freeze one constructor call while preserving cross-argument aliases."""

    keys = tuple(key for key, _ in init_kwargs)
    if any(not isinstance(key, str) for key in keys):
        raise TypeError("constructor keyword names must be strings")
    if len(keys) != len(set(keys)):
        raise ValueError("constructor keyword names must be unique")
    frozen = freeze_config_value(
        (tuple(init_args), dict(init_kwargs))
    )
    assert isinstance(frozen, tuple)
    frozen_args, frozen_kwargs = frozen
    assert isinstance(frozen_args, tuple)
    assert isinstance(frozen_kwargs, _FrozenConfigMapping)
    return frozen_args, tuple(
        (typing.cast(str, key), item)
        for key, item in frozen_kwargs.entries
    )


def freeze_resource_spec(resources: "ResourceSpec") -> "ResourceSpec":
    """Return a ResourceSpec with recursively immutable runtime environment."""

    return ResourceSpec(
        replicas=resources.replicas,
        num_cpus=resources.num_cpus,
        num_gpus=resources.num_gpus,
        runtime_env=freeze_runtime_env(dict(resources.runtime_env)),
    )


_K = TypeVar("_K")
_V = TypeVar("_V")


@dataclass(frozen=True, slots=True)
class _FrozenMapping(Mapping[_K, _V]):
    """Small immutable and pickle-safe mapping used by frozen graph indexes."""

    entries: tuple[tuple[_K, _V], ...]

    def __post_init__(self) -> None:
        """Reject duplicate keys that would make lookup ambiguous."""

        keys = tuple(key for key, _ in self.entries)
        if len(keys) != len(set(keys)):
            raise ValueError("frozen mapping keys must be unique")

    @classmethod
    def from_mapping(cls, values: Mapping[_K, _V]) -> "_FrozenMapping[_K, _V]":
        """Snapshot mapping iteration order into immutable tuple storage."""

        return cls(tuple(values.items()))

    def __getitem__(self, key: _K) -> _V:
        """Return a value by key."""

        for candidate, value in self.entries:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        """Iterate keys in frozen construction order."""

        return (key for key, _ in self.entries)

    def __len__(self) -> int:
        """Return the number of frozen key/value pairs."""

        return len(self.entries)


@dataclass(frozen=True, slots=True)
class SerializableCallableRef:
    """Importable module/qualname reference to a UDF class."""

    module: str
    qualname: str

    def __post_init__(self) -> None:
        """Validate an importable, non-local callable reference."""

        if not self.module or not self.qualname:
            raise ValueError("callable reference must be non-empty")
        if self.module in {"__main__", "__mp_main__"}:
            raise ValueError(
                "callable reference cannot target a process entry module"
            )
        if "<locals>" in self.qualname:
            raise ValueError("callable reference cannot target a local class")

    def resolve(self) -> object:
        """Resolve and return the referenced class/callable."""

        try:
            value: object = __import__(self.module, fromlist=["*"])
            for component in self.qualname.split("."):
                value = getattr(value, component)
            return value
        except (ImportError, AttributeError) as error:
            raise LookupError(
                f"cannot resolve {self.module}.{self.qualname}"
            ) from error


@dataclass(frozen=True, slots=True)
class BatchPolicy:
    """Immutable MAP batching policy."""

    max_size: int = 1
    max_wait_ms: float = 2.0

    def __post_init__(self) -> None:
        """Validate positive batch width and finite non-negative wait."""

        if (
            isinstance(self.max_size, bool)
            or not isinstance(self.max_size, int)
            or self.max_size <= 0
        ):
            raise ValueError("batch max_size must be a positive integer")
        if (
            isinstance(self.max_wait_ms, bool)
            or not isinstance(self.max_wait_ms, (int, float))
            or not math.isfinite(self.max_wait_ms)
            or self.max_wait_ms < 0
        ):
            raise ValueError("batch max_wait_ms must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    """Immutable per-MAP actor resource request."""

    replicas: int = 1
    num_cpus: float = 1.0
    num_gpus: float = 0.0
    runtime_env: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        """Validate actor counts, finite resources, and environment key order."""

        if (
            isinstance(self.replicas, bool)
            or not isinstance(self.replicas, int)
            or self.replicas <= 0
        ):
            raise ValueError("replicas must be a positive integer")
        if (
            isinstance(self.num_cpus, bool)
            or isinstance(self.num_gpus, bool)
            or not isinstance(self.num_cpus, (int, float))
            or not isinstance(self.num_gpus, (int, float))
            or not math.isfinite(self.num_cpus)
            or not math.isfinite(self.num_gpus)
            or self.num_cpus < 0
            or self.num_gpus < 0
        ):
            raise ValueError("CPU/GPU resources must be finite and non-negative")
        keys = tuple(key for key, _ in self.runtime_env)
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("runtime_env keys must be non-empty strings")
        if len(keys) != len(set(keys)):
            raise ValueError("runtime_env keys must be unique")


@dataclass(frozen=True, slots=True)
class FailurePolicy:
    """Immutable semantic error and infrastructure retry policy."""

    mode: str
    infra_retries: int = 0
    isolation_work_budget: int = 0

    def __post_init__(self) -> None:
        """Validate mode-specific retry and isolation bounds."""

        if self.mode not in {"raise", "isolate"}:
            raise ValueError("failure mode must be 'raise' or 'isolate'")
        if (
            isinstance(self.infra_retries, bool)
            or not isinstance(self.infra_retries, int)
            or self.infra_retries < 0
        ):
            raise ValueError("infra_retries must be a non-negative integer")
        if (
            isinstance(self.isolation_work_budget, bool)
            or not isinstance(self.isolation_work_budget, int)
            or self.isolation_work_budget < 0
        ):
            raise ValueError(
                "isolation_work_budget must be a non-negative integer"
            )
        if self.mode == "raise" and self.isolation_work_budget:
            raise ValueError("raise policy cannot have an isolation budget")
        if self.mode == "isolate" and self.isolation_work_budget <= 0:
            raise ValueError("isolate policy needs a positive work budget")

    @classmethod
    def raise_(cls, *, infra_retries: int = 0) -> "FailurePolicy":
        """Create fail-fast data policy with optional infrastructure retries."""

        return cls("raise", infra_retries, 0)

    @classmethod
    def isolate(
        cls,
        *,
        infra_retries: int = 0,
        work_budget: int = 64,
    ) -> "FailurePolicy":
        """Create bounded bad-grain isolation policy."""

        return cls("isolate", infra_retries, work_budget)


class ParameterKind(Enum):
    """Stable project-owned encoding of supported ``inspect`` parameter kinds."""

    POSITIONAL_ONLY = "positional_only"
    POSITIONAL_OR_KEYWORD = "positional_or_keyword"
    KEYWORD_ONLY = "keyword_only"


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """One UDF data role after removing the instance receiver and batch list."""

    name: str
    index: int
    kind: ParameterKind
    item_shape: ValueShape

    def __post_init__(self) -> None:
        """Validate a stable role name and contiguous non-negative index."""

        if not self.name or not self.name.isidentifier():
            raise ValueError("parameter name must be a valid identifier")
        if self.index < 0:
            raise ValueError("parameter index must be non-negative")


@dataclass(frozen=True, slots=True)
class InputBinding:
    """Frozen connection from one node role to a producer port."""

    role: str
    port: PortId
    parameter: ParameterSpec

    def __post_init__(self) -> None:
        """Require one canonical role name across binding and schema."""

        if self.role != self.parameter.name:
            raise ValueError("input role must match ParameterSpec.name")


@dataclass(frozen=True, slots=True)
class CallSchema:
    """UDF parameter contract plus per-call positional/keyword reconstruction."""

    parameters: tuple[ParameterSpec, ...]
    positional_count: int
    keyword_roles: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate complete and non-overlapping argument reconstruction."""

        expected_indexes = tuple(range(len(self.parameters)))
        if (
            tuple(parameter.index for parameter in self.parameters)
            != expected_indexes
        ):
            raise ValueError("CallSchema parameter indexes must be contiguous")
        names = tuple(parameter.name for parameter in self.parameters)
        if len(names) != len(set(names)):
            raise ValueError("CallSchema parameter names must be unique")
        if self.positional_count < 0 or self.positional_count > len(names):
            raise ValueError("CallSchema positional_count is out of range")
        if len(self.keyword_roles) != len(set(self.keyword_roles)):
            raise ValueError("CallSchema keyword roles must be unique")
        if any(role not in names for role in self.keyword_roles):
            raise ValueError("CallSchema keyword role is not a parameter")
        if self.positional_count + len(self.keyword_roles) != len(names):
            raise ValueError("CallSchema must bind every UDF data parameter")


class ReturnKind(Enum):
    """Supported top-level UDF return structures."""

    SINGLE = "single"
    TUPLE = "tuple"
    NAMED_TUPLE = "named_tuple"


@dataclass(frozen=True, slots=True)
class ReturnLeafSpec:
    """One physical batch column and logical output leaf."""

    slot: int
    name: str | None
    item_shape: ValueShape

    def __post_init__(self) -> None:
        """Validate a non-negative slot and optional legal field name."""

        if self.slot < 0:
            raise ValueError("return leaf slot must be non-negative")
        if self.name is not None and (
            not self.name or not self.name.isidentifier()
        ):
            raise ValueError("return leaf name must be a valid identifier")


@dataclass(frozen=True, slots=True)
class ReturnSchema:
    """Frozen single/tuple/NamedTuple UDF return ABI."""

    kind: ReturnKind
    leaves: tuple[ReturnLeafSpec, ...]
    named_tuple_type: TypeRef | None = None

    def __post_init__(self) -> None:
        """Validate arity, slots, names, and top-level return kind."""

        if not self.leaves:
            raise ValueError("ReturnSchema must contain at least one leaf")
        if tuple(leaf.slot for leaf in self.leaves) != tuple(
            range(len(self.leaves))
        ):
            raise ValueError("return leaf slots must be contiguous")
        if self.kind is ReturnKind.SINGLE:
            if len(self.leaves) != 1 or self.leaves[0].name is not None:
                raise ValueError("single return requires one unnamed leaf")
            if self.named_tuple_type is not None:
                raise ValueError("single return cannot name a tuple type")
        elif self.kind is ReturnKind.TUPLE:
            if any(leaf.name is not None for leaf in self.leaves):
                raise ValueError("anonymous tuple leaves must be unnamed")
            if self.named_tuple_type is not None:
                raise ValueError("anonymous tuple cannot name a tuple type")
        else:
            names = tuple(leaf.name for leaf in self.leaves)
            if any(name is None for name in names):
                raise ValueError("NamedTuple leaves must all be named")
            if len(names) != len(set(names)):
                raise ValueError("NamedTuple leaf names must be unique")
            if self.named_tuple_type is None:
                raise ValueError("NamedTuple schema requires its canonical type")


@dataclass(frozen=True, slots=True)
class UdfRecipe:
    """Importable UDF class and immutable constructor/runtime semantics."""

    factory: SerializableCallableRef
    init_args: tuple[object, ...] = ()
    init_kwargs: tuple[tuple[str, object], ...] = ()
    exception_atomic: bool = False

    def __post_init__(self) -> None:
        """Validate constructor keywords and frozen exception semantics."""

        keys = tuple(key for key, _ in self.init_kwargs)
        if any(not isinstance(key, str) for key in keys):
            raise ValueError("constructor keyword names must be strings")
        if len(keys) != len(set(keys)):
            raise ValueError("constructor keyword names must be unique")
        if not isinstance(self.exception_atomic, bool):
            raise TypeError("exception_atomic must be a bool")


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    """Complete immutable execution policy for one MAP node."""

    batch: BatchPolicy
    resources: ResourceSpec
    failures: FailurePolicy


@dataclass(frozen=True, slots=True)
class PhysicalOutputSpec:
    """Worker return slot mapping for one logical output port."""

    port: PortId
    return_slot: int
    shape: ValueShape
    emit_control_bits: bool

    def __post_init__(self) -> None:
        """Validate return slot and bool-only control projection encoding."""

        if self.return_slot < 0:
            raise ValueError("physical return slot must be non-negative")
        if self.emit_control_bits and not isinstance(self.shape, BoolShape):
            raise ValueError("only BoolShape can emit control bits")


@dataclass(frozen=True, slots=True)
class SourceOp:
    """Internal single-source operation."""


@dataclass(frozen=True, slots=True)
class MapOp:
    """Only graph operation permitted to execute user UDF code."""

    udf: UdfRecipe
    call_schema: CallSchema
    return_schema: ReturnSchema
    execution: ExecutionSpec
    physical_outputs: tuple[PhysicalOutputSpec, ...]


@dataclass(frozen=True, slots=True)
class FilterOp:
    """System presence transform driven by a bool mask port."""

    mask: PortId
    target: PortId


@dataclass(frozen=True, slots=True)
class ExpandOp:
    """System transform exposing one structural list layer."""

    input: PortId
    scope: ScopeDefId


@dataclass(frozen=True, slots=True)
class ReduceOp:
    """System transform closing exactly one path-local top scope."""

    input: PortId
    closes_scope: ScopeDefId


NodeOp: TypeAlias = SourceOp | MapOp | FilterOp | ExpandOp | ReduceOp


@dataclass(frozen=True, slots=True)
class PortSpec:
    """Frozen logical output port with shape and occurrence metadata."""

    id: PortId
    name: str
    producer: NodeId
    shape: ValueShape
    scope_path: tuple[ScopeDefId, ...]
    occurrence_domain: OccurrenceDomainId

    def __post_init__(self) -> None:
        """Validate the internal diagnostic port name."""

        if not self.name:
            raise ValueError("port name must be non-empty")


@dataclass(frozen=True, slots=True)
class GraphOutputSpec:
    """Stable user-visible graph output name and root-scope port."""

    name: str
    port: PortId

    def __post_init__(self) -> None:
        """Validate the stable user-visible output name."""

        if not self.name or not self.name.isidentifier():
            raise ValueError("graph output name must be a valid identifier")


@dataclass(frozen=True, slots=True)
class NodeSpec:
    """Frozen graph node with a closed operation union."""

    id: NodeId
    name: str
    op: NodeOp
    inputs: tuple[InputBinding, ...]
    outputs: tuple[PortSpec, ...]

    def __post_init__(self) -> None:
        """Validate node name, unique roles, output arity, and ownership."""

        if not self.name:
            raise ValueError("node name must be non-empty")
        roles = tuple(binding.role for binding in self.inputs)
        if len(roles) != len(set(roles)):
            raise ValueError("node input roles must be unique")
        if not self.outputs:
            raise ValueError("node must contain at least one output")
        if any(port.producer != self.id for port in self.outputs):
            raise ValueError("node output producer must equal node id")


@dataclass(frozen=True, slots=True)
class ScopePlan:
    """Static EXPAND definition and all REDUCE consumers of that scope."""

    definition: ScopeDefId
    expand_node: NodeId
    reducers: tuple[NodeId, ...]

    def __post_init__(self) -> None:
        """Validate canonical unique topological reducer routing."""

        if len(self.reducers) != len(set(self.reducers)):
            raise ValueError("scope reducer list must be unique")
        if tuple(sorted(self.reducers, key=int)) != self.reducers:
            raise ValueError("scope reducers must be in topological order")


@dataclass(frozen=True, slots=True)
class CompiledGraph:
    """Self-verifying immutable graph consumed by future runtime backends."""

    fingerprint: GraphFingerprint
    source: PortId
    nodes: tuple[NodeSpec, ...]
    outputs: tuple[GraphOutputSpec, ...]
    producer_by_port: Mapping[PortId, NodeId]
    consumers_by_port: Mapping[PortId, tuple[NodeId, ...]]
    scope_plans: Mapping[ScopeDefId, ScopePlan]

    def node(self, node_id: NodeId | int) -> NodeSpec:
        """Return a node by typed ID or compact integer."""

        requested = node_id if isinstance(node_id, NodeId) else NodeId(node_id)
        for node in self.nodes:
            if node.id == requested:
                return node
        raise KeyError(f"unknown node {int(requested)}")

    def port(self, port_id: PortId | int) -> PortSpec:
        """Return a frozen port by typed ID or compact integer."""

        requested = port_id if isinstance(port_id, PortId) else PortId(port_id)
        for node in self.nodes:
            for port in node.outputs:
                if port.id == requested:
                    return port
        raise KeyError(f"unknown port {int(requested)}")

    def producer(self, port_id: PortId | int) -> NodeSpec:
        """Return the unique producer of a port."""

        port = port_id if isinstance(port_id, PortId) else PortId(port_id)
        try:
            return self.node(self.producer_by_port[port])
        except KeyError as error:
            raise KeyError(f"port {int(port)} has no producer") from error

    def consumers(self, port_id: PortId | int) -> tuple[NodeSpec, ...]:
        """Return consumers in frozen topological order."""

        port = port_id if isinstance(port_id, PortId) else PortId(port_id)
        return tuple(
            self.node(node) for node in self.consumers_by_port.get(port, ())
        )

    def verify(self) -> None:
        """Recompute frozen cross-record invariants."""

        verify_frozen_graph(self)


@dataclass(frozen=True, slots=True, order=True)
class TraceOwnerId:
    """Random owner token preventing SymbolicPort use across traces."""

    value: bytes
    WIDTH: ClassVar[int] = 16

    def __post_init__(self) -> None:
        """Validate the random trace token width."""

        if not isinstance(self.value, bytes) or len(self.value) != self.WIDTH:
            raise ValueError("TraceOwnerId must contain exactly 16 bytes")


@dataclass(frozen=True, slots=True, order=True)
class SymbolicScopeId:
    """Trace-local static scope handle prior to stable ID assignment."""

    value: int

    def __post_init__(self) -> None:
        """Validate a non-negative trace-local scope index."""

        if (
            isinstance(self.value, bool)
            or not isinstance(self.value, int)
            or self.value < 0
        ):
            raise ValueError("SymbolicScopeId must be a non-negative integer")


@dataclass(frozen=True, slots=True, order=True)
class SymbolicDomainId:
    """Trace-local occurrence domain prior to stable ID assignment."""

    value: int

    def __post_init__(self) -> None:
        """Validate a non-negative trace-local occurrence domain index."""

        if (
            isinstance(self.value, bool)
            or not isinstance(self.value, int)
            or self.value < 0
        ):
            raise ValueError("SymbolicDomainId must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class SymbolicPort:
    """Trace-owned authoring value; never interchangeable with PortSpec."""

    trace_owner: TraceOwnerId
    symbolic_key: int
    shape: ValueShape
    symbolic_scope_path: tuple[SymbolicScopeId, ...]
    occurrence_domain: SymbolicDomainId

    def __bool__(self) -> bool:
        """Reject accidental use as a Python condition."""

        raise CompileError(
            "SymbolicPortTruthValue",
            "SymbolicPort cannot be used as a Python truth value",
        )

    def __iter__(self):
        """Reject accidental iteration/unpacking of a single port."""

        raise CompileError(
            "SymbolicPortIteration",
            "SymbolicPort cannot be iterated; use expand() for structural lists",
        )


class _SymbolicOpKind(Enum):
    """Internal trace operation discriminator."""

    SOURCE = "source"
    MAP = "map"
    FILTER = "filter"
    EXPAND = "expand"
    REDUCE = "reduce"


@dataclass(frozen=True, slots=True)
class _UdfContract:
    """Strictly parsed UDF instance method contract."""

    signature: inspect.Signature
    parameters: tuple[ParameterSpec, ...]
    return_schema: ReturnSchema
    return_type: object


@dataclass(frozen=True, slots=True)
class _MapTraceData:
    """Map spec and call-specific schema retained during tracing."""

    map_spec: object
    call_schema: CallSchema
    return_schema: ReturnSchema


@dataclass(frozen=True, slots=True)
class _ExpandTraceData:
    """Trace-local scope/domain introduced by one EXPAND."""

    scope: SymbolicScopeId
    child_domain: SymbolicDomainId


@dataclass(frozen=True, slots=True)
class SymbolicNode:
    """Trace-time node retained until pruning and stable ID assignment."""

    symbolic_key: int
    kind: _SymbolicOpKind
    inputs: tuple[SymbolicPort, ...]
    outputs: tuple[SymbolicPort, ...]
    data: object | None = None


@dataclass(slots=True)
class SymbolicGraph:
    """Trace-local append-only graph and domain parent relation."""

    owner: TraceOwnerId
    nodes: list[SymbolicNode] = field(default_factory=list)
    domain_parent: dict[SymbolicDomainId, SymbolicDomainId] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class TraceContext:
    """Active owner-isolated symbolic graph construction context."""

    graph: SymbolicGraph
    _next_node: int = 0
    _next_port: int = 0
    _next_scope: int = 0
    _next_domain: int = 1
    _contracts: dict[type, _UdfContract] = field(default_factory=dict)
    root_domain: SymbolicDomainId = SymbolicDomainId(0)

    @classmethod
    def create(cls) -> "TraceContext":
        """Create an isolated trace with a random non-semantic owner token."""

        import secrets

        owner = TraceOwnerId(secrets.token_bytes(TraceOwnerId.WIDTH))
        return cls(SymbolicGraph(owner))

    @property
    def owner(self) -> TraceOwnerId:
        """Return the current trace owner token."""

        return self.graph.owner

    def source(self) -> SymbolicPort:
        """Append the sole source node with an opaque Any item shape."""

        output = self._new_port(
            OpaqueShape(None),
            (),
            self.root_domain,
        )
        self._append(_SymbolicOpKind.SOURCE, (), (output,))
        return output

    def map_call(
        self,
        map_spec: object,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        """Bind a Map invocation and mirror its annotated return structure."""

        op_cls = getattr(map_spec, "op_cls", None)
        if not isinstance(op_cls, type):
            raise CompileError(
                "MissingUdfRun",
                "Map requires a UDF class, not an instance",
            )
        contract = self._contracts.get(op_cls)
        if contract is None:
            contract = _parse_udf_contract(op_cls)
            self._contracts[op_cls] = contract
        _validate_constructor_binding(
            op_cls,
            tuple(getattr(map_spec, "init_args", ())),
            tuple(getattr(map_spec, "init_kwargs", ())),
        )
        try:
            bound = contract.signature.bind(*args, **kwargs)
        except TypeError as error:
            raise CompileError("InvalidMapCall", str(error)) from error
        parameter_names = tuple(
            parameter.name for parameter in contract.parameters
        )
        missing = tuple(
            name for name in parameter_names if name not in bound.arguments
        )
        if missing:
            raise CompileError(
                "InvalidMapCall",
                "all run parameters must bind SymbolicPorts; missing "
                + ", ".join(missing),
            )
        inputs: list[SymbolicPort] = []
        for parameter in contract.parameters:
            value = bound.arguments[parameter.name]
            port = self.require_port(value)
            if not shapes_compatible(port.shape, parameter.item_shape):
                raise CompileError(
                    "InvalidMapCall",
                    f"port shape for role {parameter.name!r} is incompatible "
                    "with run annotation",
                )
            inputs.append(port)
        first = inputs[0]
        for port in inputs[1:]:
            if port.symbolic_scope_path != first.symbolic_scope_path:
                raise CompileError(
                    "ScopePathMismatch",
                    "all MAP inputs must have the same path-local scope",
                )
            if port.occurrence_domain != first.occurrence_domain:
                raise CompileError(
                    "OccurrenceDomainMismatch",
                    "all MAP inputs must have the same occurrence domain",
                )
        keyword_roles = tuple(
            parameter.name
            for parameter in contract.parameters
            if parameter.name in kwargs
        )
        call_schema = CallSchema(
            contract.parameters,
            len(args),
            keyword_roles,
        )
        outputs = tuple(
            self._new_port(
                leaf.item_shape,
                first.symbolic_scope_path,
                first.occurrence_domain,
            )
            for leaf in contract.return_schema.leaves
        )
        self._append(
            _SymbolicOpKind.MAP,
            tuple(inputs),
            outputs,
            _MapTraceData(map_spec, call_schema, contract.return_schema),
        )
        return _mirror_symbolic_return(
            contract.return_schema,
            contract.return_type,
            outputs,
        )

    def filter(
        self,
        mask_value: object,
        target_value: object,
    ) -> SymbolicPort:
        """Append a system FILTER after shape/domain validation."""

        mask = self.require_port(mask_value)
        target = self.require_port(target_value)
        if not isinstance(mask.shape, BoolShape):
            raise CompileError(
                "FilterMaskNotBool",
                "filter mask must be a list[bool] MAP output",
            )
        producer = next(
            (
                node
                for node in reversed(self.graph.nodes)
                if any(
                    output.symbolic_key == mask.symbolic_key
                    for output in node.outputs
                )
            ),
            None,
        )
        if producer is None or producer.kind is not _SymbolicOpKind.MAP:
            raise CompileError(
                "FilterMaskNotMapOutput",
                "filter mask must be produced directly by a MAP node",
            )
        self._require_aligned(mask, target, "FILTER")
        output = self._new_port(
            target.shape,
            target.symbolic_scope_path,
            target.occurrence_domain,
        )
        self._append(
            _SymbolicOpKind.FILTER,
            (mask, target),
            (output,),
        )
        return output

    def expand(self, group_value: object) -> SymbolicPort:
        """Append a one-layer system EXPAND and a path-local child domain."""

        group = self.require_port(group_value)
        if not isinstance(group.shape, StructuralListShape):
            raise CompileError(
                "ExpandRequiresStructuralList",
                "expand() requires an exposed StructuralListShape",
            )
        scope = SymbolicScopeId(self._next_scope)
        self._next_scope += 1
        child_domain = SymbolicDomainId(self._next_domain)
        self._next_domain += 1
        self.graph.domain_parent[child_domain] = group.occurrence_domain
        output = self._new_port(
            group.shape.element,
            group.symbolic_scope_path + (scope,),
            child_domain,
        )
        self._append(
            _SymbolicOpKind.EXPAND,
            (group,),
            (output,),
            _ExpandTraceData(scope, child_domain),
        )
        return output

    def reduce(self, items_value: object) -> SymbolicPort:
        """Append a system REDUCE that pops exactly the path-local top scope."""

        items = self.require_port(items_value)
        if not items.symbolic_scope_path:
            raise CompileError(
                "ReduceAtRootScope",
                "reduce() requires a port inside a dynamic scope",
            )
        try:
            parent_domain = self.graph.domain_parent[items.occurrence_domain]
        except KeyError as error:
            raise CompileError(
                "NonLifoReduce",
                "port occurrence domain has no matching top EXPAND",
            ) from error
        output = self._new_port(
            StructuralListShape(items.shape),
            items.symbolic_scope_path[:-1],
            parent_domain,
        )
        self._append(
            _SymbolicOpKind.REDUCE,
            (items,),
            (output,),
        )
        return output

    def require_port(self, value: object) -> SymbolicPort:
        """Require a SymbolicPort owned by this exact active trace."""

        if not isinstance(value, SymbolicPort):
            raise CompileError(
                "InvalidMapCall",
                f"expected SymbolicPort, got {type(value).__name__}",
            )
        if value.trace_owner != self.owner:
            raise CompileError(
                "ForeignPort",
                "SymbolicPort belongs to a different trace owner",
            )
        return value

    def _require_aligned(
        self,
        left: SymbolicPort,
        right: SymbolicPort,
        operation: str,
    ) -> None:
        """Validate exact path and occurrence alignment."""

        if left.symbolic_scope_path != right.symbolic_scope_path:
            raise CompileError(
                "ScopePathMismatch",
                f"{operation} inputs must have the same scope path",
            )
        if left.occurrence_domain != right.occurrence_domain:
            raise CompileError(
                "OccurrenceDomainMismatch",
                f"{operation} inputs must have the same occurrence domain",
            )

    def _new_port(
        self,
        shape: ValueShape,
        scope_path: tuple[SymbolicScopeId, ...],
        domain: SymbolicDomainId,
    ) -> SymbolicPort:
        """Allocate a trace-local port key."""

        port = SymbolicPort(
            self.owner,
            self._next_port,
            shape,
            scope_path,
            domain,
        )
        self._next_port += 1
        return port

    def _append(
        self,
        kind: _SymbolicOpKind,
        inputs: tuple[SymbolicPort, ...],
        outputs: tuple[SymbolicPort, ...],
        data: object | None = None,
    ) -> SymbolicNode:
        """Append a topologically ordered symbolic node."""

        node = SymbolicNode(
            self._next_node,
            kind,
            inputs,
            outputs,
            data,
        )
        self._next_node += 1
        self.graph.nodes.append(node)
        return node


_ACTIVE_TRACE: contextvars.ContextVar[TraceContext | None] = (
    contextvars.ContextVar("rayorch_multigrain_v3_trace", default=None)
)


def _active_trace() -> TraceContext:
    """Return the active trace or reject out-of-trace authoring calls."""

    context = _ACTIVE_TRACE.get()
    if context is None:
        raise CompileError(
            "PortUsedOutsideTrace",
            "Map/filter/expand/reduce are only valid in Pipeline.forward",
        )
    return context


def trace_map_call(
    map_spec: object,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    """Narrow API hook used by :class:`api.Map` during tracing."""

    return _active_trace().map_call(map_spec, args, kwargs)


def trace_filter(mask: object, target: object) -> SymbolicPort:
    """Narrow API hook for the system FILTER authoring function."""

    return _active_trace().filter(mask, target)


def trace_expand(group: object) -> SymbolicPort:
    """Narrow API hook for the system EXPAND authoring function."""

    return _active_trace().expand(group)


def trace_reduce(items: object) -> SymbolicPort:
    """Narrow API hook for the system REDUCE authoring function."""

    return _active_trace().reduce(items)


def validate_constructor_binding(
    op_cls: type,
    init_args: tuple[object, ...],
    init_kwargs: tuple[tuple[str, object], ...],
) -> None:
    """Public validation hook used by immutable ``Map.pre_init``."""

    _validate_constructor_binding(op_cls, init_args, init_kwargs)


def shapes_compatible(actual: ValueShape, expected: ValueShape) -> bool:
    """Return exact recursive shape compatibility with opaque Any wildcard."""

    if isinstance(actual, OpaqueShape) and isinstance(expected, OpaqueShape):
        return (
            actual.type_ref is None
            or expected.type_ref is None
            or actual.type_ref == expected.type_ref
        )
    if isinstance(actual, BoolShape) and isinstance(expected, BoolShape):
        return True
    if isinstance(actual, StructuralListShape) and isinstance(
        expected,
        StructuralListShape,
    ):
        return shapes_compatible(actual.element, expected.element)
    return False


def infer_item_shape(annotation: object) -> ValueShape:
    """Infer a per-grain shape from the canonical item annotation grammar."""

    return _parse_item_shape(annotation)


def infer_run_schemas(op_cls: type) -> tuple[CallSchema, ReturnSchema]:
    """Strictly infer declaration-level UDF call and return schemas.

    The returned CallSchema uses the canonical all-positional reconstruction
    only as an introspection view.  Actual compiled MAP nodes freeze the user's
    exact positional/keyword call form.
    """

    contract = _parse_udf_contract(op_cls)
    positional = sum(
        parameter.kind is not ParameterKind.KEYWORD_ONLY
        for parameter in contract.parameters
    )
    keyword_roles = tuple(
        parameter.name
        for parameter in contract.parameters
        if parameter.kind is ParameterKind.KEYWORD_ONLY
    )
    return (
        CallSchema(contract.parameters, positional, keyword_roles),
        contract.return_schema,
    )


def _parse_udf_contract(op_cls: type) -> _UdfContract:
    """Parse one importable UDF class without permissive fallbacks."""

    _callable_ref(op_cls)
    try:
        raw_run = inspect.getattr_static(op_cls, "run")
    except AttributeError as error:
        raise CompileError(
            "MissingUdfRun",
            f"{op_cls.__qualname__} does not define an instance run method",
        ) from error
    if isinstance(raw_run, (staticmethod, classmethod)):
        raise CompileError(
            "UnsupportedRunDescriptor",
            "run must be an instance method, not staticmethod/classmethod",
        )
    if not inspect.isfunction(raw_run):
        raise CompileError(
            "UnsupportedRunDescriptor",
            "run must be a supported Python instance method",
        )
    run_fn = inspect.unwrap(raw_run)
    try:
        raw_signature = inspect.signature(run_fn)
    except (TypeError, ValueError) as error:
        raise CompileError(
            "UnsupportedRunDescriptor",
            "run signature cannot be inspected",
        ) from error
    raw_parameters = tuple(raw_signature.parameters.values())
    if not raw_parameters:
        raise CompileError(
            "UnsupportedRunDescriptor",
            "run must declare one instance receiver",
        )
    receiver = raw_parameters[0]
    if receiver.kind not in {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }:
        raise CompileError(
            "UnsupportedRunDescriptor",
            "run receiver must be positional",
        )
    data_parameters = raw_parameters[1:]
    if not data_parameters:
        raise CompileError(
            "UnsupportedRunParameter",
            "run must consume at least one batch column",
        )
    if any(
        parameter.kind
        in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }
        for parameter in data_parameters
    ):
        raise CompileError(
            "UnsupportedRunParameter",
            "run cannot declare *args or **kwargs data parameters",
        )
    for parameter in data_parameters:
        if parameter.annotation is inspect.Parameter.empty:
            raise CompileError(
                "MissingRunParameterAnnotation",
                f"run parameter {parameter.name!r} has no annotation",
            )
    if raw_signature.return_annotation is inspect.Signature.empty:
        raise CompileError(
            "MissingRunReturnAnnotation",
            "run has no return annotation",
        )
    module = sys.modules.get(op_cls.__module__)
    globalns = vars(module) if module is not None else {}
    localns = dict(vars(op_cls))
    try:
        hints = get_type_hints(
            run_fn,
            globalns=globalns,
            localns=localns,
            include_extras=True,
        )
    except Exception as error:
        raise CompileError(
            "UnresolvableTypeHints",
            f"cannot resolve {op_cls.__qualname__}.run annotations: {error}",
        ) from error
    parameters: list[ParameterSpec] = []
    for index, parameter in enumerate(data_parameters):
        if parameter.name not in hints:
            raise CompileError(
                "MissingRunParameterAnnotation",
                f"run parameter {parameter.name!r} has no resolved annotation",
            )
        item_annotation = _batch_item_annotation(
            hints[parameter.name],
            f"run parameter {parameter.name!r}",
        )
        parameters.append(
            ParameterSpec(
                parameter.name,
                index,
                _parameter_kind(parameter.kind),
                _parse_item_shape(item_annotation),
            )
        )
    if "return" not in hints:
        raise CompileError(
            "MissingRunReturnAnnotation",
            "run has no resolved return annotation",
        )
    return_schema, return_type = _parse_return_schema(hints["return"])
    signature = raw_signature.replace(parameters=data_parameters)
    return _UdfContract(
        signature,
        tuple(parameters),
        return_schema,
        return_type,
    )


def _parameter_kind(kind: object) -> ParameterKind:
    """Convert CPython inspect kind into the stable project enum."""

    mapping = {
        inspect.Parameter.POSITIONAL_ONLY: ParameterKind.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD:
            ParameterKind.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY: ParameterKind.KEYWORD_ONLY,
    }
    try:
        return mapping[kind]
    except KeyError as error:
        raise CompileError(
            "UnsupportedRunParameter",
            f"unsupported run parameter kind: {kind}",
        ) from error


def _batch_item_annotation(annotation: object, location: str) -> object:
    """Remove the mandatory physical batch ``list`` from one column."""

    if get_origin(annotation) is not list:
        raise CompileError(
            "UnsupportedBatchColumnAnnotation",
            f"{location} must be annotated as list[T]",
        )
    arguments = get_args(annotation)
    if len(arguments) != 1:
        raise CompileError(
            "UnsupportedBatchColumnAnnotation",
            f"{location} must contain exactly one list item type",
        )
    return arguments[0]


def _parse_return_schema(
    annotation: object,
) -> tuple[ReturnSchema, object]:
    """Parse only single batch column, fixed tuple, or NamedTuple returns."""

    if _is_named_tuple_type(annotation):
        names = tuple(getattr(annotation, "_fields"))
        try:
            field_hints = get_type_hints(annotation, include_extras=True)
        except Exception as error:
            raise CompileError(
                "UnresolvableTypeHints",
                f"cannot resolve NamedTuple return fields: {error}",
            ) from error
        if tuple(field_hints) != names:
            raise CompileError(
                "UnsupportedReturnStructure",
                "NamedTuple fields must all have resolvable annotations",
            )
        leaves = tuple(
            ReturnLeafSpec(
                slot,
                name,
                _parse_item_shape(
                    _batch_item_annotation(
                        field_hints[name],
                        f"NamedTuple field {name!r}",
                    )
                ),
            )
            for slot, name in enumerate(names)
        )
        return (
            ReturnSchema(
                ReturnKind.NAMED_TUPLE,
                leaves,
                _canonical_type_ref(annotation),
            ),
            annotation,
        )
    origin = get_origin(annotation)
    if origin is tuple:
        parts = get_args(annotation)
        if not parts or Ellipsis in parts:
            raise CompileError(
                "UnsupportedReturnStructure",
                "run tuple return must have fixed non-zero arity",
            )
        leaves = tuple(
            ReturnLeafSpec(
                slot,
                None,
                _parse_item_shape(
                    _batch_item_annotation(
                        part,
                        f"tuple return slot {slot}",
                    )
                ),
            )
            for slot, part in enumerate(parts)
        )
        return ReturnSchema(ReturnKind.TUPLE, leaves), tuple
    if origin is not list:
        raise CompileError(
            "UnsupportedReturnStructure",
            "run return must be list[T], fixed tuple, or NamedTuple",
        )
    item = _batch_item_annotation(annotation, "run return")
    leaf = ReturnLeafSpec(0, None, _parse_item_shape(item))
    return ReturnSchema(ReturnKind.SINGLE, (leaf,)), annotation


def _parse_item_shape(annotation: object) -> ValueShape:
    """Interpret only the outermost unmarked item list as structural."""

    origin = get_origin(annotation)
    if origin is Annotated:
        parts = get_args(annotation)
        base, metadata = parts[0], parts[1:]
        if metadata != (OpaqueValue,):
            raise CompileError(
                "UnsupportedItemAnnotation",
                "only Annotated[list[T], OpaqueValue] is supported",
            )
        if get_origin(base) is not list:
            raise CompileError(
                "UnsupportedItemAnnotation",
                "OpaqueValue may only annotate list[T]",
            )
        return OpaqueShape(_canonical_type_ref(base))
    if annotation is Any:
        return OpaqueShape(None)
    if annotation is bool:
        return BoolShape()
    if origin is list:
        arguments = get_args(annotation)
        if len(arguments) != 1:
            raise CompileError(
                "UnsupportedItemAnnotation",
                "structural list must contain one item type",
            )
        element = (
            OpaqueShape(None)
            if arguments[0] is Any
            else OpaqueShape(_canonical_type_ref(arguments[0]))
        )
        return StructuralListShape(
            element
        )
    return OpaqueShape(_canonical_type_ref(annotation))


def _canonical_type_ref(annotation: object) -> TypeRef:
    """Encode the finite canonical annotation grammar as a TypeRef tree."""

    if annotation is Any:
        return TypeRef("typing", "Any")
    if annotation is Ellipsis:
        return TypeRef("builtins", "ellipsis")
    if isinstance(annotation, TypeVar):
        raise CompileError(
            "UnsupportedItemAnnotation",
            "TypeVar is not a canonical payload annotation",
        )
    origin = get_origin(annotation)
    if origin is Annotated:
        raise CompileError(
            "UnsupportedItemAnnotation",
            "Annotated is only valid as Annotated[list[T], OpaqueValue]",
        )
    if origin in {typing.Union, types.UnionType}:
        raise CompileError(
            "UnsupportedItemAnnotation",
            "Union payload annotations are not supported",
        )
    if origin in {typing.Literal, Callable, typing.Callable}:
        raise CompileError(
            "UnsupportedItemAnnotation",
            "Literal/Callable payload annotations are not supported",
        )
    if origin is not None:
        if origin not in {list, dict, tuple, set, frozenset}:
            raise CompileError(
                "UnsupportedItemAnnotation",
                "only parameterized builtins are canonical opaque generics",
            )
        return TypeRef(
            origin.__module__,
            origin.__qualname__,
            tuple(_canonical_type_ref(argument) for argument in get_args(annotation)),
        )
    if not isinstance(annotation, type):
        raise CompileError(
            "UnsupportedItemAnnotation",
            f"{annotation!s} is not a canonical concrete class",
        )
    if getattr(annotation, "_is_protocol", False):
        raise CompileError(
            "UnsupportedItemAnnotation",
            "Protocol payload annotations are not supported",
        )
    if "<locals>" in annotation.__qualname__:
        raise CompileError(
            "NonCanonicalTypeRef",
            "local classes cannot participate in graph fingerprints",
        )
    module = sys.modules.get(annotation.__module__)
    if module is None:
        raise CompileError(
            "NonCanonicalTypeRef",
            f"type module {annotation.__module__!r} is not loaded",
        )
    resolved: object = module
    try:
        for component in annotation.__qualname__.split("."):
            resolved = getattr(resolved, component)
    except AttributeError as error:
        raise CompileError(
            "NonCanonicalTypeRef",
            f"{annotation.__module__}.{annotation.__qualname__} is not importable",
        ) from error
    if resolved is not annotation:
        raise CompileError(
            "NonCanonicalTypeRef",
            "type module/qualname resolves to a different object",
        )
    return TypeRef(annotation.__module__, annotation.__qualname__)


def _is_named_tuple_type(annotation: object) -> bool:
    """Return whether an object is a concrete typing.NamedTuple class."""

    return bool(
        isinstance(annotation, type)
        and issubclass(annotation, tuple)
        and isinstance(getattr(annotation, "_fields", None), tuple)
        and hasattr(annotation, "__annotations__")
    )


def _mirror_symbolic_return(
    schema: ReturnSchema,
    return_type: object,
    outputs: tuple[SymbolicPort, ...],
) -> object:
    """Replace annotated return leaves with trace-owned SymbolicPorts."""

    if schema.kind is ReturnKind.SINGLE:
        return outputs[0]
    if schema.kind is ReturnKind.TUPLE:
        return outputs
    assert isinstance(return_type, type)
    return return_type(*outputs)


def _callable_ref(op_cls: type) -> SerializableCallableRef:
    """Build and verify a stable importable UDF class reference."""

    if not isinstance(op_cls, type):
        raise CompileError("MissingUdfRun", "Map UDF must be a class")
    if op_cls.__module__ in {"__main__", "__mp_main__"}:
        raise CompileError(
            "NonCanonicalTypeRef",
            "UDF classes from process entry modules are not worker-importable",
        )
    if "<locals>" in op_cls.__qualname__:
        raise CompileError(
            "NonCanonicalTypeRef",
            "UDF classes defined in local scopes are not importable",
        )
    module = sys.modules.get(op_cls.__module__)
    if module is None:
        raise CompileError(
            "NonCanonicalTypeRef",
            f"UDF module {op_cls.__module__!r} is not loaded",
        )
    resolved: object = module
    try:
        for component in op_cls.__qualname__.split("."):
            resolved = getattr(resolved, component)
    except AttributeError as error:
        raise CompileError(
            "NonCanonicalTypeRef",
            "UDF class cannot be resolved through module/qualname",
        ) from error
    if resolved is not op_cls:
        raise CompileError(
            "NonCanonicalTypeRef",
            "UDF module/qualname resolves to a different class",
        )
    return SerializableCallableRef(op_cls.__module__, op_cls.__qualname__)


def _validate_constructor_binding(
    op_cls: type,
    init_args: tuple[object, ...],
    init_kwargs: tuple[tuple[str, object], ...],
) -> None:
    """Fail authoring early when the frozen constructor call is invalid."""

    try:
        signature = inspect.signature(op_cls)
        signature.bind(*init_args, **dict(init_kwargs))
    except (TypeError, ValueError) as error:
        raise CompileError(
            "InvalidConstructorBinding",
            f"invalid {op_cls.__qualname__} constructor recipe: {error}",
        ) from error


class GraphCompiler:
    """Compile a Pipeline.forward trace into a verified immutable graph."""

    def compile(self, pipeline: object) -> CompiledGraph:
        """Trace, prune, freeze, fingerprint, and independently verify a graph."""

        forward = getattr(pipeline, "forward", None)
        if forward is None or not callable(forward):
            raise CompileError(
                "InvalidForwardSignature",
                "Pipeline must define a callable forward method",
            )
        self._validate_forward_signature(forward)
        context = TraceContext.create()
        source = context.source()
        token = _ACTIVE_TRACE.set(context)
        try:
            result = forward(source)
        finally:
            _ACTIVE_TRACE.reset(token)
        outputs = self._normalize_outputs(context, result)
        return self._freeze(context, outputs)

    def _validate_forward_signature(self, forward: object) -> None:
        """Validate exactly one required positional source parameter."""

        try:
            signature = inspect.signature(forward)
        except (TypeError, ValueError) as error:
            raise CompileError(
                "InvalidForwardSignature",
                "Pipeline.forward signature cannot be inspected",
            ) from error
        parameters = tuple(signature.parameters.values())
        if len(parameters) != 1:
            code = (
                "MultipleSourcesNotSupported"
                if len(parameters) > 1
                else "InvalidForwardSignature"
            )
            raise CompileError(
                code,
                "Pipeline.forward must have exactly one source parameter",
            )
        parameter = parameters[0]
        if parameter.kind not in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }:
            raise CompileError(
                "InvalidForwardSignature",
                "source must be positional-only or positional-or-keyword",
            )
        if parameter.default is not inspect.Parameter.empty:
            raise CompileError(
                "InvalidForwardSignature",
                "source parameter cannot have a default",
            )

    def _normalize_outputs(
        self,
        context: TraceContext,
        result: object,
    ) -> tuple[tuple[str, SymbolicPort], ...]:
        """Freeze the ordinary non-empty dict returned by forward."""

        if type(result) is not dict or not result:
            raise CompileError(
                "InvalidPipelineReturn",
                "Pipeline.forward must return a non-empty ordinary dict",
            )
        outputs: list[tuple[str, SymbolicPort]] = []
        for name, value in result.items():
            if (
                not isinstance(name, str)
                or not name
                or not name.isidentifier()
            ):
                raise CompileError(
                    "InvalidPipelineReturn",
                    "graph output names must be valid identifiers",
                )
            port = context.require_port(value)
            if port.symbolic_scope_path:
                raise CompileError(
                    "UnclosedOutputScope",
                    f"graph output {name!r} is not in root scope",
                )
            outputs.append((name, port))
        return tuple(outputs)

    def _freeze(
        self,
        context: TraceContext,
        requested_outputs: tuple[tuple[str, SymbolicPort], ...],
    ) -> CompiledGraph:
        """Prune unreachable trace records and assign all stable IDs."""

        producer_by_symbol: dict[int, SymbolicNode] = {}
        for node in context.graph.nodes:
            for port in node.outputs:
                if port.symbolic_key in producer_by_symbol:
                    raise CompileError(
                        "DanglingPort",
                        "symbolic port has multiple producers",
                    )
                producer_by_symbol[port.symbolic_key] = node
        required_node_keys: set[int] = set()
        stack = [port for _, port in requested_outputs]
        while stack:
            port = stack.pop()
            try:
                producer = producer_by_symbol[port.symbolic_key]
            except KeyError as error:
                raise CompileError(
                    "DanglingPort",
                    "symbolic output has no producer",
                ) from error
            if producer.symbolic_key in required_node_keys:
                continue
            required_node_keys.add(producer.symbolic_key)
            stack.extend(producer.inputs)
        symbolic_nodes = tuple(
            node
            for node in context.graph.nodes
            if node.symbolic_key in required_node_keys
        )
        if not any(node.kind is _SymbolicOpKind.MAP for node in symbolic_nodes):
            raise CompileError(
                "NoMapOutputs",
                "compiled graph must contain a reachable MAP output",
            )
        node_ids = {
            node.symbolic_key: NodeId(index)
            for index, node in enumerate(symbolic_nodes)
        }
        port_ids: dict[int, PortId] = {}
        next_port = 0
        for node in symbolic_nodes:
            for port in node.outputs:
                port_ids[port.symbolic_key] = PortId(next_port)
                next_port += 1
        scope_ids: dict[SymbolicScopeId, ScopeDefId] = {}
        domain_ids: dict[SymbolicDomainId, OccurrenceDomainId] = {
            context.root_domain: OccurrenceDomainId(0)
        }
        next_scope = 0
        next_domain = 1
        for node in symbolic_nodes:
            if node.kind is _SymbolicOpKind.EXPAND:
                assert isinstance(node.data, _ExpandTraceData)
                scope_ids[node.data.scope] = ScopeDefId(next_scope)
                domain_ids[node.data.child_domain] = OccurrenceDomainId(
                    next_domain
                )
                next_scope += 1
                next_domain += 1
        for node in symbolic_nodes:
            for port in (*node.inputs, *node.outputs):
                if any(scope not in scope_ids for scope in port.symbolic_scope_path):
                    raise CompileError(
                        "DanglingPort",
                        "port references a pruned/dangling scope",
                    )
                if port.occurrence_domain not in domain_ids:
                    raise CompileError(
                        "OccurrenceDomainMismatch",
                        "port references a pruned/dangling occurrence domain",
                    )
        filter_masks = {
            node.inputs[0].symbolic_key
            for node in symbolic_nodes
            if node.kind is _SymbolicOpKind.FILTER
        }
        frozen_nodes: list[NodeSpec] = []
        ports_by_symbol: dict[int, PortSpec] = {}
        for symbolic in symbolic_nodes:
            node_id = node_ids[symbolic.symbolic_key]
            frozen_outputs = tuple(
                PortSpec(
                    id=port_ids[port.symbolic_key],
                    name=self._port_name(symbolic, slot),
                    producer=node_id,
                    shape=port.shape,
                    scope_path=tuple(
                        scope_ids[scope]
                        for scope in port.symbolic_scope_path
                    ),
                    occurrence_domain=domain_ids[port.occurrence_domain],
                )
                for slot, port in enumerate(symbolic.outputs)
            )
            for symbolic_port, frozen_port in zip(
                symbolic.outputs,
                frozen_outputs,
            ):
                ports_by_symbol[symbolic_port.symbolic_key] = frozen_port
            bindings = self._freeze_bindings(symbolic, port_ids)
            op = self._freeze_op(
                symbolic,
                frozen_outputs,
                port_ids,
                scope_ids,
                filter_masks,
            )
            frozen_nodes.append(
                NodeSpec(
                    node_id,
                    self._node_name(symbolic, node_id),
                    op,
                    bindings,
                    frozen_outputs,
                )
            )
        producer_index: dict[PortId, NodeId] = {}
        consumer_index: dict[PortId, list[NodeId]] = {
            port.id: []
            for node in frozen_nodes
            for port in node.outputs
        }
        for node in frozen_nodes:
            for port in node.outputs:
                producer_index[port.id] = node.id
            for binding in node.inputs:
                try:
                    consumer_index[binding.port].append(node.id)
                except KeyError as error:
                    raise CompileError(
                        "DanglingPort",
                        "frozen input references an unknown port",
                    ) from error
        scope_plans: dict[ScopeDefId, ScopePlan] = {}
        for node in frozen_nodes:
            if isinstance(node.op, ExpandOp):
                reducers = tuple(
                    candidate.id
                    for candidate in frozen_nodes
                    if isinstance(candidate.op, ReduceOp)
                    and candidate.op.closes_scope == node.op.scope
                )
                if not reducers:
                    raise CompileError(
                        "UnclosedOutputScope",
                        "reachable EXPAND scope has no REDUCE",
                    )
                scope_plans[node.op.scope] = ScopePlan(
                    node.op.scope,
                    node.id,
                    reducers,
                )
        graph_outputs = tuple(
            GraphOutputSpec(name, port_ids[port.symbolic_key])
            for name, port in requested_outputs
        )
        source_nodes = tuple(
            node for node in frozen_nodes if isinstance(node.op, SourceOp)
        )
        if len(source_nodes) != 1:
            raise CompileError(
                "InvalidForwardSignature",
                "compiled graph must contain exactly one source node",
            )
        source_port = source_nodes[0].outputs[0].id
        producer_proxy = _FrozenMapping.from_mapping(producer_index)
        consumer_proxy = _FrozenMapping.from_mapping(
            {
                port: tuple(consumers)
                for port, consumers in consumer_index.items()
            }
        )
        scope_proxy = _FrozenMapping.from_mapping(scope_plans)
        provisional = CompiledGraph(
            GraphFingerprint(bytes(32)),
            source_port,
            tuple(frozen_nodes),
            graph_outputs,
            producer_proxy,
            consumer_proxy,
            scope_proxy,
        )
        try:
            fingerprint = GraphFingerprint.derive(
                "compiled-graph",
                provisional.source,
                provisional.nodes,
                provisional.outputs,
                tuple(provisional.scope_plans.items()),
            )
        except (TypeError, ValueError) as error:
            raise CompileError(
                "UnserializableUdfRecipe",
                f"graph contains a non-canonical recipe/config value: {error}",
            ) from error
        graph = replace(provisional, fingerprint=fingerprint)
        verify_frozen_graph(graph)
        return graph

    def _freeze_bindings(
        self,
        node: SymbolicNode,
        port_ids: Mapping[int, PortId],
    ) -> tuple[InputBinding, ...]:
        """Build node bindings in compiled role order."""

        if node.kind is _SymbolicOpKind.SOURCE:
            return ()
        if node.kind is _SymbolicOpKind.MAP:
            assert isinstance(node.data, _MapTraceData)
            parameters = node.data.call_schema.parameters
        else:
            names = {
                _SymbolicOpKind.FILTER: ("mask", "target"),
                _SymbolicOpKind.EXPAND: ("input",),
                _SymbolicOpKind.REDUCE: ("input",),
            }[node.kind]
            parameters = tuple(
                ParameterSpec(
                    name,
                    index,
                    ParameterKind.POSITIONAL_ONLY,
                    node.inputs[index].shape,
                )
                for index, name in enumerate(names)
            )
        return tuple(
            InputBinding(
                parameter.name,
                port_ids[port.symbolic_key],
                parameter,
            )
            for parameter, port in zip(parameters, node.inputs)
        )

    def _freeze_op(
        self,
        node: SymbolicNode,
        outputs: tuple[PortSpec, ...],
        port_ids: Mapping[int, PortId],
        scope_ids: Mapping[SymbolicScopeId, ScopeDefId],
        filter_masks: set[int],
    ) -> NodeOp:
        """Freeze the closed operation union for one symbolic node."""

        if node.kind is _SymbolicOpKind.SOURCE:
            return SourceOp()
        if node.kind is _SymbolicOpKind.FILTER:
            return FilterOp(
                port_ids[node.inputs[0].symbolic_key],
                port_ids[node.inputs[1].symbolic_key],
            )
        if node.kind is _SymbolicOpKind.EXPAND:
            assert isinstance(node.data, _ExpandTraceData)
            return ExpandOp(
                port_ids[node.inputs[0].symbolic_key],
                scope_ids[node.data.scope],
            )
        if node.kind is _SymbolicOpKind.REDUCE:
            closed = node.inputs[0].symbolic_scope_path[-1]
            return ReduceOp(
                port_ids[node.inputs[0].symbolic_key],
                scope_ids[closed],
            )
        assert isinstance(node.data, _MapTraceData)
        map_spec = node.data.map_spec
        op_cls = getattr(map_spec, "op_cls")
        try:
            init_args, init_kwargs = freeze_constructor_arguments(
                tuple(getattr(map_spec, "init_args", ())),
                tuple(getattr(map_spec, "init_kwargs", ())),
            )
            resources = freeze_resource_spec(
                getattr(map_spec, "resources")
            )
        except (TypeError, ValueError) as error:
            raise CompileError(
                "UnserializableUdfRecipe",
                f"cannot snapshot MAP constructor/resources: {error}",
            ) from error
        exception_atomic = getattr(op_cls, "exception_atomic", False)
        if not isinstance(exception_atomic, bool):
            raise CompileError(
                "InvalidUdfRecipe",
                "UDF class exception_atomic must be a bool",
            )
        recipe = UdfRecipe(
            _callable_ref(op_cls),
            init_args,
            init_kwargs,
            exception_atomic,
        )
        execution = ExecutionSpec(
            getattr(map_spec, "batch_policy"),
            resources,
            getattr(map_spec, "failure_policy"),
        )
        physical = tuple(
            PhysicalOutputSpec(
                output.id,
                slot,
                output.shape,
                node.outputs[slot].symbolic_key in filter_masks,
            )
            for slot, output in enumerate(outputs)
        )
        return MapOp(
            recipe,
            node.data.call_schema,
            node.data.return_schema,
            execution,
            physical,
        )

    def _node_name(self, node: SymbolicNode, node_id: NodeId) -> str:
        """Create a deterministic diagnostic node name."""

        if node.kind is _SymbolicOpKind.MAP:
            assert isinstance(node.data, _MapTraceData)
            op_cls = getattr(node.data.map_spec, "op_cls")
            return f"map_{int(node_id)}_{op_cls.__name__}"
        return f"{node.kind.value}_{int(node_id)}"

    def _port_name(self, node: SymbolicNode, slot: int) -> str:
        """Create the mandated logical return port name."""

        if node.kind is not _SymbolicOpKind.MAP:
            return "output"
        assert isinstance(node.data, _MapTraceData)
        schema = node.data.return_schema
        if schema.kind is ReturnKind.SINGLE:
            return "output"
        if schema.kind is ReturnKind.TUPLE:
            return f"output_{slot}"
        assert schema.leaves[slot].name is not None
        return typing.cast(str, schema.leaves[slot].name)


def compile(pipeline: object) -> CompiledGraph:
    """Compile a Pipeline-like object through the default GraphCompiler."""

    return GraphCompiler().compile(pipeline)


def verify_frozen_graph(graph: CompiledGraph) -> None:
    """Independently verify frozen IDs, topology, schemas, scopes, and indexes."""

    if not graph.nodes:
        raise CompileError("NoGraphOutputs", "compiled graph has no nodes")
    if not graph.outputs:
        raise CompileError("NoGraphOutputs", "compiled graph has no outputs")
    expected_node_ids = tuple(NodeId(index) for index in range(len(graph.nodes)))
    if tuple(node.id for node in graph.nodes) != expected_node_ids:
        raise CompileError(
            "GraphCycle",
            "node IDs must be contiguous topological order",
        )
    all_ports = tuple(port for node in graph.nodes for port in node.outputs)
    if tuple(port.id for port in all_ports) != tuple(
        PortId(index) for index in range(len(all_ports))
    ):
        raise CompileError(
            "DanglingPort",
            "port IDs must be globally contiguous",
        )
    port_by_id = {port.id: port for port in all_ports}
    if len(port_by_id) != len(all_ports):
        raise CompileError("DanglingPort", "duplicate frozen port ID")
    node_order = {node.id: index for index, node in enumerate(graph.nodes)}
    node_by_id = {node.id: node for node in graph.nodes}
    computed_producers = {
        port.id: node.id for node in graph.nodes for port in node.outputs
    }
    computed_consumers: dict[PortId, list[NodeId]] = {
        port.id: [] for port in all_ports
    }
    source_nodes = [node for node in graph.nodes if isinstance(node.op, SourceOp)]
    if len(source_nodes) != 1 or len(source_nodes[0].outputs) != 1:
        raise CompileError(
            "InvalidForwardSignature",
            "frozen graph must have exactly one one-port source",
        )
    if graph.source != source_nodes[0].outputs[0].id:
        raise CompileError(
            "DanglingPort",
            "graph.source does not name source port",
        )
    source_output = source_nodes[0].outputs[0]
    if (
        source_output.scope_path
        or source_output.occurrence_domain != OccurrenceDomainId(0)
        or source_output.shape != OpaqueShape(None)
    ):
        raise CompileError(
            "ScopePathMismatch",
            "source must start in root scope/domain with opaque Any shape",
        )
    expand_by_scope: dict[ScopeDefId, NodeSpec] = {}
    child_domains: set[OccurrenceDomainId] = set()
    for node in graph.nodes:
        if not isinstance(node.op, ExpandOp):
            continue
        if node.op.scope in expand_by_scope:
            raise CompileError(
                "ScopePathMismatch",
                "two EXPAND nodes share one ScopeDefId",
            )
        expand_by_scope[node.op.scope] = node
    for node in graph.nodes:
        if isinstance(node.op, SourceOp):
            if node.inputs:
                raise CompileError(
                    "DanglingPort",
                    "source node cannot have inputs",
                )
        for binding in node.inputs:
            producer = computed_producers.get(binding.port)
            if producer is None:
                raise CompileError(
                    "DanglingPort",
                    f"input port {int(binding.port)} has no producer",
                )
            if node_order[producer] >= node_order[node.id]:
                raise CompileError(
                    "GraphCycle",
                    "graph is not topologically ordered",
                )
            computed_consumers[binding.port].append(node.id)
        if isinstance(node.op, MapOp):
            _verify_map_node(node, port_by_id)
        elif isinstance(node.op, FilterOp):
            if len(node.inputs) != 2 or len(node.outputs) != 1:
                raise CompileError(
                    "DanglingPort",
                    "FILTER must have two inputs and one output",
                )
            if tuple(binding.port for binding in node.inputs) != (
                node.op.mask,
                node.op.target,
            ):
                raise CompileError(
                    "DanglingPort",
                    "FILTER operation fields do not match input bindings",
                )
            mask = port_by_id[node.op.mask]
            target = port_by_id[node.op.target]
            if not isinstance(mask.shape, BoolShape):
                raise CompileError("FilterMaskNotBool", "frozen mask is not bool")
            mask_producer = node_by_id[computed_producers[mask.id]]
            if not isinstance(mask_producer.op, MapOp):
                raise CompileError(
                    "FilterMaskNotMapOutput",
                    "frozen FILTER mask is not a direct MAP output",
                )
            _verify_aligned_ports(mask, target)
            output = node.outputs[0]
            if (
                output.shape != target.shape
                or output.scope_path != target.scope_path
                or output.occurrence_domain != target.occurrence_domain
            ):
                raise CompileError(
                    "ScopePathMismatch",
                    "FILTER output must preserve target occurrence metadata",
                )
        elif isinstance(node.op, ExpandOp):
            if len(node.inputs) != 1 or len(node.outputs) != 1:
                raise CompileError(
                    "DanglingPort",
                    "EXPAND must have one input/output",
                )
            if node.inputs[0].port != node.op.input:
                raise CompileError(
                    "DanglingPort",
                    "EXPAND operation input does not match its binding",
                )
            source = port_by_id[node.op.input]
            output = node.outputs[0]
            if not isinstance(source.shape, StructuralListShape):
                raise CompileError(
                    "ExpandRequiresStructuralList",
                    "frozen EXPAND source is not structural",
                )
            if output.shape != source.shape.element:
                raise CompileError(
                    "ExpandRequiresStructuralList",
                    "EXPAND output must expose exactly one list layer",
                )
            if output.scope_path != source.scope_path + (node.op.scope,):
                raise CompileError(
                    "ScopePathMismatch",
                    "EXPAND must append its scope definition",
                )
            if output.occurrence_domain == source.occurrence_domain:
                raise CompileError(
                    "OccurrenceDomainMismatch",
                    "EXPAND must introduce a child occurrence domain",
                )
            if output.occurrence_domain in child_domains:
                raise CompileError(
                    "OccurrenceDomainMismatch",
                    "EXPAND child domains must be unique",
                )
            child_domains.add(output.occurrence_domain)
        elif isinstance(node.op, ReduceOp):
            if len(node.inputs) != 1 or len(node.outputs) != 1:
                raise CompileError(
                    "DanglingPort",
                    "REDUCE must have one input/output",
                )
            if node.inputs[0].port != node.op.input:
                raise CompileError(
                    "DanglingPort",
                    "REDUCE operation input does not match its binding",
                )
            source = port_by_id[node.op.input]
            output = node.outputs[0]
            if not source.scope_path:
                raise CompileError("ReduceAtRootScope", "REDUCE input is root")
            if source.scope_path[-1] != node.op.closes_scope:
                raise CompileError(
                    "NonLifoReduce",
                    "REDUCE must close the path-local top scope",
                )
            if output.scope_path != source.scope_path[:-1]:
                raise CompileError(
                    "NonLifoReduce",
                    "REDUCE must pop exactly one scope",
                )
            if output.shape != StructuralListShape(source.shape):
                raise CompileError(
                    "ScopePathMismatch",
                    "REDUCE output shape must retain one collected list layer",
                )
            try:
                matching_expand = expand_by_scope[node.op.closes_scope]
            except KeyError as error:
                raise CompileError(
                    "NonLifoReduce",
                    "REDUCE closes an unknown scope definition",
                ) from error
            expand_input = port_by_id[matching_expand.op.input]
            expand_output = matching_expand.outputs[0]
            if source.occurrence_domain != expand_output.occurrence_domain:
                raise CompileError(
                    "OccurrenceDomainMismatch",
                    "REDUCE input is not in the closed scope domain",
                )
            if output.occurrence_domain != expand_input.occurrence_domain:
                raise CompileError(
                    "OccurrenceDomainMismatch",
                    "REDUCE must restore the parent occurrence domain",
                )
    if dict(graph.producer_by_port) != computed_producers:
        raise CompileError("DanglingPort", "producer_by_port index is inconsistent")
    expected_consumers = {
        port: tuple(consumers)
        for port, consumers in computed_consumers.items()
    }
    if dict(graph.consumers_by_port) != expected_consumers:
        raise CompileError(
            "DanglingPort",
            "consumers_by_port index is inconsistent",
        )
    for node in graph.nodes:
        if not isinstance(node.op, MapOp):
            continue
        for output, physical in zip(
            node.outputs,
            node.op.physical_outputs,
        ):
            expected_control = any(
                isinstance(node_by_id[consumer].op, FilterOp)
                and node_by_id[consumer].op.mask == output.id
                for consumer in computed_consumers[output.id]
            )
            if physical.emit_control_bits != expected_control:
                raise CompileError(
                    "FilterMaskNotBool",
                    "control-bit emission must exactly match FILTER mask use",
                )
    output_names = tuple(output.name for output in graph.outputs)
    if len(output_names) != len(set(output_names)):
        raise CompileError(
            "DuplicateGraphOutputName",
            "graph output names must be unique",
        )
    for output in graph.outputs:
        try:
            port = port_by_id[output.port]
        except KeyError as error:
            raise CompileError(
                "DanglingPort",
                "graph output references an unknown port",
            ) from error
        if port.scope_path:
            raise CompileError(
                "UnclosedOutputScope",
                f"graph output {output.name!r} is scoped",
            )
    if set(graph.scope_plans) != set(expand_by_scope):
        raise CompileError(
            "ScopePathMismatch",
            "scope_plans must contain every and only EXPAND definition",
        )
    for definition, plan in graph.scope_plans.items():
        if plan.definition != definition:
            raise CompileError(
                "ScopePathMismatch",
                "ScopePlan key/definition mismatch",
            )
        expand_node = graph.node(plan.expand_node)
        if not isinstance(expand_node.op, ExpandOp):
            raise CompileError(
                "ScopePathMismatch",
                "ScopePlan expand_node is not EXPAND",
            )
        if expand_node.op.scope != definition:
            raise CompileError(
                "ScopePathMismatch",
                "ScopePlan points at another scope definition",
            )
        if not plan.reducers:
            raise CompileError(
                "UnclosedOutputScope",
                "scope plan has no reducers",
            )
        expected_reducers = tuple(
            node.id
            for node in graph.nodes
            if isinstance(node.op, ReduceOp)
            and node.op.closes_scope == definition
        )
        if plan.reducers != expected_reducers:
            raise CompileError(
                "NonLifoReduce",
                "ScopePlan must route to every matching REDUCE in order",
            )
        for reducer in plan.reducers:
            reducer_node = graph.node(reducer)
            if (
                not isinstance(reducer_node.op, ReduceOp)
                or reducer_node.op.closes_scope != definition
            ):
                raise CompileError(
                    "NonLifoReduce",
                    "ScopePlan reducer closes another scope",
                )
    try:
        expected_fingerprint = GraphFingerprint.derive(
            "compiled-graph",
            graph.source,
            graph.nodes,
            graph.outputs,
            tuple(graph.scope_plans.items()),
        )
    except (TypeError, ValueError) as error:
        raise CompileError(
            "UnserializableUdfRecipe",
            f"cannot verify graph fingerprint: {error}",
        ) from error
    if graph.fingerprint != expected_fingerprint:
        raise CompileError(
            "GraphFingerprintMismatch",
            "frozen graph fingerprint does not match canonical records",
        )


def _verify_map_node(
    node: NodeSpec,
    port_by_id: Mapping[PortId, PortSpec],
) -> None:
    """Verify MAP CallSchema/ReturnSchema/physical output cross-records."""

    assert isinstance(node.op, MapOp)
    operation = node.op
    count = len(operation.return_schema.leaves)
    if count != len(node.outputs) or count != len(operation.physical_outputs):
        raise CompileError(
            "NoMapOutputs",
            "MAP return/logical/physical output arity mismatch",
        )
    if len(node.inputs) != len(operation.call_schema.parameters):
        raise CompileError(
            "InvalidMapCall",
            "MAP input bindings do not match CallSchema",
        )
    if not node.inputs:
        raise CompileError(
            "UnsupportedRunParameter",
            "MAP must consume at least one input role",
        )
    input_ports = tuple(port_by_id[binding.port] for binding in node.inputs)
    first_input = input_ports[0]
    for binding, parameter in zip(
        node.inputs,
        operation.call_schema.parameters,
    ):
        if binding.role != parameter.name or binding.parameter != parameter:
            raise CompileError(
                "InvalidMapCall",
                "MAP binding order/schema mismatch",
            )
        input_port = port_by_id[binding.port]
        if not shapes_compatible(input_port.shape, parameter.item_shape):
            raise CompileError(
                "InvalidMapCall",
                f"MAP input {binding.role!r} shape violates CallSchema",
            )
        _verify_aligned_ports(first_input, input_port)
    for output in node.outputs:
        if (
            output.scope_path != first_input.scope_path
            or output.occurrence_domain != first_input.occurrence_domain
        ):
            raise CompileError(
                "ScopePathMismatch",
                "MAP outputs must preserve input occurrence metadata",
            )
    for slot, (leaf, output, physical) in enumerate(
        zip(
            operation.return_schema.leaves,
            node.outputs,
            operation.physical_outputs,
        )
    ):
        if leaf.slot != slot or physical.return_slot != slot:
            raise CompileError(
                "UnsupportedReturnStructure",
                "MAP return slots must be contiguous and aligned",
            )
        if (
            leaf.item_shape != output.shape
            or physical.shape != output.shape
            or physical.port != output.id
        ):
            raise CompileError(
                "UnsupportedReturnStructure",
                "MAP return leaf/output/physical shape mismatch",
            )
        if physical.emit_control_bits and not isinstance(output.shape, BoolShape):
            raise CompileError(
                "FilterMaskNotBool",
                "non-bool MAP output emits control bits",
            )


def _verify_aligned_ports(left: PortSpec, right: PortSpec) -> None:
    """Verify exact scope and occurrence alignment for a system operation."""

    if left.scope_path != right.scope_path:
        raise CompileError(
            "ScopePathMismatch",
            "system inputs have different scope paths",
        )
    if left.occurrence_domain != right.occurrence_domain:
        raise CompileError(
            "OccurrenceDomainMismatch",
            "system inputs have different occurrence domains",
        )


__all__ = [
    "BatchPolicy",
    "BoolShape",
    "CallSchema",
    "CompileError",
    "CompiledGraph",
    "ExecutionSpec",
    "ExpandOp",
    "FailurePolicy",
    "FilterOp",
    "GraphCompiler",
    "GraphOutputSpec",
    "InputBinding",
    "MapOp",
    "NodeOp",
    "NodeSpec",
    "OpaqueShape",
    "OpaqueValue",
    "ParameterKind",
    "ParameterSpec",
    "PhysicalOutputSpec",
    "PortSpec",
    "ReduceOp",
    "ResourceSpec",
    "ReturnKind",
    "ReturnLeafSpec",
    "ReturnSchema",
    "ScopePlan",
    "SerializableCallableRef",
    "SourceOp",
    "StructuralListShape",
    "SymbolicGraph",
    "SymbolicNode",
    "SymbolicPort",
    "TraceContext",
    "TraceOwnerId",
    "TypeRef",
    "UdfRecipe",
    "ValueShape",
    "compile",
    "freeze_config_value",
    "freeze_constructor_arguments",
    "freeze_resource_spec",
    "freeze_runtime_env",
    "infer_item_shape",
    "infer_run_schemas",
    "shapes_compatible",
    "thaw_config_value",
    "validate_constructor_binding",
    "verify_frozen_graph",
]
