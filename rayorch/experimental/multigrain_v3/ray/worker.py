"""Persistent Ray worker actor and pure MAP batch adapters."""

from __future__ import annotations

import hashlib
import importlib
import os
import resource
import time
import traceback
from dataclasses import dataclass
from typing import Iterator

import ray

from ..model.graph import (
    BoolShape,
    CallSchema,
    OpaqueShape,
    ParameterKind,
    PhysicalOutputSpec,
    ReturnKind,
    ReturnSchema,
    StructuralListShape,
    TypeRef,
    thaw_config_value,
)
from ..model.semantics import GraphFingerprint, NodeId
from .protocol import (
    EMPTY_BLOCK,
    PROTOCOL_VERSION,
    FailureManifest,
    ManifestHeader,
    OutputLayout,
    ProtocolValidationError,
    SuccessManifest,
    WireGather,
    WireList,
    WorkerDispatch,
    WorkerErrorKind,
    header_for,
    truncate_utf8,
    validate_manifest,
    validate_wire_gather,
    validate_worker_dispatch,
)


class WorkerContractError(RuntimeError):
    """Report a worker-side violation of the frozen graph contract."""


class BadGrainError(RuntimeError):
    """Attribute a data-dependent UDF failure to exactly one batch entry."""

    def __init__(self, index: int, message: str) -> None:
        """Create a checked, entry-local failure."""

        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("BadGrainError.index must be an int")
        if index < 0:
            raise ValueError("BadGrainError.index must be non-negative")
        if not isinstance(message, str):
            raise TypeError("BadGrainError.message must be a string")
        super().__init__(message)
        self.index = index


@dataclass(frozen=True, slots=True)
class WorkerContext:
    """Freeze all graph and ABI facts owned by one persistent MAP actor."""

    protocol_version: int
    graph_fingerprint: GraphFingerprint
    node: NodeId
    call_schema: CallSchema
    return_schema: ReturnSchema
    output_schema: tuple[PhysicalOutputSpec, ...]
    max_manifest_bytes: int
    max_error_message_bytes: int

    def __post_init__(self) -> None:
        """Cross-check protocol, call, return, and physical output schemas."""

        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(
                f"WorkerContext requires protocol version {PROTOCOL_VERSION}"
            )
        if not isinstance(self.graph_fingerprint, GraphFingerprint):
            raise TypeError("graph_fingerprint must be a GraphFingerprint")
        if not isinstance(self.node, NodeId):
            raise TypeError("node must be a NodeId")
        if not isinstance(self.call_schema, CallSchema):
            raise TypeError("call_schema must be a CallSchema")
        if not isinstance(self.return_schema, ReturnSchema):
            raise TypeError("return_schema must be a ReturnSchema")
        if not isinstance(self.output_schema, tuple):
            raise TypeError("output_schema must be a tuple")
        if not self.output_schema:
            raise ValueError("output_schema must not be empty")
        if any(
            not isinstance(output, PhysicalOutputSpec)
            for output in self.output_schema
        ):
            raise TypeError(
                "output_schema must contain PhysicalOutputSpec records"
            )
        _validate_call_schema(self.call_schema)
        _validate_return_and_output_schema(
            self.return_schema,
            self.output_schema,
        )
        _positive_int(self.max_manifest_bytes, "max_manifest_bytes")
        _nonnegative_int(
            self.max_error_message_bytes,
            "max_error_message_bytes",
        )


def _nonnegative_int(value: object, field: str) -> int:
    """Validate an integer while excluding bool."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an int")
    if value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def _positive_int(value: object, field: str) -> int:
    """Validate a strictly positive integer while excluding bool."""

    result = _nonnegative_int(value, field)
    if result == 0:
        raise ValueError(f"{field} must be positive")
    return result


def _validate_call_schema(schema: CallSchema) -> None:
    """Verify the exact positional/keyword reconstruction frozen by compile."""

    parameters = schema.parameters
    if not isinstance(parameters, tuple):
        raise TypeError("CallSchema.parameters must be a tuple")
    if not parameters:
        raise ValueError("CallSchema.parameters must not be empty")
    positional_count = _nonnegative_int(
        schema.positional_count,
        "CallSchema.positional_count",
    )
    if positional_count > len(parameters):
        raise ValueError("CallSchema.positional_count exceeds parameter count")
    if not isinstance(schema.keyword_roles, tuple):
        raise TypeError("CallSchema.keyword_roles must be a tuple")
    if len(schema.keyword_roles) != len(parameters) - positional_count:
        raise ValueError("CallSchema keyword role count is inconsistent")

    names: list[str] = []
    for index, parameter in enumerate(parameters):
        if parameter.index != index:
            raise ValueError("CallSchema parameter indices must be contiguous")
        if not isinstance(parameter.name, str) or not parameter.name:
            raise ValueError("CallSchema parameter names must be non-empty")
        if parameter.name in names:
            raise ValueError("CallSchema parameter names must be unique")
        names.append(parameter.name)
        if (
            index < positional_count
            and parameter.kind is ParameterKind.KEYWORD_ONLY
        ):
            raise ValueError("keyword-only parameter cannot be reconstructed positionally")
        if (
            index >= positional_count
            and parameter.kind is ParameterKind.POSITIONAL_ONLY
        ):
            raise ValueError("positional-only parameter cannot be reconstructed by name")

    expected_keywords = tuple(names[positional_count:])
    if schema.keyword_roles != expected_keywords:
        raise ValueError(
            "CallSchema.keyword_roles must match non-positional parameter order"
        )


def _validate_return_and_output_schema(
    return_schema: ReturnSchema,
    output_schema: tuple[PhysicalOutputSpec, ...],
) -> None:
    """Verify return leaves and physical outputs form one frozen mapping."""

    if not isinstance(return_schema.leaves, tuple) or not return_schema.leaves:
        raise ValueError("ReturnSchema.leaves must be a non-empty tuple")
    if len(return_schema.leaves) != len(output_schema):
        raise ValueError("return and physical output counts differ")
    for slot, (leaf, output) in enumerate(
        zip(return_schema.leaves, output_schema)
    ):
        if leaf.slot != slot or output.return_slot != slot:
            raise ValueError("return slots must be contiguous and ordered")
        if leaf.item_shape != output.shape:
            raise ValueError("return leaf and physical output shapes differ")

    if return_schema.kind is ReturnKind.SINGLE:
        if len(return_schema.leaves) != 1:
            raise ValueError("single return schema must have one leaf")
        if return_schema.leaves[0].name is not None:
            raise ValueError("single return leaf must be anonymous")
        if return_schema.named_tuple_type is not None:
            raise ValueError("single return schema cannot name a tuple type")
    elif return_schema.kind is ReturnKind.TUPLE:
        if any(leaf.name is not None for leaf in return_schema.leaves):
            raise ValueError("plain tuple leaves must be anonymous")
        if return_schema.named_tuple_type is not None:
            raise ValueError("plain tuple schema cannot name a tuple type")
    elif return_schema.kind is ReturnKind.NAMED_TUPLE:
        if return_schema.named_tuple_type is None:
            raise ValueError("NamedTuple return schema requires its TypeRef")
        names = tuple(leaf.name for leaf in return_schema.leaves)
        if any(name is None for name in names) or len(set(names)) != len(names):
            raise ValueError("NamedTuple return leaves require unique names")
    else:
        raise ValueError(f"unsupported ReturnKind: {return_schema.kind!r}")


def interpret_wire_gather(
    gather: WireGather,
    input_blocks: tuple[object, ...],
    *,
    max_depth: int | None = None,
    max_nodes: int | None = None,
) -> object:
    """Iteratively rebuild one value without Python recursion."""

    if not isinstance(input_blocks, tuple):
        raise TypeError("input_blocks must be a tuple")
    validate_wire_gather(
        gather,
        ref_count=len(input_blocks),
        max_depth=max_depth,
        max_nodes=max_nodes,
    )

    work: list[tuple[WireGather, bool]] = [(gather, False)]
    values: list[object] = []
    while work:
        node, visited = work.pop()
        if not isinstance(node, WireList):
            try:
                values.append(input_blocks[node.ref_slot][node.row])  # type: ignore[index]
            except (IndexError, KeyError, TypeError) as exc:
                raise WorkerContractError(
                    "wire selector addresses a missing input block row"
                ) from exc
            continue
        if not visited:
            work.append((node, True))
            for child in reversed(node.children):
                work.append((child, False))
            continue
        count = len(node.children)
        if count:
            children = values[-count:]
            del values[-count:]
        else:
            children = []
        values.append(list(children))

    if len(values) != 1:
        raise WorkerContractError("wire gather interpreter produced invalid stack")
    return values[0]


def gather_columns(
    dispatch: WorkerDispatch,
    input_blocks: tuple[object, ...],
    call_schema: CallSchema,
) -> tuple[list[object], ...]:
    """Reconstruct one dense batch column for each frozen call role."""

    role_count = len(call_schema.parameters)
    validate_worker_dispatch(
        dispatch,
        input_ref_count=len(input_blocks),
        role_count=role_count,
    )
    columns: list[list[object]] = [[] for _ in range(role_count)]
    for entry in dispatch.entries:
        for role, tree in enumerate(entry.role_trees):
            columns[role].append(interpret_wire_gather(tree, input_blocks))
    return tuple(columns)


def reconstruct_call(
    schema: CallSchema,
    columns: tuple[list[object], ...],
) -> tuple[tuple[list[object], ...], dict[str, list[object]]]:
    """Recreate positional args and kwargs exactly as frozen by ``CallSchema``."""

    _validate_call_schema(schema)
    if not isinstance(columns, tuple):
        raise TypeError("columns must be a tuple")
    if len(columns) != len(schema.parameters):
        raise WorkerContractError("column count does not match CallSchema")
    if any(not isinstance(column, list) for column in columns):
        raise WorkerContractError("every reconstructed argument must be a list")
    positional = tuple(columns[: schema.positional_count])
    keyword_columns = columns[schema.positional_count :]
    keywords = dict(zip(schema.keyword_roles, keyword_columns))
    return positional, keywords


def _resolve_type_ref(type_ref: TypeRef) -> type[object]:
    """Resolve the concrete outer class required by a frozen ``TypeRef``."""

    try:
        target: object = importlib.import_module(type_ref.module)
        for component in type_ref.qualname.split("."):
            if component == "<locals>":
                raise WorkerContractError("local output types are not importable")
            target = getattr(target, component)
    except (AttributeError, ImportError) as exc:
        raise WorkerContractError(
            f"cannot resolve output type "
            f"{type_ref.module}.{type_ref.qualname}"
        ) from exc
    if not isinstance(target, type):
        raise WorkerContractError(
            f"output TypeRef {type_ref.module}.{type_ref.qualname} "
            "does not resolve to a class"
        )
    return target


def _validate_item(value: object, shape: object, path: str) -> None:
    """Validate one item at only the structural layer exposed by the graph."""

    if isinstance(shape, BoolShape):
        if type(value) is not bool:
            raise WorkerContractError(f"{path} must be bool")
        return
    if isinstance(shape, StructuralListShape):
        if not isinstance(value, list):
            raise WorkerContractError(f"{path} must be a list")
        return
    if not isinstance(shape, OpaqueShape):
        raise WorkerContractError(f"{path} has an unsupported ValueShape")
    if shape.type_ref is None:
        return
    concrete = _resolve_type_ref(shape.type_ref)
    if not isinstance(value, concrete):
        raise WorkerContractError(
            f"{path} must be an instance of "
            f"{shape.type_ref.module}.{shape.type_ref.qualname}"
        )


def split_runtime_return(
    result: object,
    schema: ReturnSchema,
) -> tuple[list[object], ...]:
    """Strictly split single, plain tuple, or exact NamedTuple returns."""

    if schema.kind is ReturnKind.SINGLE:
        if not isinstance(result, list):
            raise WorkerContractError("single-output UDF must return a list")
        return (result,)

    if schema.kind is ReturnKind.TUPLE:
        if type(result) is not tuple:
            raise WorkerContractError("tuple-output UDF must return a plain tuple")
        if len(result) != len(schema.leaves):
            raise WorkerContractError("runtime tuple return arity is incorrect")
        if any(not isinstance(column, list) for column in result):
            raise WorkerContractError("every tuple return leaf must be a list")
        return result

    if schema.kind is not ReturnKind.NAMED_TUPLE:
        raise WorkerContractError(f"unsupported ReturnKind: {schema.kind!r}")
    expected_type = schema.named_tuple_type
    if expected_type is None:
        raise WorkerContractError("NamedTuple schema has no type identity")
    result_type = type(result)
    if not (
        isinstance(result, tuple)
        and hasattr(result_type, "_fields")
        and result_type.__module__ == expected_type.module
        and result_type.__qualname__ == expected_type.qualname
    ):
        raise WorkerContractError("UDF must return the exact declared NamedTuple")
    expected_fields = tuple(leaf.name for leaf in schema.leaves)
    if tuple(result_type._fields) != expected_fields:  # type: ignore[attr-defined]
        raise WorkerContractError("runtime NamedTuple fields do not match schema")
    if len(result) != len(schema.leaves):
        raise WorkerContractError("runtime NamedTuple return arity is incorrect")
    if any(not isinstance(column, list) for column in result):
        raise WorkerContractError("every NamedTuple return leaf must be a list")
    return tuple(result)


def _control_bitset(column: list[object]) -> bytes:
    """Pack a bool column with logical index ``i`` in bit ``i`` (LSB first)."""

    packed = bytearray((len(column) + 7) // 8)
    for index, value in enumerate(column):
        if value:
            packed[index // 8] |= 1 << (index % 8)
    return bytes(packed)


def normalize_outputs(
    result: object,
    *,
    batch_size: int,
    return_schema: ReturnSchema,
    output_schema: tuple[PhysicalOutputSpec, ...],
) -> tuple[tuple[list[object], ...], tuple[OutputLayout, ...]]:
    """Validate and normalize scalar, bool, and one-layer list columns."""

    _nonnegative_int(batch_size, "batch_size")
    leaves = split_runtime_return(result, return_schema)
    if len(leaves) != len(output_schema):
        raise WorkerContractError("runtime output count differs from physical schema")

    output_columns: list[list[object]] = []
    layouts: list[OutputLayout] = []
    for slot, (column, output) in enumerate(zip(leaves, output_schema)):
        if len(column) != batch_size:
            raise WorkerContractError(
                f"return slot {slot} has {len(column)} rows; expected {batch_size}"
            )
        shape = output.shape
        if isinstance(shape, StructuralListShape):
            flattened: list[object] = []
            offsets = [0]
            for row, value in enumerate(column):
                if not isinstance(value, list):
                    raise WorkerContractError(
                        f"return slot {slot} row {row} must be a list"
                    )
                for child, item in enumerate(value):
                    _validate_item(
                        item,
                        shape.element,
                        f"return slot {slot} row {row} item {child}",
                    )
                flattened.extend(value)
                offsets.append(len(flattened))
            output_columns.append(flattened)
            layouts.append(
                OutputLayout(
                    return_slot=slot,
                    port=output.port,
                    shape=shape,
                    logical_count=batch_size,
                    row_count=len(flattened),
                    offsets=tuple(offsets),
                    control_bits=None,
                    estimated_bytes=None,
                )
            )
            continue

        normalized = list(column)
        for row, value in enumerate(normalized):
            _validate_item(value, shape, f"return slot {slot} row {row}")
        control_bits = (
            _control_bitset(normalized)
            if isinstance(shape, BoolShape) and output.emit_control_bits
            else None
        )
        output_columns.append(normalized)
        layouts.append(
            OutputLayout(
                return_slot=slot,
                port=output.port,
                shape=shape,
                logical_count=batch_size,
                row_count=batch_size,
                offsets=None,
                control_bits=control_bits,
                estimated_bytes=None,
            )
        )
    return tuple(output_columns), tuple(layouts)


def _load_qualified(module: str, qualname: str) -> object:
    """Load one importable object by its canonical module and qualname."""

    target: object = importlib.import_module(module)
    for component in qualname.split("."):
        if component == "<locals>":
            raise WorkerContractError("local UDF factories are not importable")
        target = getattr(target, component)
    return target


def instantiate_udf(recipe: object) -> object:
    """Instantiate a recipe after restoring ordinary constructor containers."""

    factory_ref = getattr(recipe, "factory", None)
    if factory_ref is not None:
        factory = _load_qualified(factory_ref.module, factory_ref.qualname)
        restored = thaw_config_value(
            (
                tuple(getattr(recipe, "init_args", ())),
                dict(getattr(recipe, "init_kwargs", ())),
            )
        )
        if (
            not isinstance(restored, tuple)
            or len(restored) != 2
            or not isinstance(restored[0], tuple)
            or not isinstance(restored[1], dict)
        ):
            raise WorkerContractError(
                "thawed UDF constructor recipe has invalid structure"
            )
        init_args, init_kwargs = restored
        if not callable(factory):
            raise WorkerContractError("UDF recipe factory is not callable")
        udf = factory(*init_args, **init_kwargs)
    elif isinstance(recipe, type):
        udf = recipe()
    elif hasattr(recipe, "run"):
        udf = recipe
    elif callable(recipe):
        udf = recipe()
    else:
        raise WorkerContractError("unsupported UDF recipe")
    if not callable(getattr(udf, "run", None)):
        raise WorkerContractError("instantiated UDF has no callable run method")
    return udf


def _rss_bytes() -> int | None:
    """Return the worker process high-water RSS when the platform exposes it."""

    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (OSError, ValueError):
        return None
    # Linux reports KiB; macOS reports bytes.  Ray's supported Linux workers are
    # the production target for this backend.
    return int(usage) * 1024


def _trace_digest() -> bytes | None:
    """Digest the active traceback without putting it in the wire manifest."""

    rendered = traceback.format_exc()
    if not rendered or rendered == "NoneType: None\n":
        return None
    return hashlib.blake2b(rendered.encode("utf-8"), digest_size=16).digest()


def _error_type(exc: BaseException) -> str:
    """Return a stable qualified exception type name."""

    cls = type(exc)
    return f"{cls.__module__}.{cls.__qualname__}"


def _failure_manifest(
    context: WorkerContext,
    header: ManifestHeader,
    exc: Exception,
    kind: WorkerErrorKind,
    bad_entry_index: int | None,
) -> FailureManifest:
    """Build a byte-bounded failure manifest or fail the generator safely."""

    message = truncate_utf8(str(exc), context.max_error_message_bytes)
    manifest = FailureManifest(
        header=header,
        kind=kind,
        bad_entry_index=bad_entry_index,
        error_type=_error_type(exc),
        message=message,
        trace_digest=_trace_digest(),
    )
    try:
        validate_manifest(
            manifest,
            max_bytes=context.max_manifest_bytes,
            expected_header=header,
            expected_output_count=len(context.output_schema),
            expected_output_schema=context.output_schema,
        )
        return manifest
    except ProtocolValidationError:
        compact = FailureManifest(
            header=header,
            kind=kind,
            bad_entry_index=bad_entry_index,
            error_type=type(exc).__name__,
            message="",
            trace_digest=None,
        )
        try:
            validate_manifest(
                compact,
                max_bytes=context.max_manifest_bytes,
                expected_header=header,
                expected_output_count=len(context.output_schema),
                expected_output_schema=context.output_schema,
            )
        except ProtocolValidationError as size_error:
            raise WorkerContractError(
                "max_manifest_bytes is too small for a failure manifest"
            ) from size_error
        return compact


@ray.remote(max_restarts=0, max_task_retries=0, max_concurrency=1)
class Worker:
    """Persist one MAP UDF instance and execute one dispatch at a time."""

    def __init__(self, context: WorkerContext, udf_recipe: object) -> None:
        """Create the UDF once inside its owning Ray actor process."""

        if not isinstance(context, WorkerContext):
            raise TypeError("context must be a WorkerContext")
        self.context = context
        self.udf = instantiate_udf(udf_recipe)
        self.calls = 0

    def ready(self) -> dict[str, int]:
        """Confirm actor/UDF initialization without executing a data batch."""

        return {"pid": os.getpid(), "calls": self.calls}

    def execute(
        self,
        dispatch: WorkerDispatch,
        *input_blocks: object,
    ) -> Iterator[object]:
        """Yield ordered data blocks and then exactly one commit-gate manifest."""

        started_at = time.monotonic()
        try:
            if not isinstance(dispatch, WorkerDispatch):
                raise WorkerContractError("dispatch is not a WorkerDispatch")
            header = header_for(dispatch)
        except Exception:
            # Without a valid wire dispatch there is no trustworthy header to
            # echo, so terminating the generator is safer than fabricating one.
            raise

        try:
            validate_worker_dispatch(
                dispatch,
                expected_version=self.context.protocol_version,
                input_ref_count=len(input_blocks),
                role_count=len(self.context.call_schema.parameters),
            )
            if dispatch.graph_fingerprint != self.context.graph_fingerprint:
                raise WorkerContractError("dispatch graph fingerprint is stale")
            if dispatch.node != self.context.node:
                raise WorkerContractError("dispatch targets a different MAP node")
            columns = gather_columns(
                dispatch,
                tuple(input_blocks),
                self.context.call_schema,
            )
            args, kwargs = reconstruct_call(self.context.call_schema, columns)
        except Exception as exc:
            manifest = _failure_manifest(
                self.context,
                header,
                exc,
                WorkerErrorKind.CONTRACT,
                None,
            )
            for _ in self.context.output_schema:
                yield EMPTY_BLOCK
            yield manifest
            return

        try:
            self.calls += 1
            result = self.udf.run(*args, **kwargs)
        except BadGrainError as exc:
            if exc.index >= len(dispatch.entries):
                replacement = WorkerContractError(
                    "BadGrainError.index is outside the dispatch batch"
                )
                manifest = _failure_manifest(
                    self.context,
                    header,
                    replacement,
                    WorkerErrorKind.CONTRACT,
                    None,
                )
            else:
                manifest = _failure_manifest(
                    self.context,
                    header,
                    exc,
                    WorkerErrorKind.BAD_GRAIN,
                    exc.index,
                )
            for _ in self.context.output_schema:
                yield EMPTY_BLOCK
            yield manifest
            return
        except Exception as exc:
            manifest = _failure_manifest(
                self.context,
                header,
                exc,
                WorkerErrorKind.GENERIC_UDF,
                None,
            )
            for _ in self.context.output_schema:
                yield EMPTY_BLOCK
            yield manifest
            return

        try:
            output_columns, layouts = normalize_outputs(
                result,
                batch_size=len(dispatch.entries),
                return_schema=self.context.return_schema,
                output_schema=self.context.output_schema,
            )
            manifest = SuccessManifest(
                header=header,
                outputs=layouts,
                worker_started_at=started_at,
                worker_finished_at=time.monotonic(),
                worker_rss_bytes=_rss_bytes(),
            )
            validate_manifest(
                manifest,
                max_bytes=self.context.max_manifest_bytes,
                expected_header=header,
                expected_output_count=len(self.context.output_schema),
                expected_output_schema=self.context.output_schema,
            )
        except Exception as exc:
            contract = (
                exc
                if isinstance(exc, WorkerContractError)
                else WorkerContractError(str(exc))
            )
            failure = _failure_manifest(
                self.context,
                header,
                contract,
                WorkerErrorKind.CONTRACT,
                None,
            )
            for _ in self.context.output_schema:
                yield EMPTY_BLOCK
            yield failure
            return

        for column in output_columns:
            yield column
        yield manifest
