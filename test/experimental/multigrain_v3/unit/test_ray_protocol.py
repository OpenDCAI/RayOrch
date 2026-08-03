from __future__ import annotations

from typing import NamedTuple

import pytest

from rayorch.experimental.multigrain_v3.model.graph import (
    BoolShape,
    CallSchema,
    OpaqueShape,
    ParameterKind,
    ParameterSpec,
    PhysicalOutputSpec,
    ReturnKind,
    ReturnLeafSpec,
    ReturnSchema,
    SerializableCallableRef,
    StructuralListShape,
    TypeRef,
    UdfRecipe,
    freeze_constructor_arguments,
    thaw_config_value,
)
from rayorch.experimental.multigrain_v3.model.semantics import (
    ActorId,
    AttemptToken,
    DispatchId,
    GrainId,
    GraphFingerprint,
    LeaseId,
    NodeId,
    PortId,
    RunId,
)
from rayorch.experimental.multigrain_v3.ray.protocol import (
    PROTOCOL_VERSION,
    ActorLease,
    FailureManifest,
    ManifestHeader,
    OutputLayout,
    ProtocolValidationError,
    SlotTake,
    SuccessManifest,
    WireList,
    WorkerDispatch,
    WorkerEntry,
    WorkerErrorKind,
    header_for,
    truncate_utf8,
    validate_manifest,
    validate_output_layouts,
    validate_wire_gather,
    validate_worker_dispatch,
)
from rayorch.experimental.multigrain_v3.ray.worker import (
    WorkerContext,
    WorkerContractError,
    gather_columns,
    interpret_wire_gather,
    instantiate_udf,
    normalize_outputs,
    reconstruct_call,
    split_runtime_return,
)


STR = OpaqueShape(TypeRef("builtins", "str"))


class NamedResult(NamedTuple):
    texts: list[str]
    flags: list[bool]


class ConstructorProbe:
    """Capture worker-thawed constructor containers for protocol tests."""

    def __init__(
        self,
        config: dict[str, object],
        *,
        labels: list[str],
    ) -> None:
        """Store restored ordinary Python values."""

        self.config = config
        self.labels = labels

    def run(self, values: list[str]) -> list[str]:
        """Return the input column unchanged."""

        return values


def _identity():
    run = RunId.new()
    node = NodeId(3)
    attempts = tuple(
        AttemptToken(run=run, grain=GrainId.new(), generation=1)
        for _ in range(2)
    )
    lease = ActorLease(
        node=node,
        slot=0,
        incarnation=2,
        actor_id=ActorId.new(),
        lease_id=LeaseId.new(),
    )
    entries = tuple(
        WorkerEntry(token=token, role_trees=(SlotTake(0, row),))
        for row, token in enumerate(attempts)
    )
    dispatch = WorkerDispatch(
        protocol_version=PROTOCOL_VERSION,
        graph_fingerprint=GraphFingerprint.derive("test-graph", "protocol"),
        run=run,
        dispatch=DispatchId.new(),
        node=node,
        lease=lease,
        entries=entries,
    )
    return dispatch, attempts


def _single_schema(shape=STR):
    call = CallSchema(
        parameters=(
            ParameterSpec(
                name="values",
                index=0,
                kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                item_shape=STR,
            ),
        ),
        positional_count=1,
        keyword_roles=(),
    )
    returns = ReturnSchema(
        kind=ReturnKind.SINGLE,
        leaves=(ReturnLeafSpec(slot=0, name=None, item_shape=shape),),
        named_tuple_type=None,
    )
    outputs = (
        PhysicalOutputSpec(
            port=PortId(8),
            return_slot=0,
            shape=shape,
            emit_control_bits=isinstance(shape, BoolShape),
        ),
    )
    return call, returns, outputs


def test_wire_gather_iterative_validation_and_interpretation():
    gather = WireList(
        (
            SlotTake(0, 1),
            WireList((SlotTake(1, 0), WireList(()))),
        )
    )
    assert validate_wire_gather(
        gather,
        ref_count=2,
        max_depth=2,
        max_nodes=5,
    ) == (5, 2)
    assert interpret_wire_gather(
        gather,
        (["a", "b"], ["c"]),
    ) == ["b", ["c", []]]

    with pytest.raises(ProtocolValidationError, match="ref_slot"):
        validate_wire_gather(gather, ref_count=1)
    with pytest.raises(ProtocolValidationError, match="max_depth"):
        validate_wire_gather(gather, max_depth=1)


def test_dispatch_and_header_validate_exact_identity_and_role_arity():
    dispatch, attempts = _identity()
    validate_worker_dispatch(dispatch, input_ref_count=1, role_count=1)
    header = header_for(dispatch)
    assert header.attempts == attempts
    assert header.lease == dispatch.lease

    wrong_version = WorkerDispatch(
        protocol_version=PROTOCOL_VERSION + 1,
        graph_fingerprint=dispatch.graph_fingerprint,
        run=dispatch.run,
        dispatch=dispatch.dispatch,
        node=dispatch.node,
        lease=dispatch.lease,
        entries=dispatch.entries,
    )
    with pytest.raises(ProtocolValidationError, match="unsupported protocol"):
        validate_worker_dispatch(wrong_version)


def test_call_schema_rebuilds_positional_and_keyword_columns():
    schema = CallSchema(
        parameters=(
            ParameterSpec(
                name="left",
                index=0,
                kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                item_shape=STR,
            ),
            ParameterSpec(
                name="right",
                index=1,
                kind=ParameterKind.KEYWORD_ONLY,
                item_shape=STR,
            ),
        ),
        positional_count=1,
        keyword_roles=("right",),
    )
    left = ["a", "b"]
    right = ["x", "y"]
    args, kwargs = reconstruct_call(schema, (left, right))
    assert args == (left,)
    assert kwargs == {"right": right}

    dispatch, _ = _identity()
    assert gather_columns(
        dispatch,
        (["first", "second"],),
        _single_schema()[0],
    ) == (["first", "second"],)


def test_runtime_return_structure_is_strict_for_tuple_and_namedtuple():
    tuple_schema = ReturnSchema(
        kind=ReturnKind.TUPLE,
        leaves=(
            ReturnLeafSpec(slot=0, name=None, item_shape=STR),
            ReturnLeafSpec(slot=1, name=None, item_shape=BoolShape()),
        ),
        named_tuple_type=None,
    )
    assert split_runtime_return((["a"], [True]), tuple_schema) == (
        ["a"],
        [True],
    )
    with pytest.raises(WorkerContractError, match="plain tuple"):
        split_runtime_return(NamedResult(["a"], [True]), tuple_schema)

    named_schema = ReturnSchema(
        kind=ReturnKind.NAMED_TUPLE,
        leaves=(
            ReturnLeafSpec(slot=0, name="texts", item_shape=STR),
            ReturnLeafSpec(slot=1, name="flags", item_shape=BoolShape()),
        ),
        named_tuple_type=TypeRef(__name__, "NamedResult"),
    )
    result = NamedResult(["a"], [True])
    assert split_runtime_return(result, named_schema) == (["a"], [True])
    with pytest.raises(WorkerContractError, match="exact declared NamedTuple"):
        split_runtime_return((["a"], [True]), named_schema)


def test_output_normalization_scalar_bool_and_one_structural_layer():
    list_shape = StructuralListShape(STR)
    return_schema = ReturnSchema(
        kind=ReturnKind.TUPLE,
        leaves=(
            ReturnLeafSpec(slot=0, name=None, item_shape=STR),
            ReturnLeafSpec(slot=1, name=None, item_shape=BoolShape()),
            ReturnLeafSpec(slot=2, name=None, item_shape=list_shape),
        ),
        named_tuple_type=None,
    )
    outputs = (
        PhysicalOutputSpec(PortId(1), 0, STR, False),
        PhysicalOutputSpec(PortId(2), 1, BoolShape(), True),
        PhysicalOutputSpec(PortId(3), 2, list_shape, False),
    )
    columns, layouts = normalize_outputs(
        (
            ["a", "b"],
            [True, False],
            [["x"], ["y", "z"]],
        ),
        batch_size=2,
        return_schema=return_schema,
        output_schema=outputs,
    )
    assert columns == (["a", "b"], [True, False], ["x", "y", "z"])
    assert layouts[0].row_count == 2
    assert layouts[1].control_bits == b"\x01"
    assert layouts[2].offsets == (0, 1, 3)
    assert layouts[2].row_count == 3

    with pytest.raises(WorkerContractError, match="must be bool"):
        normalize_outputs(
            (["a", "b"], [1, 0], [["x"], []]),
            batch_size=2,
            return_schema=return_schema,
            output_schema=outputs,
        )


def test_layout_and_manifest_validation_reject_bad_offsets_padding_and_size():
    dispatch, _ = _identity()
    header = header_for(dispatch)
    with pytest.raises(ValueError, match="monotonic"):
        OutputLayout(
            return_slot=0,
            port=PortId(1),
            shape=StructuralListShape(STR),
            logical_count=2,
            row_count=1,
            offsets=(0, 1, 0),
            control_bits=None,
            estimated_bytes=None,
        )
    with pytest.raises(ValueError, match="padding"):
        OutputLayout(
            return_slot=0,
            port=PortId(1),
            shape=BoolShape(),
            logical_count=2,
            row_count=2,
            offsets=None,
            control_bits=b"\x81",
            estimated_bytes=None,
        )

    success = SuccessManifest(
        header=header,
        outputs=(
            OutputLayout(
                return_slot=0,
                port=PortId(1),
                shape=STR,
                logical_count=2,
                row_count=2,
                offsets=None,
                control_bits=None,
                estimated_bytes=None,
            ),
        ),
        worker_started_at=1.0,
        worker_finished_at=2.0,
        worker_rss_bytes=123,
    )
    assert validate_manifest(
        success,
        max_bytes=64 * 1024,
        expected_header=header,
        expected_output_count=1,
    ) > 0
    with pytest.raises(ProtocolValidationError, match="exceeds"):
        validate_manifest(success, max_bytes=1)


def test_manifest_layouts_match_compiled_physical_outputs_exactly():
    dispatch, _ = _identity()
    header = header_for(dispatch)
    bool_spec = (
        PhysicalOutputSpec(PortId(8), 0, BoolShape(), True),
    )

    missing_bits = SuccessManifest(
        header=header,
        outputs=(
            OutputLayout(
                return_slot=0,
                port=PortId(8),
                shape=BoolShape(),
                logical_count=2,
                row_count=2,
                offsets=None,
                control_bits=None,
                estimated_bytes=None,
            ),
        ),
        worker_started_at=1.0,
        worker_finished_at=2.0,
        worker_rss_bytes=None,
    )
    with pytest.raises(ProtocolValidationError, match="control_bits presence"):
        validate_manifest(
            missing_bits,
            max_bytes=64 * 1024,
            expected_header=header,
            expected_output_schema=bool_spec,
        )

    wrong_shape = (
        OutputLayout(
            return_slot=0,
            port=PortId(8),
            shape=STR,
            logical_count=2,
            row_count=2,
            offsets=None,
            control_bits=None,
            estimated_bytes=None,
        ),
    )
    with pytest.raises(ProtocolValidationError, match="shape"):
        validate_output_layouts(wrong_shape, bool_spec)
    with pytest.raises(ProtocolValidationError, match="port"):
        validate_output_layouts(
            wrong_shape,
            (PhysicalOutputSpec(PortId(9), 0, STR, False),),
        )
    with pytest.raises(ProtocolValidationError, match="return_slot"):
        validate_output_layouts(
            wrong_shape,
            (PhysicalOutputSpec(PortId(8), 1, STR, False),),
        )


def test_worker_instantiation_thaws_frozen_constructor_values():
    shared = ["shared"]
    args, kwargs = freeze_constructor_arguments(
        (
            {
                "nested": [1, {"flags": {"a", "b"}}],
                "shared": shared,
            },
        ),
        (("labels", shared),),
    )
    recipe = UdfRecipe(
        SerializableCallableRef(__name__, "ConstructorProbe"),
        args,
        kwargs,
        exception_atomic=True,
    )

    udf = instantiate_udf(recipe)
    assert isinstance(udf.config, dict)
    assert isinstance(udf.config["nested"], list)
    assert isinstance(udf.config["nested"][1]["flags"], set)
    assert udf.config["shared"] is udf.labels
    udf.labels.append("worker")
    assert thaw_config_value(dict(recipe.init_kwargs)) == {
        "labels": ["shared"]
    }
    assert recipe.exception_atomic is True


def test_failure_manifest_index_and_utf8_byte_limit():
    dispatch, _ = _identity()
    header = ManifestHeader(
        protocol_version=dispatch.protocol_version,
        graph_fingerprint=dispatch.graph_fingerprint,
        run=dispatch.run,
        dispatch=dispatch.dispatch,
        node=dispatch.node,
        lease=dispatch.lease,
        attempts=tuple(entry.token for entry in dispatch.entries),
    )
    failure = FailureManifest(
        header=header,
        kind=WorkerErrorKind.BAD_GRAIN,
        bad_entry_index=1,
        error_type="BadInput",
        message=truncate_utf8("甲乙丙", 7),
        trace_digest=b"digest",
    )
    assert failure.message == "甲乙"
    assert len(failure.message.encode("utf-8")) <= 7
    with pytest.raises(ValueError, match="outside"):
        FailureManifest(
            header=header,
            kind=WorkerErrorKind.BAD_GRAIN,
            bad_entry_index=2,
            error_type="BadInput",
            message="bad",
            trace_digest=None,
        )


def test_worker_context_cross_checks_return_and_physical_schema():
    dispatch, _ = _identity()
    call, returns, outputs = _single_schema()
    context = WorkerContext(
        protocol_version=PROTOCOL_VERSION,
        graph_fingerprint=dispatch.graph_fingerprint,
        node=dispatch.node,
        call_schema=call,
        return_schema=returns,
        output_schema=outputs,
        max_manifest_bytes=64 * 1024,
        max_error_message_bytes=1024,
    )
    assert context.output_schema == outputs

    with pytest.raises(ValueError, match="shapes differ"):
        WorkerContext(
            protocol_version=PROTOCOL_VERSION,
            graph_fingerprint=dispatch.graph_fingerprint,
            node=dispatch.node,
            call_schema=call,
            return_schema=returns,
            output_schema=(
                PhysicalOutputSpec(
                    port=PortId(8),
                    return_slot=0,
                    shape=BoolShape(),
                    emit_control_bits=False,
                ),
            ),
            max_manifest_bytes=64 * 1024,
            max_error_message_bytes=1024,
        )
