from __future__ import annotations

import time

import pytest
import ray

from rayorch.experimental.multigrain_v3.model.graph import (
    CallSchema,
    OpaqueShape,
    ParameterKind,
    ParameterSpec,
    PhysicalOutputSpec,
    ReturnKind,
    ReturnLeafSpec,
    ReturnSchema,
    TypeRef,
)
from rayorch.experimental.multigrain_v3.model.semantics import (
    AttemptToken,
    BlockId,
    DispatchId,
    GrainId,
    GraphFingerprint,
    NodeId,
    PortId,
    RunId,
)
from rayorch.experimental.multigrain_v3.ray.protocol import (
    EMPTY_BLOCK,
    PROTOCOL_VERSION,
    FailureManifest,
    SlotTake,
    SuccessManifest,
    WorkerEntry,
    WorkerErrorKind,
)
from rayorch.experimental.multigrain_v3.ray.transport import (
    PINNED_RAY_VERSION,
    ActorPool,
    InfrastructureFailureKind,
    PendingFinalizeState,
    RayTransport,
)
from rayorch.experimental.multigrain_v3.ray.worker import (
    BadGrainError,
    WorkerContext,
)
from rayorch.experimental.multigrain_v3.runtime.dispatch import PreparedDispatch


pytestmark = pytest.mark.usefixtures("ray_cluster")


def _context(
    node: NodeId,
    shape: OpaqueShape | None = None,
) -> WorkerContext:
    shape = shape or OpaqueShape(TypeRef("builtins", "str"))
    return WorkerContext(
        protocol_version=PROTOCOL_VERSION,
        graph_fingerprint=GraphFingerprint.derive(
            "integration-graph",
            int(node),
        ),
        node=node,
        call_schema=CallSchema(
            parameters=(
                ParameterSpec(
                    name="values",
                    index=0,
                    kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                    item_shape=shape,
                ),
            ),
            positional_count=1,
            keyword_roles=(),
        ),
        return_schema=ReturnSchema(
            kind=ReturnKind.SINGLE,
            leaves=(ReturnLeafSpec(slot=0, name=None, item_shape=shape),),
            named_tuple_type=None,
        ),
        output_schema=(
            PhysicalOutputSpec(
                port=PortId(10 + int(node)),
                return_slot=0,
                shape=shape,
                emit_control_bits=False,
            ),
        ),
        max_manifest_bytes=64 * 1024,
        max_error_message_bytes=1024,
    )


def _prepared(node: NodeId, rows: int = 2) -> PreparedDispatch:
    run = RunId.new()
    attempts = tuple(
        AttemptToken(run=run, grain=GrainId.new(), generation=1)
        for _ in range(rows)
    )
    entries = tuple(
        WorkerEntry(token=token, role_trees=(SlotTake(0, row),))
        for row, token in enumerate(attempts)
    )
    return PreparedDispatch(
        id=DispatchId.new(),
        run=run,
        node=node,
        attempts=attempts,
        role_trees=tuple(() for _ in attempts),
        block_ids=(BlockId(0),),
        wire_entries=entries,
    )


def _one_completion(transport: RayTransport):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        completions = transport.poll(timeout=1)
        if completions:
            assert len(completions) == 1
            return completions[0]
    pytest.fail("Ray transport did not produce a completion")


def _finalize(transport: RayTransport, completion) -> None:
    completion.pending.semantic_applied = True
    transport.finalize_physical(completion.pending)
    assert (
        completion.pending.finalize_state
        is PendingFinalizeState.FINALIZED
    )


def test_ray_250_ordered_generator_hides_manifest_after_data_serialization_failure():
    class UnserializableYield:
        def __reduce__(self):
            raise RuntimeError("intentional data serialization failure")

    @ray.remote(max_retries=0)
    def ordered_generator_gate_spike():
        yield UnserializableYield()
        yield {"manifest": "must-not-be-observed"}

    assert ray.__version__.split("+", 1)[0] == PINNED_RAY_VERSION
    stream = ordered_generator_gate_spike.remote()
    failed_data_ref = next(stream)

    with pytest.raises(StopIteration):
        next(stream)
    with pytest.raises(ray.exceptions.RayTaskError, match="serialization failure"):
        ray.get(failed_data_ref)
    with pytest.raises(StopIteration):
        next(stream)


def test_transport_successfully_yields_output_then_validated_manifest():
    class EchoUdf:
        def run(self, values):
            return [f"echo:{value}" for value in values]

    node = NodeId(1)
    context = _context(node)
    pool = ActorPool(
        node,
        context,
        EchoUdf,
        ray_options={"num_cpus": 0},
    )
    released_inputs = []
    released_reservations = []
    transport = RayTransport(
        {node: pool},
        release_input=lambda dispatch, block: released_inputs.append(
            (dispatch, block)
        ),
        release_reservation=released_reservations.append,
    )
    try:
        prepared = _prepared(node)
        pending = transport.submit(
            prepared,
            (ray.put(["a", "b"]),),
        )
        assert pending is not None
        with pytest.raises(ValueError, match="already been submitted"):
            transport.submit(prepared, (ray.put(["a", "b"]),))
        completion = _one_completion(transport)
        assert isinstance(completion.manifest, SuccessManifest)
        assert ray.get(completion.pending.output_refs[0]) == [
            "echo:a",
            "echo:b",
        ]
        assert completion.manifest.outputs[0].logical_count == 2

        _finalize(transport, completion)
        transport.finalize_physical(completion.pending)
        assert released_inputs == [(prepared.id, BlockId(0))]
        assert released_reservations == [prepared.id]
        assert pool.slots[0].pending == 0
        assert pool.slots[0].active_lease is None
        assert prepared.id not in transport._submitted
    finally:
        transport.close()


def test_transport_never_observes_success_after_worker_data_serialization_failure():
    class UnserializableOutput:
        def __reduce__(self):
            raise RuntimeError("worker output serialization failure")

    class SerializationFailureUdf:
        def run(self, values):
            return [UnserializableOutput() for _ in values]

    node = NodeId(5)
    context = _context(node, OpaqueShape(None))
    pool = ActorPool(
        node,
        context,
        SerializationFailureUdf,
        ray_options={"num_cpus": 0},
    )
    transport = RayTransport({node: pool})
    try:
        pending = transport.submit(
            _prepared(node),
            (ray.put(["a", "b"]),),
        )
        assert pending is not None
        completion = _one_completion(transport)
        assert completion.manifest is None
        assert completion.failure is not None
        # Ray 2.50 actor generators collapse this failed yield into a short
        # stream whose completion ref succeeds; no serialization exception is
        # available to the driver, so RETURN_ARITY is the exact observable
        # classification.  Crucially, the success manifest is still hidden.
        assert (
            completion.failure.kind
            is InfrastructureFailureKind.RETURN_ARITY
        )
        assert completion.pending.manifest_ref is None
        assert pool.slots[0].incarnation == 1
        _finalize(transport, completion)
    finally:
        transport.close()


def test_controlled_bad_grain_failure_keeps_incarnation_and_discards_placeholders():
    class BadEntryUdf:
        def run(self, values):
            raise BadGrainError(1, f"bad value: {values[1]}")

    node = NodeId(2)
    context = _context(node)
    pool = ActorPool(
        node,
        context,
        BadEntryUdf,
        ray_options={"num_cpus": 0},
    )
    transport = RayTransport({node: pool})
    try:
        pending = transport.submit(
            _prepared(node),
            (ray.put(["ok", "bad"]),),
        )
        assert pending is not None
        completion = _one_completion(transport)
        manifest = completion.manifest
        assert isinstance(manifest, FailureManifest)
        assert manifest.kind is WorkerErrorKind.BAD_GRAIN
        assert manifest.bad_entry_index == 1
        assert ray.get(completion.pending.output_refs[0]) == EMPTY_BLOCK
        assert pool.slots[0].incarnation == 0

        _finalize(transport, completion)
        assert completion.pending.output_refs == []
    finally:
        transport.close()


def test_generic_failure_replaces_actor_and_late_release_cannot_touch_replacement():
    class ExplodingUdf:
        def run(self, values):
            raise RuntimeError(f"batch exploded: {values!r}")

    node = NodeId(3)
    context = _context(node)
    pool = ActorPool(
        node,
        context,
        ExplodingUdf,
        ray_options={"num_cpus": 0},
    )
    transport = RayTransport({node: pool})
    try:
        pending = transport.submit(
            _prepared(node),
            (ray.put(["a", "b"]),),
        )
        assert pending is not None
        old_handle = pending.actor_handle
        completion = _one_completion(transport)
        assert isinstance(completion.manifest, FailureManifest)
        assert completion.manifest.kind is WorkerErrorKind.GENERIC_UDF
        replacement = pool.slots[0]
        assert replacement.incarnation == 1
        assert replacement.handle is not old_handle
        assert replacement.pending == 0

        _finalize(transport, completion)
        assert pool.slots[0] is replacement
        assert replacement.pending == 0
        assert replacement.active_lease is None
    finally:
        transport.close()
        assert pool.slots == []


def test_hard_deadline_kills_exact_incarnation_and_classifies_failure():
    class SlowUdf:
        def run(self, values):
            time.sleep(2)
            return values

    node = NodeId(4)
    context = _context(node)
    pool = ActorPool(
        node,
        context,
        SlowUdf,
        ray_options={"num_cpus": 0},
    )
    transport = RayTransport({node: pool})
    try:
        pending = transport.submit(
            _prepared(node),
            (ray.put(["a", "b"]),),
            timeout_s=0,
        )
        assert pending is not None
        completion = _one_completion(transport)
        assert completion.manifest is None
        assert completion.failure is not None
        assert completion.failure.kind is InfrastructureFailureKind.DEADLINE
        assert pool.slots[0].incarnation == 1

        _finalize(transport, completion)
        assert pool.slots[0].pending == 0
    finally:
        transport.close()
