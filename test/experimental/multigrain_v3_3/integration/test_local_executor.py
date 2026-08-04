"""End-to-end local worker ABI and deterministic dummy regression."""

from __future__ import annotations

from rayorch.experimental.multigrain_v3_3.benchmark.dummy import run_dummy
from rayorch.experimental.multigrain_v3_3.executor import LocalBlockStore
from rayorch.experimental.multigrain_v3_3.model import (
    CallRef,
    DomainRef,
    EntityRef,
    GrainRef,
    MISSING,
    PortRef,
)
from rayorch.experimental.multigrain_v3_3.protocol import RowBinding
from rayorch.experimental.multigrain_v3_3.worker import (
    GroupTake,
    InvocationPlan,
    LocalWorker,
    MissingTake,
    OutputLayout,
    ScalarTake,
)


def test_worker_reconstructs_nested_groups_and_missing_without_program_access():
    class Echo:
        def run(self, scalars, groups, optional):
            assert optional == [MISSING]
            return [(scalars[0], groups[0])]

    store = LocalBlockStore()
    scalar_block = store.put(("root",))
    leaf_block = store.put((1, 2, 3))
    grain = GrainRef(CallRef(0), EntityRef(DomainRef(0), 0))
    invocation = InvocationPlan(
        grain,
        0,
        (
            ScalarTake(RowBinding(scalar_block, 0)),
            GroupTake(
                tuple(RowBinding(leaf_block, row) for row in range(3)),
                ((0, 2), (0, 1, 3)),
            ),
            MissingTake(),
        ),
    )
    report = LocalWorker(Echo).execute(
        (invocation,),
        (OutputLayout(PortRef(0)),),
        store,
    )[0]
    binding = report.outputs[0].scalar
    assert binding is not None
    assert store.get(binding) == ("root", [[1], [2, 3]])


def test_dummy_elastic_and_parent_bound_have_exact_output_parity():
    elastic = run_dummy(96, mode="elastic", batch_size=16, rounds=2)
    parent = run_dummy(96, mode="parent_bound", batch_size=16, rounds=2)

    assert elastic.outputs == parent.outputs
    assert elastic.arena.is_complete()
    assert parent.arena.is_complete()

    # CallRef(2) is the heavy TransformPages Call in DummyPipeline.
    heavy = CallRef(2)
    assert elastic.calls[heavy].grains == parent.calls[heavy].grains
    assert elastic.calls[heavy].rpcs < parent.calls[heavy].rpcs
    assert elastic.calls[heavy].average_batch > parent.calls[heavy].average_batch


def test_dummy_structural_relations_do_not_create_execution_pools():
    result = run_dummy(16, mode="elastic", batch_size=8, rounds=1)
    assert len(result.calls) == 4
    assert result.rpc_count == sum(call.rpcs for call in result.calls.values())
