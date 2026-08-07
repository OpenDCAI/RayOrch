"""Worker ABI 与 Ray dummy 的确定性回归。"""

from __future__ import annotations

from rayorch.experimental.multigrain_v3_4.benchmark.dummy import run_dummy
from rayorch.experimental.multigrain_v3_4.model import (
    CallRef,
    DomainRef,
    EntityRef,
    GrainRef,
    MISSING,
    PortRef,
)
from rayorch.experimental.multigrain_v3_4.protocol import (
    BlockRef,
    GroupTake,
    InvocationPlan,
    MissingTake,
    OutputLayout,
    RowBinding,
    ScalarTake,
)
from rayorch.experimental.multigrain_v3_4.worker import Worker


class MemoryBlockStore:
    """仅用于 Worker ABI 单测的进程内块存储，不参与运行时架构。"""

    def __init__(self) -> None:
        self._next = 0
        self.blocks: dict[BlockRef, tuple[object, ...]] = {}

    def put(self, values: tuple[object, ...]) -> BlockRef:
        """保存不可变值块并返回测试引用。"""

        ref = BlockRef(self._next)
        self._next += 1
        self.blocks[ref] = tuple(values)
        return ref

    def get(self, binding: RowBinding) -> object:
        """读取测试块中的指定行。"""

        return self.blocks[binding.block][binding.row]


def test_worker_reconstructs_nested_groups_and_missing_without_program_access():
    class Echo:
        def run(self, scalars, groups, optional):
            assert optional == [MISSING]
            return [(scalars[0], groups[0])]

    store = MemoryBlockStore()
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
    report = Worker(Echo).execute(
        (invocation,),
        (OutputLayout(PortRef(0)),),
        store,
    )[0]
    binding = report.outputs[0].scalar
    assert binding is not None
    assert store.get(binding) == ("root", [[1], [2, 3]])


def test_worker_emits_row_aligned_controls_for_expanded_masks():
    class Masks:
        def run(self, values):
            return [[value > 0, value < 0] for value in values]

    store = MemoryBlockStore()
    input_block = store.put((3,))
    parent = PortRef(0)
    child = PortRef(1)
    grain = GrainRef(CallRef(0), EntityRef(DomainRef(0), 0))
    report = Worker(Masks).execute(
        (
            InvocationPlan(
                grain,
                0,
                (ScalarTake(RowBinding(input_block, 0)),),
            ),
        ),
        (OutputLayout(parent, (child,), frozenset({parent, child})),),
        store,
    )[0]

    expanded = report.outputs[0].expansions[0]
    assert expanded.port == child
    assert expanded.controls == (True, False)
    assert tuple(store.get(binding) for binding in expanded.rows) == (True, False)


def test_dummy_elastic_and_parent_bound_have_exact_output_parity():
    elastic = run_dummy(96, mode="elastic", batch_size=16, rounds=2)
    parent = run_dummy(96, mode="parent_bound", batch_size=16, rounds=2)

    assert elastic.outputs == parent.outputs
    assert elastic.arenas[0].is_complete()
    assert parent.arenas[0].is_complete()

    # CallRef(2) is the heavy TransformPages Call in DummyPipeline.
    heavy = CallRef(2)
    assert elastic.calls[heavy].grains == parent.calls[heavy].grains
    assert elastic.calls[heavy].rpcs < parent.calls[heavy].rpcs
    assert elastic.calls[heavy].average_batch > parent.calls[heavy].average_batch


def test_dummy_structural_relations_do_not_create_execution_pools():
    result = run_dummy(16, mode="elastic", batch_size=8, rounds=1)
    assert len(result.calls) == 4
    assert result.rpc_count == sum(call.rpcs for call in result.calls.values())
