"""设计文档 21.2/21.3 的 Port、层级、失败与 limit 边界矩阵。"""

from __future__ import annotations

import pytest

import rayorch.experimental.multigrain_v3_4 as mg
from rayorch.experimental.multigrain_v3_4.executor import Executor
from rayorch.experimental.multigrain_v3_4.model import (
    GrainOutcome,
    ItemOutcome,
    ItemRef,
)
from rayorch.experimental.multigrain_v3_4.program import ExpandOrigin
from rayorch.experimental.multigrain_v3_4.protocol import (
    BlockRef,
    CallReport,
    ExpandedRows,
    OutputReport,
    RowBinding,
)
from rayorch.experimental.multigrain_v3_4.runtime import (
    ArenaEngine,
    ArenaLimits,
    CommitError,
)


class U:
    """手工驱动 Arena 时使用的占位 kernel。"""


def row(block: int, index: int = 0) -> RowBinding:
    """构造不携带真实 payload 的确定性测试行。"""

    return RowBinding(BlockRef(block), index)


class Fanout:
    """为每个输入产生固定数量 child 的列式 UDF。"""

    def __init__(self, width: int) -> None:
        self.width = width

    def run(self, values):
        """把 ordinal path 编码进值，便于检查最终顺序。"""

        return [
            [(value, ordinal) for ordinal in range(self.width)]
            for value in values
        ]


class Echo:
    """保持列式输入不变。"""

    def run(self, values):
        """返回输入列副本。"""

        return list(values)


class Pair:
    """把 root value 与层级 group 组合为可比较输出。"""

    def run(self, roots, groups):
        """按 Grain 对齐组合两个输入列。"""

        return list(zip(roots, groups))


class DeepPipeline(mg.Pipeline):
    """参数化的多层 Expand/Reduce conformance Pipeline。"""

    def __init__(self, depth: int, width: int) -> None:
        self.fanouts = [
            mg.RayModule(Fanout).pre_init(width).ray_options(batch_size=64, num_cpus=0)
            for _ in range(depth)
        ]
        self.sink = mg.RayModule(Pair).ray_options(batch_size=8, num_cpus=0)

    def forward(self, roots):
        """先连续 fan-out，再严格逐级 reduce 回 root Domain。"""

        value = roots
        for fanout in self.fanouts:
            value = mg.functional.expand(fanout(value))
        for _ in self.fanouts:
            value = mg.functional.reduce(value)
        return self.sink(roots, value)


def _nested_leaf_count(value) -> int:
    """递归统计任意 canonical group 中的叶子数。"""

    if not isinstance(value, list):
        return 1
    return sum(_nested_leaf_count(child) for child in value)


def _nested_depth(value) -> int:
    """统计一条 canonical 嵌套 group 的深度。"""

    if not isinstance(value, list):
        return 0
    return 1 + (_nested_depth(value[0]) if value else 0)


def test_twenty_unary_levels_preserve_shape_depth_without_identity_shortcut():
    with Executor(DeepPipeline(depth=20, width=1)) as executor:
        result = executor.run([7])
    root, group = result.outputs[0]

    assert root == 7
    assert _nested_depth(group) == 20
    assert _nested_leaf_count(group) == 1
    deepest_domain = max(result.arenas[0].program.domains)
    deepest = result.arenas[0].entities(deepest_domain)
    assert len(deepest) == 1
    assert len(result.arenas[0].entity_coordinate(deepest[0])) == 21


def test_five_binary_levels_preserve_all_32_ordered_leaves():
    with Executor(DeepPipeline(depth=5, width=2)) as executor:
        result = executor.run([3])
    _, group = result.outputs[0]

    assert _nested_depth(group) == 5
    assert _nested_leaf_count(group) == 32


class AllFilteredPipeline(mg.Pipeline):
    """同一个空 group 被两个计算分支消费。"""

    def __init__(self) -> None:
        self.fanout = mg.RayModule(Fanout).pre_init(4).ray_options(num_cpus=0)
        self.false = mg.RayModule(lambda values: [False for _ in values]).ray_options(num_cpus=0)
        self.left = mg.RayModule(Echo).ray_options(num_cpus=0)
        self.right = mg.RayModule(Echo).ray_options(num_cpus=0)

    def forward(self, roots):
        """Filter 全部成员后，对同一 Reduce Port 形成两个消费者。"""

        children = mg.functional.expand(self.fanout(roots))
        selected = mg.functional.filter(children, self.false(children))
        groups = mg.functional.reduce(selected)
        return self.left(groups), self.right(groups)


def test_all_filtered_is_present_empty_group_for_multiple_consumers():
    with Executor(AllFilteredPipeline()) as executor:
        result = executor.run([1, 2])

    assert result.outputs == ([[], []], [[], []])
    outcomes = [
        result.arenas[0].item_outcome(item)
        for port in result.arenas[0].program.output_tree
        for item in result.arenas[0].ordered_items(port)
    ]
    assert outcomes and all(outcome is ItemOutcome.PRESENT for outcome in outcomes)


class BranchPipeline(mg.Pipeline):
    """两个独立 Call branch，用于验证失败隔离。"""

    def __init__(self) -> None:
        self.left = mg.RayModule(U)
        self.right = mg.RayModule(U)

    def forward(self, roots):
        """返回两个互不依赖的 root-domain branch。"""

        return self.left(roots), self.right(roots)


def test_branch_failure_does_not_suppress_unrelated_branch():
    program = BranchPipeline().compile().program
    arena = ArenaEngine(program)
    root = arena.admit_sources({program.source_ports[0]: (row(0),)})[0]
    left = arena.reserve_ready()
    right = arena.reserve_ready()

    arena.commit_failure(left, "left failed")
    right_output = program.outputs_by_call[right.call][0]
    arena.commit_success(
        CallReport(right, 0, (OutputReport(right_output, scalar=row(2)),))
    )

    left_output = program.outputs_by_call[left.call][0]
    assert arena.item_outcome(ItemRef(left_output, root)) is ItemOutcome.FAILED
    assert arena.item_outcome(ItemRef(right_output, root)) is ItemOutcome.PRESENT


class OptionalPipeline(mg.Pipeline):
    """Filter 输出作为 optional 非 driving 输入。"""

    def __init__(self) -> None:
        self.render = mg.RayModule(U)
        self.mask = mg.RayModule(U)
        self.merge = mg.RayModule(U).driven_by(0)

    def forward(self, roots):
        """aux DROPPED 可执行，aux FAILED 必须 suppress。"""

        children = mg.functional.expand(self.render(roots))
        selected = mg.functional.filter(children, self.mask(children))
        return self.merge(children, mg.functional.optional(selected))


def _optional_case(mask_failure: bool):
    """手工运行一个 child，返回 merge Grain 及 Arena。"""

    program = OptionalPipeline().compile().program
    arena = ArenaEngine(program)
    arena.admit_sources({program.source_ports[0]: (row(0),)})
    render = arena.reserve_ready()
    render_output = program.outputs_by_call[render.call][0]
    child_port = next(
        port for port, spec in program.ports.items()
        if isinstance(spec.origin, ExpandOrigin)
    )
    arena.commit_success(
        CallReport(
            render,
            0,
            (
                OutputReport(
                    render_output,
                    expansions=(ExpandedRows(child_port, (row(1),)),),
                ),
            ),
        )
    )
    mask = arena.reserve_ready()
    if mask_failure:
        arena.commit_failure(mask, "mask failed")
    else:
        mask_output = program.outputs_by_call[mask.call][0]
        arena.commit_success(
            CallReport(
                mask,
                0,
                (OutputReport(mask_output, scalar=row(2), control=False),),
            )
        )
    merge_call = max(program.calls)
    merge_record = next(
        record for grain, record in arena.state.grains.items()
        if grain.call == merge_call
    )
    return arena, merge_record


def test_optional_dropped_is_missing_but_optional_failed_suppresses():
    dropped_arena, dropped = _optional_case(mask_failure=False)
    failed_arena, failed = _optional_case(mask_failure=True)

    assert dropped.outcome is None
    assert dropped_arena.invocation_plan(dropped.ref).inputs[1].__class__.__name__ == "MissingTake"
    assert failed.outcome is GrainOutcome.SUPPRESSED
    assert failed_arena.ready_count == 0


def test_hard_entity_limit_rejects_expand_without_partial_publication():
    class ExpandOnly(mg.Pipeline):
        """用于 hard limit 原子性测试的单 fan-out Pipeline。"""

        def __init__(self) -> None:
            self.render = mg.RayModule(U)

        def forward(self, roots):
            """返回 child Port。"""

            return mg.functional.expand(self.render(roots))

    program = ExpandOnly().compile().program
    arena = ArenaEngine(program, limits=ArenaLimits(max_entities=2))
    root = arena.admit_sources({program.source_ports[0]: (row(0),)})[0]
    grain = arena.reserve_ready()
    output = program.outputs_by_call[grain.call][0]
    child_port = next(
        port for port, spec in program.ports.items()
        if isinstance(spec.origin, ExpandOrigin)
    )

    with pytest.raises(CommitError, match="max_entities"):
        arena.commit_success(
            CallReport(
                grain,
                0,
                (
                    OutputReport(
                        output,
                        expansions=(
                            ExpandedRows(child_port, (row(1, 0), row(1, 1), row(1, 2))),
                        ),
                    ),
                ),
            )
        )

    assert arena.entities(program.port(child_port).domain) == ()
    assert ItemRef(output, root) not in arena.state.items
    assert arena.state.shapes == {}


def test_hard_entity_limit_counts_root_and_child_entities_together():
    class ExpandOne(mg.Pipeline):
        def __init__(self) -> None:
            self.render = mg.RayModule(U)

        def forward(self, roots):
            return mg.functional.expand(self.render(roots))

    program = ExpandOne().compile().program
    arena = ArenaEngine(program, limits=ArenaLimits(max_entities=1))
    root = arena.admit_sources({program.source_ports[0]: (row(60),)})[0]
    grain = arena.reserve_ready()
    output = program.outputs_by_call[grain.call][0]
    child_port = next(
        port
        for port, spec in program.ports.items()
        if isinstance(spec.origin, ExpandOrigin)
    )

    with pytest.raises(CommitError, match="max_entities"):
        arena.commit_success(
            CallReport(
                grain,
                0,
                (
                    OutputReport(
                        output,
                        expansions=(ExpandedRows(child_port, (row(61),)),),
                    ),
                ),
            )
        )

    assert arena.entity_count == 1
    assert ItemRef(output, root) not in arena.state.items
