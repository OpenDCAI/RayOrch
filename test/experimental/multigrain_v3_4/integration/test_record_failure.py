"""batch 内逐 Grain 失败、多输出原子性与 parent 隔离回归。"""

from __future__ import annotations

import rayorch.experimental.multigrain_v3_4 as mg
from rayorch.experimental.multigrain_v3_4.executor import Executor
from rayorch.experimental.multigrain_v3_4.model import ItemOutcome
from rayorch.experimental.multigrain_v3_4.protocol import RecordFailure


class Render:
    """每个 parent 产生三个有序 leaf。"""

    def run(self, parents):
        """构造确定性的 parent/ordinal 值。"""

        return [[(parent, ordinal) for ordinal in range(3)] for parent in parents]


class FailOneLeaf:
    """只把 parent=1、ordinal=1 标为业务失败。"""

    def run(self, leaves):
        """同一 batch 中同时返回成功值与 RecordFailure。"""

        return [
            RecordFailure(f"bad leaf {leaf}") if leaf == (1, 1) else leaf
            for leaf in leaves
        ]


class Assemble:
    """把成功 leaf group 与 parent 对齐。"""

    def run(self, parents, groups):
        """生成便于比较的最终值。"""

        return list(zip(parents, groups))


class BadLeafPipeline(mg.Pipeline):
    """真实 Expand→bad leaf→Reduce→Assemble 失败传播拓扑。"""

    def __init__(self) -> None:
        self.render = mg.RayModule(Render).ray_options(batch_size=8, num_cpus=0)
        self.transform = mg.RayModule(FailOneLeaf).ray_options(batch_size=16, num_cpus=0)
        self.assemble = mg.RayModule(Assemble).ray_options(batch_size=8, num_cpus=0)

    def forward(self, parents):
        """失败 leaf 只 suppress 所属 parent 的 group/assemble Grain。"""

        leaves = mg.functional.expand(self.render(parents))
        values = self.transform(leaves)
        groups = mg.functional.reduce(values)
        return self.assemble(parents, groups)


def test_bad_leaf_suppresses_only_its_parent_inside_one_batch():
    with Executor(BadLeafPipeline()) as executor:
        result = executor.run([0, 1, 2])

    assert result.outputs[0] == (0, [(0, 0), (0, 1), (0, 2)])
    assert result.outputs[1] is ItemOutcome.SUPPRESSED
    assert result.outputs[2] == (2, [(2, 0), (2, 1), (2, 2)])

    transform_metrics = result.calls[next(
        call for call, spec in result.arenas[0].program.calls.items()
        if spec.kernel.target is FailOneLeaf
    )]
    assert transform_metrics.rpcs == 1
    assert transform_metrics.grains == 9


class MultiOutputFail:
    """一个输出成功、另一个失败，用于验证整 Grain 原子失败。"""

    def run(self, values):
        """第二条 Grain 在第二输出列标记失败。"""

        return (
            [value * 10 for value in values],
            [
                RecordFailure("second output failed") if value == 2 else value * 100
                for value in values
            ],
        )


class MultiOutputPipeline(mg.Pipeline):
    """返回两个独立 Port 的 multi-output Call。"""

    def __init__(self) -> None:
        self.call = mg.RayModule(MultiOutputFail, num_outputs=2).ray_options(
            batch_size=8
        )

    def forward(self, values):
        """保留两个输出 Port，检查同一 Grain 不会部分可见。"""

        return self.call(values)


def test_record_failure_makes_all_outputs_of_one_grain_failed():
    with Executor(MultiOutputPipeline()) as executor:
        left, right = executor.run([1, 2, 3]).outputs

    assert left == [10, ItemOutcome.FAILED, 30]
    assert right == [100, ItemOutcome.FAILED, 300]
