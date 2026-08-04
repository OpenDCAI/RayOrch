"""source admission 与 multi-output report 的 hard-limit 原子性。"""

from __future__ import annotations

import pytest

import rayorch.experimental.multigrain_v3_3 as mg
from rayorch.experimental.multigrain_v3_3.protocol import (
    BlockRef,
    CallReport,
    OutputReport,
    RowBinding,
)
from rayorch.experimental.multigrain_v3_3.runtime import (
    ArenaEngine,
    ArenaLimits,
    CommitError,
)


class U:
    """手工提交测试的占位 kernel。"""


def row(block: int) -> RowBinding:
    """构造测试行引用。"""

    return RowBinding(BlockRef(block), 0)


class TwoSources(mg.Pipeline):
    """两个行对齐 sources，用于 admission 容量预检。"""

    def forward(self, left, right):
        """返回一个 source；另一个仍必须被原子接纳。"""

        del right
        return left


def test_source_item_limit_fails_before_entities_or_first_column_publish():
    program = TwoSources().compile().program
    arena = ArenaEngine(program, limits=ArenaLimits(max_items=1))

    with pytest.raises(CommitError, match="max_items"):
        arena.admit_sources(
            {
                program.source_ports[0]: (row(0),),
                program.source_ports[1]: (row(1),),
            }
        )

    assert arena.state.items == {}
    assert all(arena.entities(domain) == () for domain in program.domains)


class TwoOutputs(mg.Pipeline):
    """单 Grain 两输出，用于 report 原子预检。"""

    def __init__(self) -> None:
        self.call = mg.RayModule(U, num_outputs=2)

    def forward(self, values):
        """保留两个输出 Port。"""

        return self.call(values)


def test_multi_output_item_limit_exposes_neither_output():
    program = TwoOutputs().compile().program
    # source admission 消耗一个 Item，只剩一个 slot，不能容纳两个输出。
    arena = ArenaEngine(program, limits=ArenaLimits(max_items=2))
    arena.admit_sources({program.source_ports[0]: (row(0),)})
    grain = arena.reserve_ready()
    left, right = program.outputs_by_call[grain.call]

    with pytest.raises(CommitError, match="max_items"):
        arena.commit_success(
            CallReport(
                grain,
                0,
                (
                    OutputReport(left, scalar=row(1)),
                    OutputReport(right, scalar=row(2)),
                ),
            )
        )

    assert all(item.port not in {left, right} for item in arena.state.items)
    assert arena.state.grains[grain].active_attempt == 0
