"""1,000-parent 性能消融与二层动态 fan-out 回归。"""

from __future__ import annotations

from rayorch.experimental.multigrain_v3_3.benchmark.dummy_matrix import (
    NestedDummyPipeline,
    run_local_matrix,
)
from rayorch.experimental.multigrain_v3_3.executor import LocalExecutor


def test_local_1000_parent_batch_cap_matrix_has_exact_parity_and_fewer_rpcs():
    records = run_local_matrix(parents=1_000, rounds=2)

    assert len(records) == 6
    for batch_size in (16, 32, 64):
        parent, elastic = [
            record for record in records if record.batch_size == batch_size
        ]
        assert parent.mode == "parent_bound"
        assert elastic.mode == "elastic"
        assert parent.output_digest == elastic.output_digest
        assert elastic.heavy_rpcs < parent.heavy_rpcs
        assert elastic.heavy_average_batch > parent.heavy_average_batch
        assert parent.parents == elastic.parents == 1_000


def test_nested_dummy_preserves_empty_intermediate_groups_and_order():
    result = LocalExecutor(
        NestedDummyPipeline(batch_size=16, rounds=2)
    ).run(range(64))

    assert len(result.outputs) == 64
    assert [document for document, _ in result.outputs] == list(range(64))
    assert result.outputs[0] == (0, [])  # document 0 的 Page fan-out 为 N=0。
    assert result.arena.is_complete()
    assert len(result.arena.program.domains) == 3
