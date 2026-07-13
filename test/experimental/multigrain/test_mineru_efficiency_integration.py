"""Slow LPT/bubble evidence using real MinerU artifact work estimates."""
from __future__ import annotations

import time

import pytest

pytest.importorskip("PIL")

from rayorch.experimental import multigrain as mg
from rayorch.experimental.multigrain.graph import PhysicalHints
from rayorch.experimental.multigrain.ray_executor import (
    MultigrainRayExecutor,
    lpt_shard_planner,
)

from test.experimental.multigrain.mineru_integration_ops import (
    AssembleDoc,
    DocsToPages,
    PagesToBlocks,
    PowerLawImageFeature,
    load_artifact_docs,
    power_image_work,
)


pytestmark = [
    pytest.mark.slow,
    pytest.mark.mineru_integration,
    pytest.mark.usefixtures("ray_cluster"),
]

REPLICAS = 4


@pytest.fixture(scope="module")
def artifact_docs():
    docs = load_artifact_docs()
    if len(docs) < 2:
        pytest.skip("bounded MinerU regression artifacts are unavailable")
    return docs


def _blocks(docs):
    return [
        block
        for doc in docs
        for page in doc["pages"]
        for block in page["blocks"]
    ]


def _contiguous(total: int, replicas: int) -> list[list[int]]:
    replicas = min(total, replicas)
    width, extra = divmod(total, replicas)
    out = []
    start = 0
    for index in range(replicas):
        size = width + (1 if index < extra else 0)
        out.append(list(range(start, start + size)))
        start += size
    return out


def _loads(parts, values):
    return [sum(power_image_work(values[index]) for index in part) for part in parts]


class CompoundedFanoutPipe(mg.Pipeline):
    def __init__(self, sleep_scale: float = 0.03) -> None:
        super().__init__()
        self.to_pages = mg.Expand(DocsToPages, parent=0, child_label="page")
        self.to_blocks = mg.Expand(PagesToBlocks, parent=0, num_outputs=2)
        self.vision = mg.Map(
            PowerLawImageFeature,
            sleep_scale,
            name="power_vision",
            physical=PhysicalHints(replicas=REPLICAS),
        )
        self.assemble = mg.Reduce(AssembleDoc, name="assemble_power")

    def forward(self, docs):
        pages = self.to_pages(docs)
        blocks, _ = self.to_blocks(pages)
        features = self.vision(blocks)
        return self.assemble(mg.group_by(docs, features)), features


def test_real_artifact_lpt_loads_reduce_analytic_bubble(artifact_docs):
    values = _blocks(artifact_docs)
    batch = mg.source(values, name="blocks")
    contiguous = _contiguous(len(values), REPLICAS)
    lpt = lpt_shard_planner(power_image_work)(None, [batch], REPLICAS)
    contiguous_loads = _loads(contiguous, values)
    lpt_loads = _loads(lpt, values)

    def bubble(loads):
        return 1.0 - sum(loads) / (len(loads) * max(loads))

    # This is not a synthetic weight vector: service estimates come from the
    # dimensions/text lengths of the serialized MinerU artifact records.
    assert max(lpt_loads) < max(contiguous_loads) * 0.8
    assert bubble(lpt_loads) < bubble(contiguous_loads) * 0.4
    assert sorted(index for shard in lpt for index in shard) == list(range(len(values)))


def test_compounded_fanout_lpt_reduces_measured_bubble_without_semantic_drift(
    artifact_docs,
):
    graph = CompoundedFanoutPipe().compile()
    inputs = {
        "docs": mg.source(
            artifact_docs,
            name="docs",
            display_key=lambda doc: doc["name"],
        )
    }
    contiguous_metrics = mg.RunMetrics()
    lpt_metrics = mg.RunMetrics()
    contiguous = MultigrainRayExecutor(metrics=contiguous_metrics)
    balanced = MultigrainRayExecutor(
        shard_planner=lpt_shard_planner(power_image_work),
        metrics=lpt_metrics,
    )
    contiguous.warm_pools(graph)
    balanced.warm_pools(graph)
    try:
        # Warm the stateless Expand/Reduce worker path as well as actor pools;
        # otherwise whichever executor runs first pays process-import startup.
        contiguous.execute(graph, inputs)
        balanced.execute(graph, inputs)
        contiguous_metrics.nodes.clear()
        lpt_metrics.nodes.clear()

        started = time.perf_counter()
        contiguous_out = contiguous.execute(graph, inputs)
        contiguous_wall = time.perf_counter() - started

        started = time.perf_counter()
        lpt_out = balanced.execute(graph, inputs)
        lpt_wall = time.perf_counter() - started
    finally:
        contiguous.shutdown()
        balanced.shutdown()

    contiguous_summary, contiguous_features = contiguous_out
    lpt_summary, lpt_features = lpt_out
    assert lpt_summary.values == contiguous_summary.values
    assert lpt_summary.record_ids == contiguous_summary.record_ids
    assert {
        rid: (value, lineage)
        for rid, value, lineage in zip(
            lpt_features.record_ids,
            lpt_features.values,
            lpt_features.lineage,
        )
    } == {
        rid: (value, lineage)
        for rid, value, lineage in zip(
            contiguous_features.record_ids,
            contiguous_features.values,
            contiguous_features.lineage,
        )
    }

    contiguous_stage = contiguous_metrics.by_name("power_vision")
    lpt_stage = lpt_metrics.by_name("power_vision")
    assert contiguous_stage is not None and lpt_stage is not None
    print(
        {
            "contiguous_wall_s": round(contiguous_wall, 4),
            "lpt_wall_s": round(lpt_wall, 4),
            "contiguous_stage_s": round(contiguous_stage.stage_makespan_s, 4),
            "lpt_stage_s": round(lpt_stage.stage_makespan_s, 4),
            "contiguous_bubble": round(contiguous_stage.idle_bubble_frac, 4),
            "lpt_bubble": round(lpt_stage.idle_bubble_frac, 4),
        }
    )
    assert lpt_stage.idle_bubble_frac < contiguous_stage.idle_bubble_frac * 0.5
    assert lpt_stage.stage_makespan_s < contiguous_stage.stage_makespan_s * 0.85
    assert lpt_wall < contiguous_wall * 0.9
