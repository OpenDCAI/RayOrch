"""M2 instrumentation (#3) + fault-injection/recovery (#5) on dummy data.

Pure-code, CPU-only proof that the executor now emits the headline M2 numbers and
recovers locally, using auto-generated tabular/image sources (no real data):

* **idle bubble** -- ``RunMetrics`` reports per-stage load-imbalance; LPT shrinks
  the SleepMap stage's bubble below contiguous while keeping output identical;
* **lineage footprint** -- ``lineage_footprint`` reports records/bytes overhead;
* **recovery** -- an injected task crash on one shard is retried and recomputes
  ONLY that shard's rows (lineage-local), and the output matches the fault-free
  run.

Marked ``slow`` + shared ``ray_cluster`` fixture (CPU only, no GPU).
"""
from __future__ import annotations

import pytest

from rayorch.experimental import multigrain as mg
from rayorch.experimental.multigrain.ir import WorkerPoolSpec
from rayorch.experimental.multigrain.execution import RunMetrics, lineage_footprint
from rayorch.experimental.multigrain.ray import (
    FaultSpec,
    MultigrainRayExecutor,
    lpt_shard_planner,
)

from test.experimental.multigrain.metrics_ops import (
    Assemble,
    MakeRows,
    SleepMap,
    make_image_docs,
    make_tabular_docs,
    row_work,
)

pytestmark = [pytest.mark.slow, pytest.mark.usefixtures("ray_cluster")]

REPLICAS = 4

# Deterministic long-tail fan-out with no single dominating row (so LPT can
# actually balance): 21 rows, total work 91, max row 9 << 91/4.
_BUBBLE_DOCS = [
    {"name": "d0", "works": [5, 1, 9, 2, 7]},
    {"name": "d1", "works": [3, 8]},
    {"name": "d2", "works": [2, 6, 4, 1]},
    {"name": "d3", "works": [9, 1]},
    {"name": "d4", "works": [4, 2, 6, 8, 3, 1]},
    {"name": "d5", "works": [7, 2]},
]


def _pipe(replicas: int, *, max_shard_retries: int = 2) -> mg.Pipeline:
    class Pipe(mg.Pipeline):
        def __init__(self) -> None:
            super().__init__()
            self.split = mg.Expand(MakeRows, parent=0, child_label="row")
            self.embed = mg.Map(
                SleepMap,
                workers=WorkerPoolSpec(replicas=replicas),
                recovery=mg.RecoveryPolicy(
                    max_shard_retries=max_shard_retries
                ),
            )
            self.assemble = mg.Reduce(Assemble)

        def forward(self, docs):
            rows = self.split(docs)
            embeds = self.embed(rows)
            return self.assemble(mg.group_by(docs, embeds))

    return Pipe()


def _docs_batch(docs):
    return mg.source(docs, name="docs", display_key=lambda d: d["name"])


# --------------------------------------------------------------------------
# lineage footprint accounting
# --------------------------------------------------------------------------
def test_lineage_footprint_reports_records_and_bytes() -> None:
    docs = _docs_batch(_BUBBLE_DOCS)
    pages = _pipe(1).split(docs)  # the fanned-out rows carry the lineage

    fp = lineage_footprint([pages])
    assert fp["records"] == len(pages) == 21
    assert fp["ancestor_entries"] >= fp["records"]  # each row knows >=1 ancestor
    assert fp["bytes_per_record"] > 0
    assert fp["approx_bytes"] > fp["records"]


# --------------------------------------------------------------------------
# idle bubble: measured, and LPT shrinks it without changing output
# --------------------------------------------------------------------------
def test_lpt_shrinks_measured_bubble_and_preserves_output() -> None:
    docs = _docs_batch(_BUBBLE_DOCS)

    cont_metrics = RunMetrics()
    cont = MultigrainRayExecutor(metrics=cont_metrics).execute(
        _pipe(REPLICAS).compile(), {"docs": docs}
    )

    lpt_metrics = RunMetrics()
    lpt = MultigrainRayExecutor(
        shard_planner=lpt_shard_planner(row_work), metrics=lpt_metrics
    ).execute(_pipe(REPLICAS).compile(), {"docs": docs})

    # correctness: reordering + rebalancing does not change the result
    assert lpt.values == cont.values
    assert lpt.record_ids == cont.record_ids == docs.record_ids

    cont_stage = cont_metrics.by_name("SleepMap")
    lpt_stage = lpt_metrics.by_name("SleepMap")
    assert cont_stage is not None and lpt_stage is not None
    assert len(cont_stage.shard_busy_s) == REPLICAS
    assert len(lpt_stage.shard_busy_s) == REPLICAS

    # contiguous leaves a real bubble; LPT drives it down (and never worse)
    assert cont_stage.idle_bubble_frac > 0.1
    assert lpt_stage.idle_bubble_frac < cont_stage.idle_bubble_frac
    assert lpt_stage.stage_makespan_s < cont_stage.stage_makespan_s


def test_metrics_populated_for_every_stage() -> None:
    docs = _docs_batch(make_image_docs())
    metrics = RunMetrics()
    MultigrainRayExecutor(metrics=metrics).execute(_pipe(REPLICAS).compile(), {"docs": docs})

    names = {node.name for node in metrics.nodes}
    assert {"MakeRows", "SleepMap", "Assemble"} <= names
    assert metrics.total_wall_s > 0.0
    split = metrics.by_name("MakeRows")
    mapped = metrics.by_name("SleepMap")
    assert split is not None and mapped is not None
    assert split.fanout_ratio > 1.0
    assert sum(mapped.shard_rows_in) == mapped.rows_in
    assert sum(mapped.shard_rows_out) == mapped.rows_out
    assert mapped.lineage_bytes_out > 0
    assert {
        "shard_rows_in",
        "shard_rows_out",
        "fanout_ratio",
        "relation_entries_out",
        "lineage_bytes_out",
    } <= mapped.to_dict().keys()


# --------------------------------------------------------------------------
# fault injection + lineage-local recovery
# --------------------------------------------------------------------------
def test_injected_task_crash_is_retried_and_recovers_only_failed_shard() -> None:
    docs = _docs_batch(make_tabular_docs())

    healthy = MultigrainRayExecutor(
        shard_planner=lpt_shard_planner(row_work)
    ).execute(_pipe(REPLICAS).compile(), {"docs": docs})

    metrics = RunMetrics()
    recovered = MultigrainRayExecutor(
        shard_planner=lpt_shard_planner(row_work),
        metrics=metrics,
        faults=[FaultSpec(node="SleepMap", fail_shards=frozenset({0}))],
    ).execute(_pipe(REPLICAS).compile(), {"docs": docs})

    # crash was transparently recovered: identical output to the fault-free run
    assert recovered.values == healthy.values

    stage = metrics.by_name("SleepMap")
    assert stage is not None
    assert stage.retries == 1  # exactly one shard failed once, then succeeded
    # recovery recomputed ONLY the failed shard's rows, not the whole stage
    assert 0 < stage.recovery_rows < stage.rows_in


def test_fault_exhausting_retries_propagates() -> None:
    docs = _docs_batch(make_tabular_docs())
    with pytest.raises(Exception):
        MultigrainRayExecutor(
            faults=[
                FaultSpec(
                    node="SleepMap",
                    fail_shards=frozenset({0}),
                    fail_until_attempt=99,
                )
            ],
        ).execute(
            _pipe(REPLICAS, max_shard_retries=1).compile(),
            {"docs": docs},
        )
