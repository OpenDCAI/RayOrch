"""Lineage correctness under Ray parallelism + work-aware cross-shard reordering.

The optimization we care about (LPT rebalancing of an imbalanced 1:N fan-out)
*permutes rows across shards*. This suite proves that permutation does not
corrupt the lineage machinery, by treating the single-task local executor as the
ground truth and asserting the parallel + reordered run is byte-for-byte equal:

* identity/ordinal:  document -> Expand -> Map -> Reduce reassembles every
  document's pages in their original order even though LPT scattered them;
* fault isolation:   a single bad page is quarantined with the *same*
  ``ErrorTrace`` (source item, logical item, upstream op path) regardless of
  which shard/worker it happened to land on, and healthy pages survive.

Marked ``slow`` + uses the shared ``ray_cluster`` fixture (CPU only, no GPU).
"""
from __future__ import annotations

import pytest

from rayorch.experimental import multigrain as mg
from rayorch.experimental.multigrain.ir import WorkerPoolSpec
from rayorch.experimental.multigrain.ray import MultigrainRayExecutor, lpt_shard_planner
from rayorch.experimental.multigrain.ray.executor import _contiguous_ranges

from test.experimental.multigrain.lineage_ops import (
    BOOM_DOC,
    BOOM_PAGE,
    AssembleDoc,
    EmbedPage,
    EmbedPageOk,
    SplitPages,
    page_work,
)

pytestmark = [pytest.mark.slow, pytest.mark.usefixtures("ray_cluster")]

REPLICAS = 4


def _make_docs() -> list[dict]:
    # varied page counts AND varied per-page work so LPT genuinely reorders rows
    specs = [
        ("d00", [1]),
        ("d01", [5, 1, 9, 2, 7]),
        ("d02", [3, 8]),
        ("d03", [2, 6, 4, 1]),  # page BOOM_PAGE (=2) is the fault-injection target
        ("d04", [9, 9, 1]),
        ("d05", [1]),
        ("d06", [4, 2, 6, 8, 3, 1]),
        ("d07", [7, 2]),
    ]
    return [{"name": name, "page_works": works} for name, works in specs]


def _docs_batch():
    return mg.source(_make_docs(), name="docs", display_key=lambda doc: doc["name"])


class Pipe(mg.Pipeline):
    def __init__(self, embed_cls, replicas: int) -> None:
        super().__init__()
        self.split = mg.Expand(SplitPages, parent=0, child_label="page")
        self.embed = mg.Map(embed_cls, workers=WorkerPoolSpec(replicas=replicas))
        self.assemble = mg.Reduce(AssembleDoc)

    def forward(self, docs):
        pages = self.split(docs)
        embeds = self.embed(pages)
        return self.assemble(mg.group_by(docs, embeds))


def _pages_batch():
    return Pipe(EmbedPageOk, 1).split(_docs_batch())


# --------------------------------------------------------------------------
# Precondition: LPT actually reorders rows across shards (otherwise the test
# below would pass trivially and prove nothing).
# --------------------------------------------------------------------------
def test_lpt_reorders_rows_across_shards() -> None:
    pages = _pages_batch()
    contiguous = [list(rng) for rng in _contiguous_ranges(len(pages), REPLICAS)]
    lpt = lpt_shard_planner(page_work)(None, [pages], REPLICAS)

    assert {frozenset(s) for s in lpt} != {frozenset(s) for s in contiguous}
    # at least one row ends up on a different shard than contiguous would place it
    cont_of = {i: s for s, part in enumerate(contiguous) for i in part}
    lpt_of = {i: s for s, part in enumerate(lpt) for i in part}
    assert any(cont_of[i] != lpt_of[i] for i in range(len(pages)))


# --------------------------------------------------------------------------
# Identity + ordinal invariance under parallel + reordered execution
# --------------------------------------------------------------------------
def test_reordered_parallel_run_matches_local_identity_and_order() -> None:
    docs = _docs_batch()

    local = mg.MultigrainExecutor().execute(Pipe(EmbedPageOk, 1).compile(), {"docs": docs})
    parallel = MultigrainRayExecutor(
        shard_planner=lpt_shard_planner(page_work)
    ).execute(Pipe(EmbedPageOk, REPLICAS).compile(), {"docs": docs})

    # anchor grain: one row per document, in document order
    assert parallel.record_ids == local.record_ids == docs.record_ids
    # Reduce restored per-document page order despite LPT scattering the pages
    assert parallel.values == local.values
    d06 = next(v for v in parallel.values if v.startswith("d06="))
    assert d06 == "d06=[emb(d06#p0)|emb(d06#p1)|emb(d06#p2)|emb(d06#p3)|emb(d06#p4)|emb(d06#p5)]"


# --------------------------------------------------------------------------
# Fault isolation invariance: same quarantine trace regardless of sharding
# --------------------------------------------------------------------------
def test_quarantine_localizes_same_page_under_reordered_parallelism() -> None:
    docs = _docs_batch()

    local = mg.MultigrainExecutor().execute(Pipe(EmbedPage, 1).compile(), {"docs": docs})
    parallel = MultigrainRayExecutor(
        shard_planner=lpt_shard_planner(page_work)
    ).execute(Pipe(EmbedPage, REPLICAS).compile(), {"docs": docs})

    # healthy documents identical; the bad document survives minus its bad page
    assert parallel.values == local.values

    assert len(local.errors) == 1
    assert len(parallel.errors) == 1
    local_err, par_err = local.errors[0], parallel.errors[0]

    # Every diagnostic field, including ancestry, display parent, grain and
    # rendered error text, is invariant to worker/shard assignment.
    assert par_err == local_err

    # The trace localizes the *same* logical item no matter which worker ran it.
    assert par_err.logical_item == local_err.logical_item == f"{BOOM_DOC}/page={BOOM_PAGE}"
    assert par_err.source_item == local_err.source_item == BOOM_DOC
    assert par_err.failed_op == local_err.failed_op == "EmbedPage"
    assert par_err.upstream_path == local_err.upstream_path == ("SplitPages", "EmbedPage")
    assert par_err.action == "quarantined"

    # the faulted document kept its other pages, in order, missing only page 2
    bad_doc = next(v for v in parallel.values if v.startswith(f"{BOOM_DOC}="))
    assert f"#p{BOOM_PAGE})" not in bad_doc
    assert bad_doc == f"{BOOM_DOC}=[emb({BOOM_DOC}#p0)|emb({BOOM_DOC}#p1)|emb({BOOM_DOC}#p3)]"
