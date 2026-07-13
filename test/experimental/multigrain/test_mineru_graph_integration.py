"""Slow graph-motif integration tests over real MinerU image objects."""
from __future__ import annotations

from collections import Counter

import pytest

Image = pytest.importorskip("PIL.Image")

from rayorch.experimental import multigrain as mg
from rayorch.experimental.multigrain.graph import (
    MissingChildPolicy,
    PhysicalHints,
)
from rayorch.experimental.multigrain.ray_executor import MultigrainRayExecutor

from test.experimental.multigrain.mineru_integration_ops import (
    AssembleDoc,
    DocsToPages,
    ImageFeature,
    KeepRole,
    LayoutFeature,
    LinkVisualEvidence,
    MergeFeatures,
    PagesToBlocks,
    load_artifact_docs,
)


pytestmark = [pytest.mark.slow, pytest.mark.mineru_integration]


@pytest.fixture(scope="module")
def artifact_docs():
    docs = load_artifact_docs()
    if len(docs) < 2:
        pytest.skip("bounded MinerU regression artifacts are unavailable")
    return docs


def _source(docs):
    return mg.source(docs, name="docs", display_key=lambda doc: doc["name"])


class CompoundDiamondPipe(mg.Pipeline):
    """doc -> page -> (block, metadata), then a diamond join and doc reduce."""

    def __init__(self, replicas: int = 1, sleep_scale: float = 0.0) -> None:
        super().__init__()
        hints = PhysicalHints(replicas=replicas)
        self.to_pages = mg.Expand(DocsToPages, parent=0, child_label="page")
        self.to_blocks = mg.Expand(
            PagesToBlocks,
            parent=0,
            child_label="block",
            num_outputs=2,
        )
        self.image = mg.Map(
            ImageFeature,
            sleep_scale,
            name="image_feature",
            physical=hints,
        )
        self.layout = mg.Map(LayoutFeature, name="layout_feature", physical=hints)
        self.merge = mg.Map(MergeFeatures, name="merge_features")
        self.assemble = mg.Reduce(
            AssembleDoc,
            name="assemble_doc",
            missing_child=MissingChildPolicy.FAIL_CLOSED,
        )

    def forward(self, docs):
        pages = self.to_pages(docs)
        blocks, metadata = self.to_blocks(pages)
        image_features = self.image(blocks)
        layout_features = self.layout(blocks)
        merged = self.merge(image_features, layout_features)
        summary = self.assemble(mg.group_by(docs, merged))
        return (
            summary,
            pages,
            blocks,
            metadata,
            image_features,
            layout_features,
            merged,
        )


class NestedRelatePipe(mg.Pipeline):
    """Two nested 1:N expansions, a branch, true M:N relate, then N:1."""

    def __init__(self) -> None:
        super().__init__()
        self.to_pages = mg.Expand(DocsToPages, parent=0, child_label="page")
        self.to_blocks = mg.Expand(PagesToBlocks, parent=0, num_outputs=2)
        self.visual = mg.Filter(KeepRole, "visual", name="visual")
        self.evidence = mg.Filter(KeepRole, "evidence", name="evidence")
        self.link = mg.Relate(
            LinkVisualEvidence,
            on={"visual": "page_key", "evidence": "page_key"},
            output_grain="visual_evidence",
            name="link_visual_evidence",
        )
        self.assemble = mg.Reduce(AssembleDoc, name="assemble_links")

    def forward(self, docs):
        pages = self.to_pages(docs)
        blocks, _ = self.to_blocks(pages)
        visual = self.visual(blocks)
        evidence = self.evidence(blocks)
        linked = self.link(visual, evidence)
        return self.assemble(mg.group_by(docs, linked)), linked, visual, evidence


class MultiRootRelatePipe(mg.Pipeline):
    """An expanded document root joins an independently supplied catalog root."""

    def __init__(self) -> None:
        super().__init__()
        self.to_pages = mg.Expand(DocsToPages, parent=0)
        self.to_blocks = mg.Expand(PagesToBlocks, parent=0, num_outputs=2)
        self.visual = mg.Filter(KeepRole, "visual", name="root_visual")
        self.link = mg.Relate(
            LinkVisualEvidence,
            on={"visual": "page_key", "evidence": "page_key"},
            output_grain="catalog_link",
            name="catalog_link",
        )

    def forward(self, docs, catalog):
        pages = self.to_pages(docs)
        blocks, _ = self.to_blocks(pages)
        return self.link(self.visual(blocks), catalog)


def test_real_artifact_fixture_is_bounded_and_carries_image_objects(artifact_docs):
    blocks = [
        block
        for doc in artifact_docs
        for page in doc["pages"]
        for block in page["blocks"]
    ]
    assert blocks
    assert len(artifact_docs) <= 3
    assert all(isinstance(block["image"], Image.Image) for block in blocks)
    assert any(len(page["blocks"]) == 0 for page in artifact_docs[0]["pages"])
    assert any(block["role"] == "visual" for block in blocks)
    assert any(block["role"] == "evidence" for block in blocks)


def test_compound_expand_multioutput_and_diamond_lineage(artifact_docs):
    graph = CompoundDiamondPipe().compile()
    outputs = mg.MultigrainExecutor().execute(graph, {"docs": _source(artifact_docs)})
    summary, pages, blocks, metadata, image_features, layout_features, merged = outputs

    assert len(pages) == sum(len(doc["pages"]) for doc in artifact_docs)
    assert len(blocks) == len(metadata) == len(image_features) == len(layout_features)
    assert image_features.record_ids == layout_features.record_ids == merged.record_ids
    assert all(isinstance(value["image"], Image.Image) for value in blocks.values)
    assert all("DocsToPages" in path and "PagesToBlocks" in path for path in merged.lineage)
    assert all("image_feature" in path and "layout_feature" in path for path in merged.lineage)
    assert [value["count"] for value in summary.values] == [
        sum(len(page["blocks"]) for page in doc["pages"])
        for doc in artifact_docs
    ]


def test_nested_expand_diamond_matches_ray_execution(artifact_docs, ray_cluster):
    graph = CompoundDiamondPipe(replicas=2).compile()
    inputs = {"docs": _source(artifact_docs)}
    local = mg.MultigrainExecutor().execute(graph, inputs)
    executor = MultigrainRayExecutor()
    try:
        distributed = executor.execute(graph, inputs)
    finally:
        executor.shutdown()

    for local_port, ray_port in zip(local, distributed):
        assert ray_port.record_ids == local_port.record_ids
        assert ray_port.lineage == local_port.lineage
    assert distributed[0].values == local[0].values
    assert distributed[-1].values == local[-1].values


def test_mn_relate_emits_full_cartesian_matches_and_preserves_both_roots(artifact_docs):
    graph = NestedRelatePipe().compile()
    summary, linked, visual, evidence = mg.MultigrainExecutor().execute(
        graph,
        {"docs": _source(artifact_docs)},
    )
    visual_counts = Counter(item["page_key"] for item in visual.values)
    evidence_counts = Counter(item["page_key"] for item in evidence.values)
    expected = sum(
        visual_counts[key] * evidence_counts[key]
        for key in visual_counts.keys() & evidence_counts.keys()
    )

    assert len(linked) == expected
    assert sum(item["count"] for item in summary.values) == expected
    assert all({ref.role for ref in refs} == {"visual", "evidence"} for refs in linked.relations)
    assert all("docs" in ancestors for ancestors in linked.ancestors)


def test_multi_root_join_attaches_independent_catalog_lineage(artifact_docs):
    docs = _source(artifact_docs)
    catalog_values = [
        block
        for doc in artifact_docs
        for page in doc["pages"]
        for block in page["blocks"]
        if block["role"] == "evidence"
    ]
    catalog = mg.source(catalog_values, name="catalog")
    graph = MultiRootRelatePipe().compile()
    linked = mg.MultigrainExecutor().execute(
        graph,
        {"docs": docs, "catalog": catalog},
    )

    assert linked.values
    assert all("docs" in row for row in linked.ancestors)
    assert all("catalog" in row for row in linked.ancestors)
    assert all({ref.port for ref in refs} >= {"root_visual", "catalog"} for refs in linked.relations)
