"""Nested and parallel Expand/Reduce fibers through real Ray actors."""

from __future__ import annotations

import pytest

import rayorch.experimental.multigrain_v3 as mg


pytestmark = pytest.mark.usefixtures("ray_cluster")


class Pages:
    def run(self, documents):
        return [
            [
                {"doc": doc, "page": page}
                for page in range(page_count)
            ]
            for doc, page_count in documents
        ]


class Regions:
    def run(self, pages):
        return [
            [
                f"{page['doc']}:p{page['page']}:r{region}"
                for region in range(page["page"] + 1)
            ]
            for page in pages
        ]


class Tag:
    def run(self, rows):
        return [f"tag({row})" for row in rows]


class Gather:
    def run(self, groups):
        return [tuple(group) for group in groups]


class PageRecord:
    def run(self, region_groups):
        return [
            f"page[{','.join(group)}]"
            for group in region_groups
        ]


class NestedPipeline(mg.Pipeline):
    def __init__(self):
        self.pages = mg.Expand(Pages).ray_options(batch_size=2)
        self.regions = mg.Expand(Regions).ray_options(batch_size=4)
        self.tag = mg.Map(Tag).ray_options(batch_size=8)
        self.inner = mg.Reduce(PageRecord).ray_options(batch_size=4)
        self.outer = mg.Reduce(Gather).ray_options(batch_size=2)

    def forward(self, documents):
        pages = self.pages(documents)
        regions = self.regions(pages)
        tagged = self.tag(regions)
        page_records = self.inner(anchor=pages, members=tagged)
        return self.outer(anchor=documents, members=page_records)


def test_two_expand_two_reduce_restores_both_ordinal_levels():
    """Inner Region groups close before ordered Page groups close."""

    result = mg.Executor(NestedPipeline()).run(
        [("a", 2), ("b", 1)]
    )
    assert result.get() == (
        (
            "page[tag(a:p0:r0)]",
            "page[tag(a:p1:r0),tag(a:p1:r1)]",
        ),
        ("page[tag(b:p0:r0)]",),
    )


class Ocr:
    def run(self, pages):
        return [f"ocr:{page['doc']}:{page['page']}" for page in pages]


class Layout:
    def run(self, pages):
        return [f"layout:{page['doc']}:{page['page']}" for page in pages]


class DualGather:
    def run(self, groups):
        return [tuple(group) for group in groups]


class MultiReducePipeline(mg.Pipeline):
    def __init__(self):
        self.pages = mg.Expand(Pages).ray_options(batch_size=2)
        self.ocr = mg.Map(Ocr).ray_options(batch_size=8)
        self.layout = mg.Map(Layout).ray_options(batch_size=8)
        self.ocr_reduce = mg.Reduce(DualGather).ray_options(batch_size=2)
        self.layout_reduce = mg.Reduce(DualGather).ray_options(batch_size=2)

    def forward(self, documents):
        pages = self.pages(documents)
        ocr = self.ocr(pages)
        layout = self.layout(pages)
        return (
            self.ocr_reduce(anchor=documents, members=ocr),
            self.layout_reduce(anchor=documents, members=layout),
        )


def test_one_expand_can_feed_multiple_independent_reduces():
    """Two Reduce stages share cardinality but own independent accumulators."""

    result = mg.Executor(MultiReducePipeline()).run(
        [("a", 2), ("b", 1)]
    )
    assert result.get() == (
        ("ocr:a:0", "ocr:a:1"),
        ("ocr:b:0",),
        ("layout:a:0", "layout:a:1"),
        ("layout:b:0",),
    )


class PageAndRegionSummary(mg.Pipeline):
    def __init__(self):
        self.pages = mg.Expand(Pages).ray_options(batch_size=2)
        self.regions = mg.Expand(Regions).ray_options(batch_size=4)
        self.tag = mg.Map(Tag).ray_options(batch_size=8)
        self.inner = mg.Reduce(PageRecord).ray_options(batch_size=4)
        self.page_summary = mg.Reduce(Gather).ray_options(batch_size=2)
        self.region_summary = mg.Reduce(Gather).ray_options(batch_size=2)

    def forward(self, documents):
        pages = self.pages(documents)
        regions = self.regions(pages)
        page_records = self.inner(
            anchor=pages,
            members=self.tag(regions),
        )
        return (
            self.page_summary(anchor=documents, members=page_records),
            self.region_summary(anchor=documents, members=page_records),
        )


def test_multiple_expands_can_close_then_feed_multiple_outer_reduces():
    """Two Expand levels and multiple Reduce consumers compose without flattening."""

    result = mg.Executor(PageAndRegionSummary()).run([("a", 2)])
    expected = (
        "page[tag(a:p0:r0)]",
        "page[tag(a:p1:r0),tag(a:p1:r1)]",
    )
    assert result.get() == (expected, expected)
