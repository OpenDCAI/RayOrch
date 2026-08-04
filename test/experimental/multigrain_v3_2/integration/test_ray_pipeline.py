"""通过真实 Ray actor 执行 multi-output partial expansion。"""

from __future__ import annotations

import pytest

import rayorch.experimental.multigrain_v3_2 as mg


pytestmark = pytest.mark.usefixtures("ray_cluster")


class Render:
    def run(self, pdfs):
        page_groups = []
        metadata = []
        for name, count in pdfs:
            page_groups.append([f"{name}:p{index}" for index in range(count)])
            metadata.append(f"meta({name})")
        return page_groups, metadata


class Ocr:
    def run(self, pages):
        return [f"ocr({page})" for page in pages]


class Layout:
    def run(self, pages):
        return [f"layout({page})" for page in pages]


class Assemble:
    def run(self, metadata, content_groups, layout_groups, page_groups):
        return [
            (
                meta,
                tuple(contents),
                tuple(layouts),
                tuple(pages),
            )
            for meta, contents, layouts, pages in zip(
                metadata,
                content_groups,
                layout_groups,
                page_groups,
            )
        ]


class Pipeline(mg.Pipeline):
    def __init__(self):
        self.render = mg.RayModule(Render).ray_options(
            num_outputs=2,
            batch_size=2,
        )
        self.ocr = mg.RayModule(Ocr).ray_options(batch_size=8)
        self.layout = mg.RayModule(Layout).ray_options(batch_size=8)
        self.assemble = mg.RayModule(Assemble).ray_options(batch_size=2)

    def forward(self, pdfs):
        page_groups, metadata = self.render(pdfs)
        pages = mg.functional.expand(page_groups)
        contents = self.ocr(pages)
        layouts = self.layout(pages)
        content_groups, layout_groups = mg.functional.reduce_aligned(
            contents,
            layouts,
        )
        return self.assemble(
            metadata,
            content_groups,
            layout_groups,
            page_groups,
        )


def test_partial_multioutput_expand_and_branch_reduce_roundtrip():
    result = mg.Executor(Pipeline()).run(
        [("a", 2), ("empty", 0), ("b", 1)]
    )

    assert result.get() == (
        (
            "meta(a)",
            ("ocr(a:p0)", "ocr(a:p1)"),
            ("layout(a:p0)", "layout(a:p1)"),
            ("a:p0", "a:p1"),
        ),
        ("meta(empty)", (), (), ()),
        (
            "meta(b)",
            ("ocr(b:p0)",),
            ("layout(b:p0)",),
            ("b:p0",),
        ),
    )
    # 六个 actor stage：render、structural expand、ocr、layout、
    # structural reduce、assemble；Source 不创建 actor。
    assert tuple(
        key
        for key in sorted(result.metrics)
        if key.startswith("actor_count_stage_")
    ) == (
        "actor_count_stage_1",
        "actor_count_stage_2",
        "actor_count_stage_3",
        "actor_count_stage_4",
        "actor_count_stage_5",
        "actor_count_stage_6",
    )
