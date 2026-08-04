"""V3.1 functional 语法通过未修改的 V3 actor runtime 执行。"""

from __future__ import annotations

import pytest

import rayorch.experimental.multigrain_v3_1 as mg


pytestmark = pytest.mark.usefixtures("ray_cluster")


class Render:
    def run(self, pdfs):
        groups = []
        for pdf, count in pdfs:
            groups.append([f"{pdf}:p{index}" for index in range(count)])
        return groups


class Ocr:
    def run(self, pages):
        return [f"ocr({page})" for page in pages]


class Assemble:
    def run(self, pdfs, content_groups, page_groups):
        return [
            (pdf[0], tuple(contents), tuple(pages))
            for pdf, contents, pages in zip(
                pdfs,
                content_groups,
                page_groups,
            )
        ]


class Pipeline(mg.Pipeline):
    def __init__(self):
        self.render = mg.RayModule(Render).ray_options(batch_size=2)
        self.ocr = mg.RayModule(Ocr).ray_options(batch_size=8)
        self.assemble = mg.RayModule(Assemble).ray_options(batch_size=2)

    def forward(self, pdfs):
        page_groups = self.render(pdfs)
        pages = mg.functional.expand(page_groups)
        contents = self.ocr(pages)
        content_groups = mg.functional.reduce(contents)
        return self.assemble(pdfs, content_groups, page_groups)


def test_functional_expand_reduce_preserve_order_and_empty_groups():
    result = mg.Executor(Pipeline()).run(
        [("a", 3), ("empty", 0), ("b", 2)]
    )

    assert result.get() == (
        (
            "a",
            ("ocr(a:p0)", "ocr(a:p1)", "ocr(a:p2)"),
            ("a:p0", "a:p1", "a:p2"),
        ),
        ("empty", (), ()),
        (
            "b",
            ("ocr(b:p0)", "ocr(b:p1)"),
            ("b:p0", "b:p1"),
        ),
    )
    assert result.metrics["actor_count_stage_1"] == 1.0
    assert result.metrics["actor_count_stage_2"] == 1.0
    assert result.metrics["actor_count_stage_3"] == 1.0
    assert "actor_count_stage_4" not in result.metrics
