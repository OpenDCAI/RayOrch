from __future__ import annotations

from rayorch.experimental import multigrain as mg
from rayorch.experimental.multigrain.passes import InsertRebatchAfterExpandPass

from test.experimental.multigrain.test_pdf_mvp import (
    Assemble,
    Layout,
    OCR,
    PdfToImages,
)
from test.experimental.multigrain.test_upper_primitives import (
    KeepPages,
    MatchImagesAndCaptions,
    ScoreAndKeep,
    pair_relation,
)


class ExecMineruPipe(mg.Pipeline):
    def __init__(self) -> None:
        super().__init__()
        self.pdf_to_images = mg.Expand(
            PdfToImages,
            {"a.pdf": 2},
            parent=0,
            child_label="page",
            num_outputs=2,
        )
        self.layout = mg.Map(Layout)
        self.ocr = mg.Map(OCR)
        self.assemble = mg.Reduce(Assemble)

    def forward(self, pdfs):
        images, page_meta = self.pdf_to_images(pdfs)
        layouts = self.layout(images)
        texts = self.ocr(images, layouts)
        return self.assemble(mg.group_by(pdfs, texts, page_meta))


def test_local_executor_runs_compiled_expand_map_reduce_ir() -> None:
    graph = ExecMineruPipe().compile()
    pdfs = mg.source(["a.pdf"], name="pdfs")

    markdown = mg.MultigrainExecutor().execute(graph, {"pdfs": pdfs})

    assert markdown.values == ["a.pdf|pages=0,1|texts=2"]
    assert markdown.record_ids == pdfs.record_ids


def test_local_executor_runs_rebatch_transformed_ir() -> None:
    graph = ExecMineruPipe().compile()
    transformed = InsertRebatchAfterExpandPass().run(graph).graph
    pdfs = mg.source(["a.pdf"], name="pdfs")

    markdown = mg.MultigrainExecutor().execute(transformed, {"pdfs": pdfs})

    assert markdown.values == ["a.pdf|pages=0,1|texts=2"]


class ExecFilterPipe(mg.Pipeline):
    def __init__(self) -> None:
        super().__init__()
        self.keep = mg.Filter(KeepPages)

    def forward(self, pages):
        return self.keep(pages)


def test_local_executor_runs_compiled_filter_ir() -> None:
    graph = ExecFilterPipe().compile()
    pages = mg.source(["good-0", "bad-1", "good-2"], name="pages")

    kept = mg.MultigrainExecutor().execute(graph, {"pages": pages})

    assert kept.values == ["good-0", "good-2"]
    assert kept.record_ids == [pages.record_ids[0], pages.record_ids[2]]


class ExecSelectPipe(mg.Pipeline):
    def __init__(self) -> None:
        super().__init__()
        self.select = mg.Select(ScoreAndKeep, num_annotations=1)

    def forward(self, pages):
        return self.select(pages)


def test_local_executor_runs_select_lowered_map_filter_project_ir() -> None:
    graph = ExecSelectPipe().compile()
    pages = mg.source(["good-0", "bad-1", "good-2"], name="pages")

    kept_pages, kept_scores = mg.MultigrainExecutor().execute(graph, {"pages": pages})

    assert kept_pages.values == ["good-0", "good-2"]
    assert kept_pages.record_ids == [pages.record_ids[0], pages.record_ids[2]]
    assert kept_scores.values == [0.9, 0.9]


class ExecRelatePipe(mg.Pipeline):
    def __init__(self) -> None:
        super().__init__()
        self.match = mg.Relate(
            MatchImagesAndCaptions,
            roles=("image", "caption"),
            output_grain="pair",
        )

    def forward(self, images, captions):
        return self.match(images, captions)


def test_local_executor_runs_compiled_relate_ir_with_relation_fn() -> None:
    graph = ExecRelatePipe().compile()
    images = mg.source(["img-0", "img-1"], name="image")
    captions = mg.source(["cap-1", "cap-0"], name="caption")

    pairs = mg.MultigrainExecutor(
        relation_fns={"MatchImagesAndCaptions": pair_relation}
    ).execute(graph, {"images": images, "captions": captions})

    assert pairs.display_keys == [
        "image=img-0/caption=cap-0",
        "image=img-1/caption=cap-1",
    ]
    assert pairs.relations[0][0].role == "image"
    assert pairs.relations[0][1].role == "caption"
