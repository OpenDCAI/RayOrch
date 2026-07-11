from __future__ import annotations

import pytest

from rayorch.experimental import multigrain as mg
from rayorch.runtime import BadRecordError


class PdfToImages:
    def __init__(self, page_counts: dict[str, int]) -> None:
        self.page_counts = page_counts

    def run(self, pdfs: list[str]):
        image_groups = []
        meta_groups = []
        for pdf in pdfs:
            count = self.page_counts[pdf]
            image_groups.append([f"{pdf}:image:{page}" for page in range(count)])
            meta_groups.append(
                [{"pdf": pdf, "page": page} for page in range(count)]
            )
        return image_groups, meta_groups


class Layout:
    def run(self, images: list[str]) -> list[str]:
        return [f"layout:{image}" for image in images]


class OCR:
    def __init__(self, bad_image: str | None = None) -> None:
        self.bad_image = bad_image

    def run(self, images: list[str], layouts: list[str]) -> list[str]:
        for index, image in enumerate(images):
            if image == self.bad_image:
                raise BadRecordError("ocr failed", index=index)
        return [f"text:{image}:{layout}" for image, layout in zip(images, layouts)]


class Assemble:
    def run(
        self,
        pdfs: list[str],
        text_groups: list[list[str]],
        meta_groups: list[list[dict[str, int | str]]],
    ) -> list[str]:
        output = []
        for pdf, texts, metas in zip(pdfs, text_groups, meta_groups):
            pages = ",".join(str(meta["page"]) for meta in metas)
            output.append(f"{pdf}|pages={pages}|texts={len(texts)}")
        return output


class MineruPipe:
    def __init__(
        self,
        page_counts: dict[str, int],
        *,
        bad_image: str | None = None,
    ) -> None:
        self.pdf_to_images = mg.Expand(
            PdfToImages,
            page_counts,
            parent=0,
            child_label="page",
        )
        self.layout = mg.Map(Layout)
        self.ocr = mg.Map(OCR, bad_image)
        self.assemble = mg.Reduce(Assemble)

    def forward(self, pdfs: mg.PortBatch) -> mg.PortBatch:
        images, page_meta = self.pdf_to_images(pdfs)
        layouts = self.layout(images)
        texts = self.ocr(images, layouts)
        return self.assemble(mg.group_by(pdfs, texts, page_meta))


def test_expand_map_reduce_groups_pages_back_to_documents() -> None:
    pdfs = mg.source(["a.pdf", "b.pdf"], name="document")
    pipe = MineruPipe({"a.pdf": 2, "b.pdf": 3})

    markdown = pipe.forward(pdfs)

    assert markdown.values == [
        "a.pdf|pages=0,1|texts=2",
        "b.pdf|pages=0,1,2|texts=3",
    ]
    assert markdown.record_ids == pdfs.record_ids
    assert markdown.trace_item(document="a.pdf")[0]["lineage"] == ["Assemble"]


def test_page_failure_produces_readable_trace_and_keeps_healthy_pages() -> None:
    pdfs = mg.source(["a.pdf"], name="document")
    pipe = MineruPipe({"a.pdf": 3}, bad_image="a.pdf:image:1")

    markdown = pipe.forward(pdfs)

    # The MVP uses fail-open reduce semantics: page metadata remains available
    # while OCR text is missing for the quarantined page.
    assert markdown.values == ["a.pdf|pages=0,1,2|texts=2"]
    assert len(markdown.errors) == 1
    trace = markdown.errors[0]
    assert trace.source_item == "a.pdf"
    assert trace.logical_item == "a.pdf/page=1"
    assert trace.failed_op == "OCR"
    assert trace.grain == "PdfToImages"
    assert trace.parent == "a.pdf"
    assert trace.upstream_path == ("PdfToImages", "Layout", "OCR")
    assert trace.action == "quarantined"


def test_rebatch_preserves_identity_relation_and_child_order() -> None:
    pdfs = mg.source(["short.pdf", "long.pdf", "mid.pdf"], name="document")
    expand = mg.Expand(
        PdfToImages,
        {"short.pdf": 1, "long.pdf": 5, "mid.pdf": 2},
        parent=0,
        child_label="page",
    )
    images, page_meta = expand(pdfs)

    image_batches = mg.rebatch(images, batch_size=3)
    meta_batches = mg.rebatch(page_meta, batch_size=3)
    rebalanced_images = mg.concat(image_batches, name=images.name)
    rebalanced_meta = mg.concat(meta_batches, name=page_meta.name)

    assert [len(batch) for batch in image_batches] == [3, 3, 2]
    assert rebalanced_images.record_ids == images.record_ids
    assert rebalanced_images.ordinals == images.ordinals

    assemble = mg.Reduce(Assemble)
    markdown = assemble(mg.group_by(pdfs, rebalanced_images, rebalanced_meta))

    assert markdown.values == [
        "short.pdf|pages=0|texts=1",
        "long.pdf|pages=0,1,2,3,4|texts=5",
        "mid.pdf|pages=0,1|texts=2",
    ]


def test_map_rejects_cross_grain_inputs_without_group_by() -> None:
    pdfs = mg.source(["a.pdf"], name="document")
    images, _ = mg.Expand(
        PdfToImages,
        {"a.pdf": 2},
        parent=0,
        child_label="page",
    )(pdfs)

    with pytest.raises(ValueError, match="cannot align"):
        mg.Map(Layout)(pdfs, images)

