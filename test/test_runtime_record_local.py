from __future__ import annotations

from rayorch.runtime import BadRecordError, LineageStore, MicroBatch, run_rowwise


def test_rowwise_success_keeps_row_ids_and_tracks_path() -> None:
    lineage = LineageStore()
    batch = MicroBatch.source({"x": [1, 2, 3]})

    out, bad = run_rowwise(
        lambda xs: [x + 1 for x in xs],
        batch,
        op="add1",
        inputs=("x",),
        outputs=("y",),
        lineage=lineage,
    )

    assert bad == []
    assert out.columns["y"] == [2, 3, 4]
    assert out.row_ids == batch.row_ids
    assert lineage.trace(out.path_ids[0]) == ["add1"]


def test_rowwise_bad_record_is_quarantined_and_healthy_rows_continue() -> None:
    lineage = LineageStore()
    batch = MicroBatch.source({"pdf": ["a.pdf", "bad.pdf", "c.pdf"]})

    def pdf2img(pdfs):
        pages = []
        for pdf in pdfs:
            if pdf == "bad.pdf":
                raise ValueError("corrupt pdf")
            pages.append([f"page<{pdf}>"])
        return pages

    out, bad = run_rowwise(
        pdf2img,
        batch,
        op="pdf2img",
        inputs=("pdf",),
        outputs=("images",),
        lineage=lineage,
    )

    assert out.columns["images"] == [["page<a.pdf>"], ["page<c.pdf>"]]
    assert [r.values["pdf"] for r in bad] == ["bad.pdf"]
    assert lineage.quarantined == bad


def test_flash_mineru_shaped_dummy_pipeline_quarantines_bad_pdf() -> None:
    lineage = LineageStore()
    batch = MicroBatch.source(
        {
            "pdf": ["paper0.pdf", "corrupt.pdf", "paper2.pdf"],
            "meta": [{"name": "paper0"}, {"name": "corrupt"}, {"name": "paper2"}],
        },
        dataset="flash-mineru-dummy",
    )

    def pdf2img(pdfs, meta):
        images = []
        for i, (pdf, item) in enumerate(zip(pdfs, meta)):
            if pdf == "corrupt.pdf":
                raise BadRecordError("pdf parser failed", index=i)
            item["pages"] = 2
            images.append([f"img<{pdf}:0>", f"img<{pdf}:1>"])
        return images, meta

    def layout(images, meta):
        blocks = []
        for pages, item in zip(images, meta):
            item["blocks"] = len(pages)
            blocks.append([[{"page": i}] for i, _ in enumerate(pages)])
        return blocks, meta

    def ocr(blocks, meta):
        text = []
        for per_pdf, item in zip(blocks, meta):
            item["ocr_pages"] = len(per_pdf)
            text.append([f"ocr<{item['name']}>:{page[0]['page']}" for page in per_pdf])
        return text, meta

    def convert(text, meta):
        return [
            f"{item['name']}.md pages={item['ocr_pages']} blocks={item['blocks']}"
            for item in meta
        ]

    b1, bad = run_rowwise(
        pdf2img,
        batch,
        op="pdf2img",
        inputs=("pdf", "meta"),
        outputs=("images", "meta"),
        lineage=lineage,
    )
    b2, more_bad = run_rowwise(
        layout,
        b1,
        op="layout",
        inputs=("images", "meta"),
        outputs=("blocks", "meta"),
        lineage=lineage,
    )
    b3, more_bad2 = run_rowwise(
        ocr,
        b2,
        op="ocr",
        inputs=("blocks", "meta"),
        outputs=("text", "meta"),
        lineage=lineage,
    )
    b4, more_bad3 = run_rowwise(
        convert,
        b3,
        op="convert",
        inputs=("text", "meta"),
        outputs=("markdown",),
        lineage=lineage,
    )

    assert [r.values["pdf"] for r in bad] == ["corrupt.pdf"]
    assert more_bad == more_bad2 == more_bad3 == []
    assert b4.columns["markdown"] == [
        "paper0.md pages=2 blocks=2",
        "paper2.md pages=2 blocks=2",
    ]
    assert lineage.trace(b4.path_ids[0]) == ["pdf2img", "layout", "ocr", "convert"]
