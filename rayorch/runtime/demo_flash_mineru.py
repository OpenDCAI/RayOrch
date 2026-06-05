"""Run a small Flash-MinerU-like runtime demo."""
from __future__ import annotations

from .core import BadRecordError, LineageStore, MicroBatch, QuarantineRecord, run_rowwise


def _print_batch(name: str, batch: MicroBatch, lineage: LineageStore) -> None:
    print(f"\n[{name}] healthy rows = {len(batch)}")
    print("  row_ids :", batch.row_ids)
    print("  path_ids:", batch.path_ids)
    if batch.path_ids:
        print("  trace   :", lineage.trace(batch.path_ids[0]))
    for col, values in batch.columns.items():
        print(f"  {col}: {values}")


def _print_bad(name: str, bad: list[QuarantineRecord]) -> None:
    print(f"[{name}] quarantined = {len(bad)}")
    for record in bad:
        print(
            "  bad row:",
            {
                "row_id": record.row_id,
                "path_id": record.path_id,
                "op": record.op,
                "error": record.error,
                "values": record.values,
            },
        )


def main() -> None:
    lineage = LineageStore()
    source = MicroBatch.source(
        {
            "pdf": ["paper0.pdf", "corrupt.pdf", "paper2.pdf"],
            "meta": [
                {"name": "paper0"},
                {"name": "corrupt"},
                {"name": "paper2"},
            ],
        },
        dataset="flash-mineru-demo",
    )
    _print_batch("source", source, lineage)

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
            item["layout_blocks"] = len(pages)
            blocks.append([[{"type": "text", "page": i}] for i, _ in enumerate(pages)])
        return blocks, meta

    def ocr(blocks, meta):
        text = []
        for per_pdf_blocks, item in zip(blocks, meta):
            item["ocr_pages"] = len(per_pdf_blocks)
            text.append([
                f"ocr<{item['name']}>:page{page_blocks[0]['page']}"
                for page_blocks in per_pdf_blocks
            ])
        return text, meta

    def convert(text, meta):
        return [
            f"{item['name']}.md pages={item['ocr_pages']} "
            f"blocks={item['layout_blocks']} text={text_rows}"
            for text_rows, item in zip(text, meta)
        ]

    b1, bad1 = run_rowwise(
        pdf2img,
        source,
        op="pdf2img",
        inputs=("pdf", "meta"),
        outputs=("images", "meta"),
        lineage=lineage,
    )
    _print_batch("after pdf2img", b1, lineage)
    _print_bad("after pdf2img", bad1)

    b2, bad2 = run_rowwise(
        layout,
        b1,
        op="layout",
        inputs=("images", "meta"),
        outputs=("blocks", "meta"),
        lineage=lineage,
    )
    _print_batch("after layout", b2, lineage)
    _print_bad("after layout", bad2)

    b3, bad3 = run_rowwise(
        ocr,
        b2,
        op="ocr",
        inputs=("blocks", "meta"),
        outputs=("text", "meta"),
        lineage=lineage,
    )
    _print_batch("after ocr", b3, lineage)
    _print_bad("after ocr", bad3)

    b4, bad4 = run_rowwise(
        convert,
        b3,
        op="convert",
        inputs=("text", "meta"),
        outputs=("markdown",),
        lineage=lineage,
    )
    _print_batch("after convert", b4, lineage)
    _print_bad("after convert", bad4)

    print("\n[lineage store]")
    print("  paths      :", lineage.paths)
    print("  row_path   :", lineage.row_path)
    print("  quarantined:", lineage.quarantined)


if __name__ == "__main__":
    main()
