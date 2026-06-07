from __future__ import annotations

import time

import ray

from rayorch import DagPipeline, PipeRef, RuntimeRayModule
from rayorch.runtime import BadRecordError


def cleanup_modules(*modules) -> None:
    for module in modules:
        for actor in getattr(module, "actors", []):
            try:
                ray.kill(actor)
            except Exception:
                pass


def cleanup_pipeline(pipe: DagPipeline) -> None:
    cleanup_modules(
        *[
            value
            for value in pipe.__dict__.values()
            if hasattr(value, "actors")
        ]
    )


def trace_path(paths: dict[str, tuple[str, str]], path: str) -> list[str]:
    ops: list[str] = []
    seen: set[str] = set()
    while path != "source" and path not in seen:
        seen.add(path)
        parent, op = paths[path]
        ops.append(op)
        path = parent
    return list(reversed(ops))


class Pdf2ImgOp:
    def run(self, pdfs, meta):
        images = []
        for i, (pdf, item) in enumerate(zip(pdfs, meta)):
            if pdf.endswith("bad.pdf"):
                raise BadRecordError("pdf parser failed", index=i)
            item["pages"] = 2
            images.append([f"img<{pdf}:0>", f"img<{pdf}:1>"])
        return images, meta


class LayoutOp:
    def run(self, images, meta):
        blocks = []
        for pages, item in zip(images, meta):
            item["layout_blocks"] = len(pages)
            blocks.append([[{"page": i, "type": "text"}] for i, _ in enumerate(pages)])
        return blocks, meta


class OcrOp:
    def run(self, blocks, meta):
        text = []
        for per_pdf_blocks, item in zip(blocks, meta):
            item["ocr_pages"] = len(per_pdf_blocks)
            text.append([
                f"ocr<{item['name']}>:{page_blocks[0]['page']}"
                for page_blocks in per_pdf_blocks
            ])
        return text, meta


class ConvertOp:
    def run(self, text, meta):
        return [
            f"{item['name']}.md pages={item['ocr_pages']} blocks={item['layout_blocks']}"
            for item in meta
        ]


class RuntimeMineruPipe(DagPipeline):
    def __init__(
        self,
        *,
        pdf_replicas: int = 4,
        layout_replicas: int = 4,
        ocr_replicas: int = 4,
        output_replicas: int = 4,
        max_inflight: int = 4,
    ):
        self.pdf2img = RuntimeRayModule(
            Pdf2ImgOp,
            replicas=pdf_replicas,
            max_inflight=max_inflight,
            num_outputs=2,
        ).pre_init()
        self.layout = RuntimeRayModule(
            LayoutOp,
            replicas=layout_replicas,
            max_inflight=max_inflight,
            num_outputs=2,
        ).pre_init()
        self.ocr = RuntimeRayModule(
            OcrOp,
            replicas=ocr_replicas,
            max_inflight=max_inflight,
            num_outputs=2,
        ).pre_init()
        self.convert = RuntimeRayModule(
            ConvertOp,
            replicas=output_replicas,
            max_inflight=max_inflight,
            num_outputs=1,
        ).pre_init()
        super().__init__()

    def forward(self, pdf: PipeRef, meta: PipeRef):
        images, meta = self.pdf2img(pdf, meta)
        blocks, meta = self.layout(images, meta)
        text, meta = self.ocr(blocks, meta)
        markdown = self.convert(text, meta)
        return markdown


class FaultPdf2ImageOp:
    def run(self, pdf, meta):
        images = []
        for i, (item_pdf, item_meta) in enumerate(zip(pdf, meta)):
            time.sleep(0.05)
            if "bad_pdf" in item_pdf:
                raise BadRecordError("pdf decode failed", index=i)
            item_meta["pages"] = 2
            images.append([f"img:{item_pdf}:0", f"img:{item_pdf}:1"])
        return images, meta


class FaultLayoutOp:
    def run(self, images, meta):
        layouts = []
        for pages, item_meta in zip(images, meta):
            time.sleep(0.05)
            item_meta["layout_pages"] = len(pages)
            layouts.append([f"layout:{item_meta['name']}:{i}" for i, _ in enumerate(pages)])
        return layouts, meta


class FaultOcrOp:
    def run(self, layouts, meta):
        texts = []
        for i, (item_layouts, item_meta) in enumerate(zip(layouts, meta)):
            time.sleep(0.05)
            if item_meta["name"].endswith("bad_ocr"):
                raise BadRecordError("ocr failed", index=i)
            item_meta["ocr_pages"] = len(item_layouts)
            texts.append([
                f"ocr:{item_meta['name']}:{j}"
                for j, _ in enumerate(item_layouts)
            ])
        return texts, meta


class FaultOutputOp:
    def run(self, texts, meta):
        return [
            f"{item_meta['name']}.md pages={len(item_texts)}"
            for item_texts, item_meta in zip(texts, meta)
        ]


class FaultPipe(DagPipeline):
    def __init__(self):
        self.pdf2img = RuntimeRayModule(
            FaultPdf2ImageOp, replicas=2, max_inflight=4, num_outputs=2
        ).pre_init()
        self.layout = RuntimeRayModule(
            FaultLayoutOp, replicas=4, max_inflight=4, num_outputs=2
        ).pre_init()
        self.ocr = RuntimeRayModule(
            FaultOcrOp, replicas=4, max_inflight=4, num_outputs=2
        ).pre_init()
        self.output = RuntimeRayModule(
            FaultOutputOp, replicas=2, max_inflight=4, num_outputs=1
        ).pre_init()
        super().__init__()

    def forward(self, pdf: PipeRef, meta: PipeRef):
        images, meta = self.pdf2img(pdf, meta)
        layouts, meta = self.layout(images, meta)
        texts, meta = self.ocr(layouts, meta)
        markdown = self.output(texts, meta)
        return markdown
