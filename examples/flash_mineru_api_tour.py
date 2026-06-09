"""Compare the common DagExecutor and RuntimeDagExecutor APIs.

The two pipelines share the same Flash-MinerU-shaped business DAG:

    pdf -> pdf2image -> layout -> ocr -> markdown

Run:

    python examples/flash_mineru_api_tour.py
"""
from __future__ import annotations

import argparse
import time
from typing import Any

import ray

from rayorch import (
    BadRecordError,
    DagExecutor,
    DagPipeline,
    Dispatch,
    RayModule,
    RuntimeDagExecutor,
    RuntimeRayModule,
    SequentialExecutor,
)


class Pdf2ImageDummy:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay

    def run(
        self,
        pdf: list[str],
        meta: list[dict[str, Any]],
    ) -> tuple[list[list[str]], list[dict[str, Any]]]:
        images = []
        for index, (path, item) in enumerate(zip(pdf, meta)):
            time.sleep(self.delay)
            if item.get("fault") == "pdf":
                raise BadRecordError("cannot decode PDF", index=index)
            item["pages"] = 2
            images.append([f"image:{path}:0", f"image:{path}:1"])
        return images, meta


class LayoutDummy:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay

    def run(
        self,
        images: list[list[str]],
        meta: list[dict[str, Any]],
    ) -> tuple[list[list[str]], list[dict[str, Any]]]:
        layouts = []
        for pages, item in zip(images, meta):
            time.sleep(self.delay)
            layouts.append(
                [f"layout:{item['name']}:{index}" for index in range(len(pages))]
            )
        return layouts, meta


class OcrDummy:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay

    def run(
        self,
        images: list[list[str]],
        layouts: list[list[str]],
        meta: list[dict[str, Any]],
    ) -> tuple[list[list[str]], list[dict[str, Any]]]:
        texts = []
        for index, (pages, page_layouts, item) in enumerate(
            zip(images, layouts, meta)
        ):
            time.sleep(self.delay)
            if item.get("fault") == "ocr":
                raise BadRecordError("OCR rejected document", index=index)
            texts.append(
                [
                    f"text:{item['name']}:{page}:{layout}"
                    for page, layout in zip(pages, page_layouts)
                ]
            )
        return texts, meta


class MarkdownDummy:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay

    def run(
        self,
        texts: list[list[str]],
        meta: list[dict[str, Any]],
    ) -> list[str]:
        output = []
        for pages, item in zip(texts, meta):
            time.sleep(self.delay)
            output.append(f"{item['name']}.md pages={len(pages)}")
        return output


class GenericFlashMineruPipeline(DagPipeline):
    """Lightweight RayModule pipeline: callers provide pre-split batches."""

    def __init__(self, delay: float) -> None:
        self.pdf2image = RayModule(
            Pdf2ImageDummy,
            replicas=2,
            dispatch_mode=Dispatch.SHARD_CONTIGUOUS,
            max_inflight=2,
            num_outputs=2,
        ).pre_init(delay)
        self.layout = RayModule(
            LayoutDummy,
            replicas=4,
            dispatch_mode=Dispatch.SHARD_CONTIGUOUS,
            max_inflight=4,
            num_outputs=2,
        ).pre_init(delay)
        self.ocr = RayModule(
            OcrDummy,
            replicas=4,
            dispatch_mode=Dispatch.SHARD_CONTIGUOUS,
            max_inflight=4,
            num_outputs=2,
        ).pre_init(delay)
        self.markdown = RayModule(
            MarkdownDummy,
            replicas=2,
            dispatch_mode=Dispatch.SHARD_CONTIGUOUS,
            max_inflight=2,
        ).pre_init(delay)
        super().__init__()

    def forward(
        self,
        pdf: list[str],
        meta: list[dict[str, Any]],
    ) -> list[str]:
        images, meta = self.pdf2image(pdf, meta)
        layouts, meta = self.layout(images, meta)
        texts, meta = self.ocr(images, layouts, meta)
        markdown = self.markdown(texts=texts, meta=meta)
        return markdown


class RuntimeFlashMineruPipeline(DagPipeline):
    """Runtime pipeline: the executor owns actors, batching, and lineage."""

    def __init__(self, delay: float) -> None:
        self.pdf2image = RuntimeRayModule(
            Pdf2ImageDummy,
            replicas=2,
            max_inflight=2,
            num_outputs=2,
        ).pre_init(delay)
        self.layout = RuntimeRayModule(
            LayoutDummy,
            replicas=4,
            max_inflight=4,
            num_outputs=2,
        ).pre_init(delay)
        self.ocr = RuntimeRayModule(
            OcrDummy,
            replicas=4,
            max_inflight=4,
            num_outputs=2,
        ).pre_init(delay)
        self.markdown = RuntimeRayModule(
            MarkdownDummy,
            replicas=2,
            max_inflight=2,
        ).pre_init(delay)
        super().__init__()

    def forward(
        self,
        pdf: list[str],
        meta: list[dict[str, Any]],
    ) -> list[str]:
        images, meta = self.pdf2image(pdf, meta)
        layouts, meta = self.layout(images, meta)
        texts, meta = self.ocr(images, layouts, meta)
        markdown = self.markdown(texts=texts, meta=meta)
        return markdown


def make_documents(
    count: int,
    *,
    faults: bool = False,
) -> tuple[list[str], list[dict[str, Any]]]:
    pdfs = [f"paper-{index:03d}.pdf" for index in range(count)]
    meta = [{"name": f"paper-{index:03d}", "fault": None} for index in range(count)]
    if faults and count >= 3:
        meta[1]["fault"] = "pdf"
        meta[-1]["fault"] = "ocr"
    return pdfs, meta


def chunk(values: list[Any], batch_size: int) -> list[list[Any]]:
    return [
        values[start : start + batch_size]
        for start in range(0, len(values), batch_size)
    ]


def close_generic_pipeline(pipeline: GenericFlashMineruPipeline) -> None:
    for module in (
        pipeline.pdf2image,
        pipeline.layout,
        pipeline.ocr,
        pipeline.markdown,
    ):
        for actor in module.actors:
            ray.kill(actor)


def run_generic(count: int, batch_size: int, delay: float) -> None:
    print("\n=== DagExecutor: lightweight pre-batched API ===")
    pipeline = GenericFlashMineruPipeline(delay)
    try:
        # A Pipeline itself eagerly executes one application batch.
        pdfs, meta = make_documents(min(count, batch_size))
        eager_output = pipeline(pdfs, meta)
        print("pipeline(pdf, meta):", eager_output)

        pdfs, meta = make_documents(count)
        pdf_batches = chunk(pdfs, batch_size)
        meta_batches = chunk(meta, batch_size)

        # SequentialExecutor and DagExecutor consume columns of pre-split batches.
        sequential = SequentialExecutor(pipeline)
        sequential_output = sequential.run(pdf=pdf_batches, meta=meta_batches)
        print("SequentialExecutor.run(...):", sequential_output)

        executor = DagExecutor(pipeline, max_batches_inflight=4)
        output = executor.run(pdf_batches, meta_batches)
        print("DagExecutor.run(pdf_batches, meta_batches):", output)

        # Executors are reusable; each run gets a fresh internal scheduler.
        retry_output = executor.run(pdf=pdf_batches[:1], meta=meta_batches[:1])
        print("reused executor:", retry_output)
    finally:
        close_generic_pipeline(pipeline)


def run_runtime(count: int, batch_size: int, delay: float) -> None:
    print("\n=== RuntimeDagExecutor: raw columns + RuntimeResult API ===")
    pipeline = RuntimeFlashMineruPipeline(delay)

    try:
        pipeline(["one.pdf"], [{"name": "one", "fault": None}])
    except RuntimeError as error:
        print("pipeline(...) is intentionally unavailable:", error)

    pdfs, meta = make_documents(count, faults=True)

    # RuntimeDagExecutor starts and owns RuntimeRayModule actors. It also splits
    # raw columns into MicroBatch objects and closes owned actors on context exit.
    with RuntimeDagExecutor(
        pipeline,
        batch_size=batch_size,
        max_batches_inflight=4,
        dataset="api-tour",
    ) as executor:
        results = executor.run(pdf=pdfs, meta=meta)

        for batch_index, result in enumerate(results):
            print(
                f"batch={batch_index} "
                f"rows={result.batch.row_ids} "
                f"markdown={result.batch.columns['markdown']}"
            )
            for error in result.quarantined:
                print(
                    f"  quarantined row={error.row_id} op={error.op} "
                    f"path={result.trace(error.path_id)} "
                    f"error={error.error!r}"
                )

        # Mapping input is equivalent to keyword columns and demonstrates reuse.
        clean_pdfs, clean_meta = make_documents(min(count, batch_size))
        clean = executor.run({"pdf": clean_pdfs, "meta": clean_meta})
        print("reused executor with mapping input:", clean[0].batch.columns["markdown"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--delay", type=float, default=0.01)
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True, num_cpus=16, include_dashboard=False)
    try:
        run_generic(args.documents, args.batch_size, args.delay)
        run_runtime(args.documents, args.batch_size, args.delay)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
