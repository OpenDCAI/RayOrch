from __future__ import annotations

import time

import pytest
import ray

from rayorch import (
    DagExecutor,
    DagPipeline,
    Dispatch,
    RayModule,
    RuntimeDagExecutor,
    RuntimeRayModule,
)
from test.runtime.helpers import cleanup_modules


class Pdf2ImageDummyOp:
    def __init__(self, sleep_per_item: float = 0.05):
        self.sleep_per_item = sleep_per_item

    def run(self, pdf, meta):
        images = []
        for item_pdf, item_meta in zip(pdf, meta):
            time.sleep(self.sleep_per_item)
            item_meta["pages"] = 2
            images.append([f"img:{item_pdf}:0", f"img:{item_pdf}:1"])
        return images, meta


class LayoutDummyOp:
    def __init__(self, sleep_per_item: float = 2.0):
        self.sleep_per_item = sleep_per_item

    def run(self, images, meta):
        layouts = []
        for pages, item_meta in zip(images, meta):
            time.sleep(self.sleep_per_item)
            item_meta["layout_pages"] = len(pages)
            layouts.append([f"layout:{item_meta['name']}:{i}" for i, _ in enumerate(pages)])
        return layouts, meta


class OcrDummyOp:
    def __init__(self, sleep_per_item: float = 2.0):
        self.sleep_per_item = sleep_per_item

    def run(self, layouts, meta):
        texts = []
        for item_layouts, item_meta in zip(layouts, meta):
            time.sleep(self.sleep_per_item)
            item_meta["ocr_pages"] = len(item_layouts)
            texts.append([f"ocr:{item_meta['name']}:{i}" for i, _ in enumerate(item_layouts)])
        return texts, meta


class OutputDummyOp:
    def __init__(self, sleep_per_item: float = 0.05):
        self.sleep_per_item = sleep_per_item

    def run(self, texts, meta):
        output = []
        for item_texts, item_meta in zip(texts, meta):
            time.sleep(self.sleep_per_item)
            output.append(
                f"{item_meta['name']}.md pages={len(item_texts)} "
                f"layout={item_meta['layout_pages']} ocr={item_meta['ocr_pages']}"
            )
        return output


class GenericFlashPipe(DagPipeline):
    def __init__(self):
        self.pdf2img = RayModule(
            Pdf2ImageDummyOp,
            replicas=2,
            dispatch_mode=Dispatch.SHARD_CONTIGUOUS,
            max_inflight=2,
            num_outputs=2,
        ).pre_init()
        self.layout = RayModule(
            LayoutDummyOp,
            replicas=4,
            dispatch_mode=Dispatch.SHARD_CONTIGUOUS,
            max_inflight=4,
            num_outputs=2,
        ).pre_init()
        self.ocr = RayModule(
            OcrDummyOp,
            replicas=4,
            dispatch_mode=Dispatch.SHARD_CONTIGUOUS,
            max_inflight=4,
            num_outputs=2,
        ).pre_init()
        self.output = RayModule(
            OutputDummyOp,
            replicas=2,
            dispatch_mode=Dispatch.SHARD_CONTIGUOUS,
            max_inflight=2,
        ).pre_init()
        super().__init__()

    def forward(
        self,
        pdf: list[str],
        meta: list[dict[str, object]],
    ) -> list[str]:
        images, meta = self.pdf2img(pdf, meta)
        layouts, meta = self.layout(images, meta)
        texts, meta = self.ocr(layouts, meta)
        return self.output(texts, meta)


class RuntimeFlashPipe(DagPipeline):
    def __init__(self):
        self.pdf2img = RuntimeRayModule(
            Pdf2ImageDummyOp, replicas=2, max_inflight=2, num_outputs=2
        ).pre_init()
        self.layout = RuntimeRayModule(
            LayoutDummyOp, replicas=4, max_inflight=4, num_outputs=2
        ).pre_init()
        self.ocr = RuntimeRayModule(
            OcrDummyOp, replicas=4, max_inflight=4, num_outputs=2
        ).pre_init()
        self.output = RuntimeRayModule(
            OutputDummyOp, replicas=2, max_inflight=2, num_outputs=1
        ).pre_init()
        super().__init__()

    def forward(
        self,
        pdf: list[str],
        meta: list[dict[str, object]],
    ) -> list[str]:
        images, meta = self.pdf2img(pdf, meta)
        layouts, meta = self.layout(images, meta)
        texts, meta = self.ocr(layouts, meta)
        markdown = self.output(texts, meta)
        return markdown


@pytest.mark.slow
def test_flash_mineru_like_runtime_matches_generic_dag_executor() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=24)
    generic_pipe = GenericFlashPipe()
    runtime_pipe = RuntimeFlashPipe()
    runtime_executor = None
    try:
        pdf_batches = []
        meta_batches = []
        runtime_pdfs = []
        runtime_meta = []
        for batch_index in range(5):
            pdfs = [f"flash_{batch_index}_{i}.pdf" for i in range(8)]
            meta = [{"name": f"flash_{batch_index}_{i}"} for i in range(8)]
            pdf_batches.append(pdfs)
            meta_batches.append(meta)
            runtime_pdfs.extend(pdfs)
            runtime_meta.extend(
                {"name": f"flash_{batch_index}_{i}"}
                for i in range(8)
            )

        generic_executor = DagExecutor(generic_pipe, max_batches_inflight=5)
        runtime_executor = RuntimeDagExecutor(
            runtime_pipe,
            batch_size=8,
            max_batches_inflight=5,
            dataset="flash-bench",
        )

        generic_executor.run(
            [["warmup.pdf"]],
            [[{"name": "warmup"}]],
        )
        runtime_executor.run(
            pdf=["warmup.pdf"],
            meta=[{"name": "warmup"}],
        )

        start = time.perf_counter()
        generic_output = generic_executor.run(pdf_batches, meta_batches)
        generic_time = time.perf_counter() - start

        start = time.perf_counter()
        runtime_results = runtime_executor.run(pdf=runtime_pdfs, meta=runtime_meta)
        runtime_time = time.perf_counter() - start

        runtime_output = [
            result.batch.columns["markdown"] for result in runtime_results
        ]
        assert runtime_output == generic_output
        assert sum(len(batch) for batch in runtime_output) == 40
        assert [
            row_id
            for result in runtime_results
            for row_id in result.batch.row_ids
        ] == [f"flash-bench:{index}" for index in range(40)]
        assert runtime_time < generic_time * 1.25 + 1.0
        assert generic_time < runtime_time * 1.25 + 1.0
        print(
            "flash-like generic/runtime:",
            round(generic_time, 3),
            round(runtime_time, 3),
        )
    finally:
        if runtime_executor is not None:
            runtime_executor.close()
        cleanup_modules(
            generic_pipe.pdf2img,
            generic_pipe.layout,
            generic_pipe.ocr,
            generic_pipe.output,
            runtime_pipe.pdf2img,
            runtime_pipe.layout,
            runtime_pipe.ocr,
            runtime_pipe.output,
        )
        ray.shutdown()
