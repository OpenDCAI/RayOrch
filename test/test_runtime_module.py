from __future__ import annotations

import ray
import time

from rayorch import (
    DagExecutor,
    DagPipeline,
    Dispatch,
    PipeRef,
    RayModule,
    RuntimeDagExecutor,
    RuntimeRayModule,
)
from rayorch.runtime import BadRecordError, MicroBatch, RuntimeNodeSpec
from rayorch.runtime import RuntimeResult
import pytest


def _cleanup(*modules: RuntimeRayModule) -> None:
    for module in modules:
        for actor in getattr(module, "actors", []):
            try:
                ray.kill(actor)
            except Exception:
                pass


def _trace(paths: dict[str, tuple[str, str]], path: str) -> list[str]:
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


class SleepOp:
    def run(self, x):
        time.sleep(2.0)
        return [f"done:{v}" for v in x]


class ShortSleepOp:
    def run(self, x):
        time.sleep(0.35)
        return [v + 1 for v in x]


class RuntimeShortSleepOp:
    def run(self, x):
        time.sleep(0.35)
        return [v + 1 for v in x]


class FlashPdf2ImageDummyOp:
    def __init__(self, *, sleep_per_item: float = 0.05):
        self.sleep_per_item = sleep_per_item

    def run(self, pdf, meta):
        images = []
        for item_pdf, item_meta in zip(pdf, meta):
            time.sleep(self.sleep_per_item)
            item_meta["pages"] = 2
            images.append([f"img:{item_pdf}:0", f"img:{item_pdf}:1"])
        return images, meta


class FlashLayoutDummyOp:
    def __init__(self, *, sleep_per_item: float = 2.0):
        self.sleep_per_item = sleep_per_item

    def run(self, images, meta):
        layouts = []
        for pages, item_meta in zip(images, meta):
            time.sleep(self.sleep_per_item)
            item_meta["layout_pages"] = len(pages)
            layouts.append([f"layout:{item_meta['name']}:{i}" for i, _ in enumerate(pages)])
        return layouts, meta


class FlashOcrDummyOp:
    def __init__(self, *, sleep_per_item: float = 2.0):
        self.sleep_per_item = sleep_per_item

    def run(self, layouts, meta):
        texts = []
        for item_layouts, item_meta in zip(layouts, meta):
            time.sleep(self.sleep_per_item)
            item_meta["ocr_pages"] = len(item_layouts)
            texts.append([f"ocr:{item_meta['name']}:{i}" for i, _ in enumerate(item_layouts)])
        return texts, meta


class FlashOutputDummyOp:
    def __init__(self, *, sleep_per_item: float = 0.05):
        self.sleep_per_item = sleep_per_item

    def run(self, texts, meta):
        markdown = []
        for item_texts, item_meta in zip(texts, meta):
            time.sleep(self.sleep_per_item)
            markdown.append(
                f"{item_meta['name']}.md pages={len(item_texts)} "
                f"layout={item_meta['layout_pages']} ocr={item_meta['ocr_pages']}"
            )
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
            texts.append([f"ocr:{item_meta['name']}:{j}" for j, _ in enumerate(item_layouts)])
        return texts, meta


class FaultOutputOp:
    def run(self, texts, meta):
        return [f"{item_meta['name']}.md pages={len(item_texts)}" for item_texts, item_meta in zip(texts, meta)]


def test_runtime_module_shards_microbatch_across_replicas() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=8)
    pdf2img = RuntimeRayModule(
        Pdf2ImgOp,
        inputs=("pdf", "meta"),
        outputs=("images", "meta"),
        op="pdf2img",
        replicas=4,
        max_inflight=4,
    ).pre_init()
    try:
        pdfs = [f"doc_{i}.pdf" for i in range(16)]
        pdfs[5] = "doc_bad.pdf"
        batch = MicroBatch.source(
            {
                "pdf": pdfs,
                "meta": [{"name": f"doc_{i}"} for i in range(16)],
            },
            dataset="runtime-module-shard",
        )

        result = pdf2img(batch)

        assert isinstance(result, RuntimeResult)
        assert len(result.batch) == 15
        assert result.batch.row_ids == [
            row_id for i, row_id in enumerate(batch.row_ids) if i != 5
        ]
        assert [r.values["pdf"] for r in result.quarantined] == ["doc_bad.pdf"]
        assert _trace(result.paths, result.batch.path_ids[0]) == ["pdf2img"]
    finally:
        _cleanup(pdf2img)
        if ray.is_initialized():
            ray.shutdown()


def test_runtime_module_can_bind_spec_after_pre_init() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=4)
    pdf2img = RuntimeRayModule(Pdf2ImgOp, replicas=2, max_inflight=2).pre_init()
    pdf2img.bind_runtime_spec(
        RuntimeNodeSpec(
            node="pdf2img",
            inputs=("pdf", "meta"),
            outputs=("images", "meta"),
        )
    )
    try:
        batch = MicroBatch.source(
            {
                "pdf": ["doc_0.pdf", "doc_bad.pdf", "doc_2.pdf"],
                "meta": [{"name": "doc_0"}, {"name": "bad"}, {"name": "doc_2"}],
            },
            dataset="runtime-module-bind",
        )

        result = pdf2img(batch)

        assert len(result.batch) == 2
        assert [r.values["pdf"] for r in result.quarantined] == ["doc_bad.pdf"]
        assert _trace(result.paths, result.batch.path_ids[0]) == ["pdf2img"]
    finally:
        _cleanup(pdf2img)
        if ray.is_initialized():
            ray.shutdown()


def test_runtime_module_requires_spec_before_direct_call() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=2)
    pdf2img = RuntimeRayModule(Pdf2ImgOp, replicas=1).pre_init()
    try:
        batch = MicroBatch.source(
            {
                "pdf": ["doc_0.pdf"],
                "meta": [{"name": "doc_0"}],
            },
            dataset="runtime-module-unbound",
        )
        with pytest.raises(RuntimeError, match="requires a RuntimeNodeSpec"):
            pdf2img(batch)
    finally:
        _cleanup(pdf2img)
        if ray.is_initialized():
            ray.shutdown()


def test_dag_compile_binds_runtime_spec_from_forward_names() -> None:
    class RuntimePipe(DagPipeline):
        def __init__(self):
            self.pdf2img = RuntimeRayModule(Pdf2ImgOp, replicas=1)
            super().__init__()

        def forward(self, pdf: PipeRef, meta: PipeRef):
            images, meta = self.pdf2img(pdf, meta)
            return images, meta

    pipe = RuntimePipe().compile()

    assert pipe.pdf2img.runtime_spec == RuntimeNodeSpec(
        node="pdf2img",
        inputs=("pdf", "meta"),
        outputs=("images", "meta"),
    )
    assert pipe._compiled.nodes["pdf2img"].num_outputs == 2


def test_dag_compile_rejects_runtime_output_arity_mismatch() -> None:
    class RuntimePipe(DagPipeline):
        def __init__(self):
            self.pdf2img = RuntimeRayModule(Pdf2ImgOp, replicas=1, num_outputs=2)
            super().__init__()

        def forward(self, pdf: PipeRef, meta: PipeRef):
            images = self.pdf2img(pdf, meta)
            return images

    with pytest.raises(ValueError, match="assigns 1 outputs"):
        RuntimePipe().compile()


def test_dag_compile_keeps_ast_hints_aligned_with_nested_calls() -> None:
    class PassOp:
        def run(self, x):
            return x

    class RuntimePipe(DagPipeline):
        def __init__(self):
            self.keep = RuntimeRayModule(PassOp, replicas=1)
            super().__init__()

        def forward(self, pdf: PipeRef):
            first = self.keep(pdf)
            passthrough = self.keep(self.keep(first))
            second = self.keep(passthrough)
            return second

    pipe = RuntimePipe().compile()

    assert pipe._compiled.nodes["keep"].output_names == ("first",)
    assert pipe._compiled.nodes["keep_1"].output_names == ("keep_1.out0",)
    assert pipe._compiled.nodes["keep_2"].output_names == ("passthrough",)
    assert pipe._compiled.nodes["keep_3"].output_names == ("second",)
    assert pipe.keep.runtime_spec == RuntimeNodeSpec(
        node="keep_3",
        inputs=("passthrough",),
        outputs=("second",),
    )


def test_runtime_modules_flash_mineru_like_pipeline() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=16)
    pdf2img = RuntimeRayModule(
        Pdf2ImgOp,
        inputs=("pdf", "meta"),
        outputs=("images", "meta"),
        op="pdf2img",
        replicas=4,
        max_inflight=4,
    ).pre_init()
    layout = RuntimeRayModule(
        LayoutOp,
        inputs=("images", "meta"),
        outputs=("blocks", "meta"),
        op="layout",
        replicas=4,
        max_inflight=4,
    ).pre_init()
    ocr = RuntimeRayModule(
        OcrOp,
        inputs=("blocks", "meta"),
        outputs=("text", "meta"),
        op="ocr",
        replicas=4,
        max_inflight=4,
    ).pre_init()
    convert = RuntimeRayModule(
        ConvertOp,
        inputs=("text", "meta"),
        outputs=("markdown",),
        op="convert",
        replicas=4,
        max_inflight=4,
    ).pre_init()
    try:
        pdfs = [f"paper_{i}.pdf" for i in range(12)]
        pdfs[3] = "paper_bad.pdf"
        source = MicroBatch.source(
            {
                "pdf": pdfs,
                "meta": [{"name": f"paper_{i}"} for i in range(12)],
            },
            dataset="flash-mineru-runtime-module",
        )

        r1 = pdf2img(source)
        r2 = layout(r1.batch)
        r3 = ocr(r2.batch)
        r4 = convert(r3.batch)

        all_bad = r1.quarantined + r2.quarantined + r3.quarantined + r4.quarantined
        paths = {}
        for result in (r1, r2, r3, r4):
            paths.update(result.paths)

        assert [r.values["pdf"] for r in all_bad] == ["paper_bad.pdf"]
        assert len(r4.batch) == 11
        assert r4.batch.columns["markdown"][0] == "paper_0.md pages=2 blocks=2"
        assert r4.batch.columns["markdown"][-1] == "paper_11.md pages=2 blocks=2"
        assert _trace(paths, r4.batch.path_ids[0]) == [
            "pdf2img",
            "layout",
            "ocr",
            "convert",
        ]
    finally:
        _cleanup(pdf2img, layout, ocr, convert)
        if ray.is_initialized():
            ray.shutdown()


def test_runtime_dag_executor_runs_flash_mineru_like_pipeline() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=16)

    class RuntimeMineruPipe(DagPipeline):
        def __init__(self):
            self.pdf2img = RuntimeRayModule(
                Pdf2ImgOp, replicas=4, max_inflight=4, num_outputs=2
            ).pre_init()
            self.layout = RuntimeRayModule(
                LayoutOp, replicas=4, max_inflight=4, num_outputs=2
            ).pre_init()
            self.ocr = RuntimeRayModule(
                OcrOp, replicas=4, max_inflight=4, num_outputs=2
            ).pre_init()
            self.convert = RuntimeRayModule(
                ConvertOp, replicas=4, max_inflight=4, num_outputs=1
            ).pre_init()
            super().__init__()

        def forward(self, pdf: PipeRef, meta: PipeRef):
            images, meta = self.pdf2img(pdf, meta)
            blocks, meta = self.layout(images, meta)
            text, meta = self.ocr(blocks, meta)
            markdown = self.convert(text, meta)
            return markdown

    pipe = RuntimeMineruPipe()
    try:
        pdfs = [f"dag_{i}.pdf" for i in range(16)]
        pdfs[6] = "dag_bad.pdf"
        source = MicroBatch.source(
            {
                "pdf": pdfs,
                "meta": [{"name": f"dag_{i}"} for i in range(16)],
            },
            dataset="runtime-dag-flash-mineru",
        )

        result = RuntimeDagExecutor().run(pipe, source)

        assert pipe.pdf2img.runtime_spec == RuntimeNodeSpec(
            node="pdf2img",
            inputs=("pdf", "meta"),
            outputs=("images", "meta"),
        )
        assert len(result.batch) == 15
        assert result.batch.columns["markdown"][0] == "dag_0.md pages=2 blocks=2"
        assert result.batch.columns["markdown"][-1] == "dag_15.md pages=2 blocks=2"
        assert [r.values["pdf"] for r in result.quarantined] == ["dag_bad.pdf"]
        assert _trace(result.paths, result.batch.path_ids[0]) == [
            "pdf2img",
            "layout",
            "ocr",
            "convert",
        ]
    finally:
        _cleanup(pipe.pdf2img, pipe.layout, pipe.ocr, pipe.convert)
        if ray.is_initialized():
            ray.shutdown()


def test_runtime_dag_executor_uses_ray_replicas_for_dummy_latency() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=8)

    class SleepPipe(DagPipeline):
        def __init__(self):
            self.sleep = RuntimeRayModule(
                SleepOp, replicas=4, max_inflight=4, num_outputs=1
            ).pre_init()
            super().__init__()

        def forward(self, x: PipeRef):
            y = self.sleep(x)
            return y

    pipe = SleepPipe()
    try:
        source = MicroBatch.source(
            {"x": list(range(16))},
            dataset="runtime-dag-sleep",
        )

        start = time.perf_counter()
        result = RuntimeDagExecutor().run(pipe, source)
        elapsed = time.perf_counter() - start

        assert result.batch.columns["y"] == [f"done:{i}" for i in range(16)]
        assert len(result.quarantined) == 0
        assert elapsed < 4.5
    finally:
        _cleanup(pipe.sleep)
        if ray.is_initialized():
            ray.shutdown()


def test_runtime_dag_executor_overlaps_microbatches_like_dag_executor() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=12)

    class NormalSleepPipe(DagPipeline):
        def __init__(self):
            self.a = RayModule(ShortSleepOp, replicas=1, max_inflight=2).pre_init()
            self.b = RayModule(ShortSleepOp, replicas=1, max_inflight=2).pre_init()
            self.c = RayModule(ShortSleepOp, replicas=1, max_inflight=2).pre_init()
            super().__init__()

        def forward(self, x: PipeRef):
            y = self.a(x)
            y = self.b(y)
            y = self.c(y)
            return y

    class RuntimeSleepPipe(DagPipeline):
        def __init__(self):
            self.a = RuntimeRayModule(
                RuntimeShortSleepOp, replicas=1, max_inflight=2, num_outputs=1
            ).pre_init()
            self.b = RuntimeRayModule(
                RuntimeShortSleepOp, replicas=1, max_inflight=2, num_outputs=1
            ).pre_init()
            self.c = RuntimeRayModule(
                RuntimeShortSleepOp, replicas=1, max_inflight=2, num_outputs=1
            ).pre_init()
            super().__init__()

        def forward(self, x: PipeRef):
            y = self.a(x)
            y = self.b(y)
            y = self.c(y)
            return y

    normal_pipe = NormalSleepPipe()
    runtime_pipe = RuntimeSleepPipe()
    try:
        normal_batches = [[i] for i in range(4)]
        runtime_batches = [
            MicroBatch.source({"x": [i]}, dataset=f"overlap-{i}")
            for i in range(4)
        ]

        start = time.perf_counter()
        sequential = RuntimeDagExecutor(max_batches_inflight=1).run(
            runtime_pipe,
            runtime_batches,
        )
        runtime_sequential = time.perf_counter() - start

        start = time.perf_counter()
        runtime_results = RuntimeDagExecutor(max_batches_inflight=4).run(
            runtime_pipe,
            runtime_batches,
        )
        runtime_overlap = time.perf_counter() - start

        start = time.perf_counter()
        normal_results = DagExecutor(max_batches_inflight=4).run(
            normal_pipe,
            normal_batches,
        )
        normal_overlap = time.perf_counter() - start

        assert [r.batch.columns["y"] for r in runtime_results] == [[3], [4], [5], [6]]
        assert [r.batch.columns["y"] for r in sequential] == [[3], [4], [5], [6]]
        assert normal_results == [[3], [4], [5], [6]]
        print(
            "runtime sequential/overlap/dag overlap:",
            round(runtime_sequential, 3),
            round(runtime_overlap, 3),
            round(normal_overlap, 3),
        )
        assert runtime_overlap < runtime_sequential * 0.8
        assert runtime_overlap < normal_overlap * 1.6 + 0.5
    finally:
        _cleanup(runtime_pipe.a, runtime_pipe.b, runtime_pipe.c)
        for module in (normal_pipe.a, normal_pipe.b, normal_pipe.c):
            for actor in getattr(module, "actors", []):
                try:
                    ray.kill(actor)
                except Exception:
                    pass
        if ray.is_initialized():
            ray.shutdown()


def test_runtime_flash_mineru_like_benchmark_matches_generic_dag_executor() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=24)

    class GenericFlashPipe(DagPipeline):
        def __init__(self):
            self.pdf2img = RayModule(
                FlashPdf2ImageDummyOp,
                replicas=2,
                dispatch_mode=Dispatch.SHARD_CONTIGUOUS,
                max_inflight=2,
                num_outputs=2,
            ).pre_init(sleep_per_item=0.05)
            self.layout = RayModule(
                FlashLayoutDummyOp,
                replicas=4,
                dispatch_mode=Dispatch.SHARD_CONTIGUOUS,
                max_inflight=4,
                num_outputs=2,
            ).pre_init(sleep_per_item=2.0)
            self.ocr = RayModule(
                FlashOcrDummyOp,
                replicas=4,
                dispatch_mode=Dispatch.SHARD_CONTIGUOUS,
                max_inflight=4,
                num_outputs=2,
            ).pre_init(sleep_per_item=2.0)
            self.output = RayModule(
                FlashOutputDummyOp,
                replicas=2,
                dispatch_mode=Dispatch.SHARD_CONTIGUOUS,
                max_inflight=2,
                num_outputs=1,
            ).pre_init(sleep_per_item=0.05)
            super().__init__()

        def forward(self, pdf: PipeRef, meta: PipeRef):
            images, meta = self.pdf2img(pdf, meta)
            layouts, meta = self.layout(images, meta)
            texts, meta = self.ocr(layouts, meta)
            markdown = self.output(texts, meta)
            return markdown

    class RuntimeFlashPipe(DagPipeline):
        def __init__(self):
            self.pdf2img = RuntimeRayModule(
                FlashPdf2ImageDummyOp,
                replicas=2,
                max_inflight=2,
                num_outputs=2,
            ).pre_init(sleep_per_item=0.05)
            self.layout = RuntimeRayModule(
                FlashLayoutDummyOp,
                replicas=4,
                max_inflight=4,
                num_outputs=2,
            ).pre_init(sleep_per_item=2.0)
            self.ocr = RuntimeRayModule(
                FlashOcrDummyOp,
                replicas=4,
                max_inflight=4,
                num_outputs=2,
            ).pre_init(sleep_per_item=2.0)
            self.output = RuntimeRayModule(
                FlashOutputDummyOp,
                replicas=2,
                max_inflight=2,
                num_outputs=1,
            ).pre_init(sleep_per_item=0.05)
            super().__init__()

        def forward(self, pdf: PipeRef, meta: PipeRef):
            images, meta = self.pdf2img(pdf, meta)
            layouts, meta = self.layout(images, meta)
            texts, meta = self.ocr(layouts, meta)
            markdown = self.output(texts, meta)
            return markdown

    def cleanup_generic(pipe):
        for module in (pipe.pdf2img, pipe.layout, pipe.ocr, pipe.output):
            for actor in getattr(module, "actors", []):
                try:
                    ray.kill(actor)
                except Exception:
                    pass

    generic_pipe = GenericFlashPipe()
    runtime_pipe = RuntimeFlashPipe()
    try:
        batches = []
        runtime_batches = []
        for bi in range(5):
            pdfs = [f"flash_{bi}_{i}.pdf" for i in range(8)]
            meta = [{"name": f"flash_{bi}_{i}"} for i in range(8)]
            batches.append((pdfs, meta))
            runtime_batches.append(
                MicroBatch.source(
                    {"pdf": pdfs, "meta": [{"name": f"flash_{bi}_{i}"} for i in range(8)]},
                    dataset=f"flash-bench-{bi}",
                )
            )
        pdf_batches = [pdfs for pdfs, _ in batches]
        meta_batches = [meta for _, meta in batches]

        start = time.perf_counter()
        generic_out = DagExecutor(max_batches_inflight=5).run(
            generic_pipe,
            pdf_batches,
            meta_batches,
        )
        generic_elapsed = time.perf_counter() - start

        start = time.perf_counter()
        runtime_out = RuntimeDagExecutor(max_batches_inflight=5).run(
            runtime_pipe,
            runtime_batches,
        )
        runtime_elapsed = time.perf_counter() - start

        runtime_markdown = [result.batch.columns["markdown"] for result in runtime_out]
        assert runtime_markdown == generic_out
        assert sum(len(batch) for batch in runtime_markdown) == 40
        assert all(len(result.quarantined) == 0 for result in runtime_out)
        print(
            "flash-like generic/runtime:",
            round(generic_elapsed, 3),
            round(runtime_elapsed, 3),
        )
        assert runtime_elapsed < generic_elapsed * 1.25 + 1.0
        assert generic_elapsed < runtime_elapsed * 1.25 + 1.0
    finally:
        cleanup_generic(generic_pipe)
        _cleanup(runtime_pipe.pdf2img, runtime_pipe.layout, runtime_pipe.ocr, runtime_pipe.output)
        if ray.is_initialized():
            ray.shutdown()


def test_runtime_dag_executor_localizes_errors_with_multiple_batches_inflight() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=16)

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

    def make_batch(batch_idx: int, pdfs: list[str], names: list[str]) -> MicroBatch:
        return MicroBatch.source(
            {
                "pdf": pdfs,
                "meta": [{"name": name} for name in names],
            },
            dataset=f"fault-overlap-{batch_idx}",
        )

    pipe = FaultPipe()
    try:
        batches = [
            make_batch(
                0,
                ["b0_doc0.pdf", "b0_doc1.pdf", "b0_doc2.pdf", "b0_doc3.pdf"],
                ["b0_doc0", "b0_doc1", "b0_doc2", "b0_doc3"],
            ),
            make_batch(
                1,
                ["b1_doc0.pdf", "b1_bad_pdf.pdf", "b1_doc2.pdf", "b1_doc3.pdf"],
                ["b1_doc0", "b1_bad_pdf", "b1_doc2", "b1_doc3"],
            ),
            make_batch(
                2,
                ["b2_doc0.pdf", "b2_doc1.pdf", "b2_doc2.pdf", "b2_doc3.pdf"],
                ["b2_doc0", "b2_bad_ocr", "b2_doc2", "b2_doc3"],
            ),
            make_batch(
                3,
                ["b3_doc0.pdf", "b3_doc1.pdf", "b3_bad_pdf.pdf", "b3_doc3.pdf"],
                ["b3_doc0", "b3_bad_ocr", "b3_bad_pdf", "b3_doc3"],
            ),
        ]

        results = RuntimeDagExecutor(max_batches_inflight=4).run(pipe, batches)

        assert [len(result.batch) for result in results] == [4, 3, 3, 2]
        assert [len(result.quarantined) for result in results] == [0, 1, 1, 2]
        assert results[0].batch.columns["markdown"] == [
            "b0_doc0.md pages=2",
            "b0_doc1.md pages=2",
            "b0_doc2.md pages=2",
            "b0_doc3.md pages=2",
        ]
        assert results[1].batch.columns["markdown"] == [
            "b1_doc0.md pages=2",
            "b1_doc2.md pages=2",
            "b1_doc3.md pages=2",
        ]
        assert results[2].batch.columns["markdown"] == [
            "b2_doc0.md pages=2",
            "b2_doc2.md pages=2",
            "b2_doc3.md pages=2",
        ]
        assert results[3].batch.columns["markdown"] == [
            "b3_doc0.md pages=2",
            "b3_doc3.md pages=2",
        ]

        assert [(r.op, r.values["pdf"]) for r in results[1].quarantined] == [
            ("pdf2img", "b1_bad_pdf.pdf")
        ]
        assert [(r.op, r.values["meta"]["name"]) for r in results[2].quarantined] == [
            ("ocr", "b2_bad_ocr")
        ]
        assert sorted((r.op, r.values["meta"]["name"]) for r in results[3].quarantined) == [
            ("ocr", "b3_bad_ocr"),
            ("pdf2img", "b3_bad_pdf"),
        ]

        assert _trace(results[1].paths, results[1].batch.path_ids[0]) == [
            "pdf2img",
            "layout",
            "ocr",
            "output",
        ]
        assert _trace(results[2].paths, results[2].quarantined[0].path_id) == [
            "pdf2img",
            "layout",
        ]
        b3_by_op = {record.op: record for record in results[3].quarantined}
        assert _trace(results[3].paths, b3_by_op["pdf2img"].path_id) == []
        assert _trace(results[3].paths, b3_by_op["ocr"].path_id) == [
            "pdf2img",
            "layout",
        ]
    finally:
        _cleanup(pipe.pdf2img, pipe.layout, pipe.ocr, pipe.output)
        if ray.is_initialized():
            ray.shutdown()


def test_runtime_module_handles_varied_microbatch_sizes() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=8)
    pdf2img = RuntimeRayModule(
        Pdf2ImgOp,
        inputs=("pdf", "meta"),
        outputs=("images", "meta"),
        op="pdf2img",
        replicas=4,
        max_inflight=4,
    ).pre_init()
    try:
        cases = [
            (1, None),
            (2, 0),
            (3, 2),
            (4, 1),
            (5, 4),
            (16, 7),
            (17, 16),
        ]
        for n, bad_idx in cases:
            pdfs = [f"case_{n}_{i}.pdf" for i in range(n)]
            if bad_idx is not None:
                pdfs[bad_idx] = f"case_{n}_bad.pdf"
            batch = MicroBatch.source(
                {
                    "pdf": pdfs,
                    "meta": [{"name": f"case_{n}_{i}"} for i in range(n)],
                },
                dataset=f"varied-{n}",
            )

            result = pdf2img(batch)
            expected_bad = 0 if bad_idx is None else 1

            assert len(result.quarantined) == expected_bad
            assert len(result.batch) == n - expected_bad
            assert result.batch.row_ids == [
                row_id for i, row_id in enumerate(batch.row_ids) if i != bad_idx
            ]
            if bad_idx is not None:
                assert result.quarantined[0].row_id == batch.row_ids[bad_idx]
    finally:
        _cleanup(pdf2img)
        if ray.is_initialized():
            ray.shutdown()
