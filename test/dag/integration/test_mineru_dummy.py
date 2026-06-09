"""
MinerU-shaped DAG integration tests for ``rayorch.dag_new_pipeline``.

Validates the same operator wiring as ``Flash-MinerU/flash_mineru/main_four_rayorch_dag_new.py``
(``MinerURayOrchDagNewPipeline``) and ``test_dummy_mineru_dag.py``, using lightweight dummy Ops so
CI can run without GPUs or MinerU weights. Ray uses the host CPU count unless
``RAYORCH_TEST_NUM_CPUS`` is set.

Archive / design notes: ``docs/dag_new_pipeline_architecture.md`` (source: ``rayorch/dag_new_pipeline.py``).
"""

from __future__ import annotations

import os
from typing import List

import ray

from rayorch import Dispatch, RayModule
from rayorch.dag_new_pipeline import DagExecutor, DagPipeline, SequentialExecutor


def _ray_init_local() -> None:
    """Initialize Ray for tests without capping below host capacity.

    By default Ray detects local CPU count. Set ``RAYORCH_TEST_NUM_CPUS`` to force
    a specific slot count (e.g. constrained CI runners).
    """
    if ray.is_initialized():
        return
    raw = os.environ.get("RAYORCH_TEST_NUM_CPUS", "").strip()
    if raw:
        ray.init(ignore_reinit_error=True, num_cpus=max(1, int(raw)))
    else:
        ray.init(ignore_reinit_error=True)


def _chunked(items: List[str], batch_size: int) -> List[List[str]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _flatten(nested: List[List[str]]) -> List[str]:
    out: List[str] = []
    for x in nested:
        out.extend(x)
    return out


def _cleanup_modules(*modules: RayModule) -> None:
    for m in modules:
        for actor in getattr(m, "actors", []):
            try:
                ray.kill(actor)
            except Exception:
                pass


class DummyBlock(dict):
    """ContentBlock-like dict subclass to mimic MinerU block payloads."""

    def __init__(self, type: str, bbox: list[float], content: str | None = None):
        super().__init__()
        self["type"] = type
        self["bbox"] = bbox
        self["angle"] = None
        self["content"] = content

    @property
    def type(self) -> str:
        return self["type"]

    @type.setter
    def type(self, value: str) -> None:
        self["type"] = value

    @property
    def content(self) -> str | None:
        return self.get("content")

    @content.setter
    def content(self, value: str | None) -> None:
        self["content"] = value


class Pdf2ImageDummyOp:
    def run(self, pdf_path_list: List[str]) -> List[List[dict]]:
        out: List[List[dict]] = []
        for pdf in pdf_path_list:
            # Deterministic across processes (avoid PYTHONHASHSEED-dependent ``hash()``).
            page_n = 2 + (sum(ord(c) for c in pdf) % 3)
            pages = []
            for i in range(page_n):
                pages.append(
                    {
                        "pdf_path": pdf,
                        "page_id": i,
                        "page_width": 1000,
                        "page_height": 1400,
                        "img_pil": f"dummy_img<{pdf}:{i}>",
                    }
                )
            out.append(pages)
        return out


class LayoutDetectionDummyOp:
    def run(self, image_dict_list: List[List[dict]]) -> List[List[List[DummyBlock]]]:
        output: List[List[List[DummyBlock]]] = []
        for pdf_pages in image_dict_list:
            blocks_list = []
            for page in pdf_pages:
                page_blocks = [
                    DummyBlock(
                        type="text",
                        bbox=[10.0, 20.0, 300.0, 100.0],
                        content=f"layout<{page['page_id']}>#0",
                    )
                ]
                blocks_list.append(page_blocks)
            output.append(blocks_list)
        return output


class OCRDummyOp:
    def run(
        self,
        image_dict_list: List[List[dict]],
        blocks_list_per_pdf: List[List[List[DummyBlock]]],
    ) -> List[List[List[dict]]]:
        out: List[List[List[dict]]] = []
        for pdf_pages, per_pdf_blocks in zip(image_dict_list, blocks_list_per_pdf):
            pdf_out = []
            for page, page_blocks in zip(pdf_pages, per_pdf_blocks):
                page_out = []
                for b in page_blocks:
                    page_out.append(
                        {
                            "type": b.type,
                            "bbox": b["bbox"],
                            "angle": b.get("angle"),
                            "content": f"ocr<{page['page_id']}>:{b.content}",
                        }
                    )
                pdf_out.append(page_out)
            out.append(pdf_out)
        return out


class Convert2MDDummyOp:
    def run(self, model_results: List[List[List[dict]]], images: List[List[dict]]) -> List[str]:
        outs: List[str] = []
        for idx, image_pages in enumerate(images):
            pdf_path = image_pages[0]["pdf_path"]
            name = os.path.splitext(os.path.basename(pdf_path))[0]
            n_pages = len(model_results[idx])
            n_blocks = sum(len(p) for p in model_results[idx])
            outs.append(f"{name}.md (pages={n_pages}, blocks={n_blocks})")
        return outs


class MineruShapedDummyPipeline(DagPipeline):
    """Same DAG shape as MinerU: pdf2img -> layout -> ocr -> img2md."""

    def __init__(self, *, replicas: int) -> None:
        self.pdf2img = RayModule(Pdf2ImageDummyOp, replicas=1, num_gpus_per_replica=0.0).pre_init()
        self.layout = RayModule(
            LayoutDetectionDummyOp,
            replicas=replicas,
            num_gpus_per_replica=0.0,
            dispatch_mode=Dispatch.SHARD_CONTIGUOUS,
            max_inflight=max(1, replicas),
        ).pre_init()
        self.ocr = RayModule(
            OCRDummyOp,
            replicas=replicas,
            num_gpus_per_replica=0.0,
            dispatch_mode=Dispatch.SHARD_CONTIGUOUS,
            max_inflight=max(1, replicas),
        ).pre_init()
        self.img2md = RayModule(
            Convert2MDDummyOp,
            replicas=1,
            num_gpus_per_replica=0.0,
            dispatch_mode=Dispatch.BROADCAST,
        ).pre_init()
        super().__init__()

    def forward(self, x):
        images = self.pdf2img(x)
        layout_items = self.layout(images)
        ocr_results = self.ocr(images, layout_items)
        return self.img2md(model_results=ocr_results, images=images)


def test_mineru_shaped_compiled_graph_topology() -> None:
    _ray_init_local()
    pipe: MineruShapedDummyPipeline | None = None
    try:
        pipe = MineruShapedDummyPipeline(replicas=2)
        pipe.compile()
        g = pipe._compiled
        assert g is not None
        assert g.topo_order == ("pdf2img", "layout", "ocr", "img2md")
        assert set(g.nodes) == {"pdf2img", "layout", "ocr", "img2md"}
        assert g.input_keys == ("__input__x",)
    finally:
        if pipe is not None:
            _cleanup_modules(pipe.pdf2img, pipe.layout, pipe.ocr, pipe.img2md)
        if ray.is_initialized():
            ray.shutdown()


def test_mineru_shaped_serial_matches_dag_executor() -> None:
    _ray_init_local()
    pipe: MineruShapedDummyPipeline | None = None
    try:
        replicas = 2
        pipe = MineruShapedDummyPipeline(replicas=replicas)
        pdfs = [f"/tmp/mineru_test_{i}.pdf" for i in range(5)]
        batches = _chunked(pdfs, batch_size=2)

        serial = SequentialExecutor(pipe).run(batches)
        dag = DagExecutor(pipe, max_batches_inflight=3).run(batches)

        assert serial == dag
        assert _flatten(serial) == [
            "mineru_test_0.md (pages=2, blocks=2)",
            "mineru_test_1.md (pages=3, blocks=3)",
            "mineru_test_2.md (pages=4, blocks=4)",
            "mineru_test_3.md (pages=2, blocks=2)",
            "mineru_test_4.md (pages=3, blocks=3)",
        ]
    finally:
        if pipe is not None:
            _cleanup_modules(pipe.pdf2img, pipe.layout, pipe.ocr, pipe.img2md)
        if ray.is_initialized():
            ray.shutdown()


def test_mineru_shaped_sequential_executor_is_reusable() -> None:
    _ray_init_local()
    pipe: MineruShapedDummyPipeline | None = None
    try:
        pipe = MineruShapedDummyPipeline(replicas=1)
        batches = [["/x/a.pdf"], ["/x/b.pdf"]]
        executor = SequentialExecutor(pipe)
        first = executor.run(batches)
        second = executor.run(batches)
        assert first == second
    finally:
        if pipe is not None:
            _cleanup_modules(pipe.pdf2img, pipe.layout, pipe.ocr, pipe.img2md)
        if ray.is_initialized():
            ray.shutdown()
