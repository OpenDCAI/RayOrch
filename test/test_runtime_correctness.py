from __future__ import annotations

import pytest
import ray

from rayorch import DagPipeline, PipeRef, RuntimeDagExecutor, RuntimeRayModule
from rayorch.runtime import MicroBatch, RuntimeNodeSpec, RuntimeResult

from test.runtime_test_utils import (
    FaultPipe,
    Pdf2ImgOp,
    RuntimeMineruPipe,
    cleanup_modules,
    cleanup_pipeline,
    trace_path,
)


def test_runtime_module_shards_microbatch_across_replicas() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=8)
    module = RuntimeRayModule(
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
        source = MicroBatch.source(
            {
                "pdf": pdfs,
                "meta": [{"name": f"doc_{i}"} for i in range(16)],
            },
            dataset="runtime-module-shard",
        )

        result = module(source)

        assert isinstance(result, RuntimeResult)
        assert len(result.batch) == 15
        assert result.batch.row_ids == [
            row_id for i, row_id in enumerate(source.row_ids) if i != 5
        ]
        assert [record.values["pdf"] for record in result.quarantined] == [
            "doc_bad.pdf"
        ]
        assert trace_path(result.paths, result.batch.path_ids[0]) == ["pdf2img"]
    finally:
        cleanup_modules(module)
        ray.shutdown()


def test_runtime_module_can_bind_spec_after_pre_init() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=4)
    module = RuntimeRayModule(Pdf2ImgOp, replicas=2, max_inflight=2).pre_init()
    module.bind_runtime_spec(
        RuntimeNodeSpec(
            node="pdf2img",
            inputs=("pdf", "meta"),
            outputs=("images", "meta"),
        )
    )
    try:
        source = MicroBatch.source(
            {
                "pdf": ["doc_0.pdf", "doc_bad.pdf", "doc_2.pdf"],
                "meta": [{"name": "doc_0"}, {"name": "bad"}, {"name": "doc_2"}],
            },
            dataset="runtime-module-bind",
        )
        result = module(source)
        assert len(result.batch) == 2
        assert [record.values["pdf"] for record in result.quarantined] == [
            "doc_bad.pdf"
        ]
    finally:
        cleanup_modules(module)
        ray.shutdown()


def test_runtime_module_requires_spec_before_direct_call() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=2)
    module = RuntimeRayModule(Pdf2ImgOp, replicas=1).pre_init()
    try:
        source = MicroBatch.source(
            {"pdf": ["doc_0.pdf"], "meta": [{"name": "doc_0"}]},
            dataset="runtime-module-unbound",
        )
        with pytest.raises(RuntimeError, match="requires a RuntimeNodeSpec"):
            module(source)
    finally:
        cleanup_modules(module)
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


def test_runtime_dag_executor_runs_flash_mineru_like_pipeline() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=16)
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

        result = RuntimeDagExecutor(pipe).run(source)

        assert len(result.batch) == 15
        assert result.batch.columns["markdown"][0] == "dag_0.md pages=2 blocks=2"
        assert result.batch.columns["markdown"][-1] == "dag_15.md pages=2 blocks=2"
        assert [record.values["pdf"] for record in result.quarantined] == [
            "dag_bad.pdf"
        ]
        assert trace_path(result.paths, result.batch.path_ids[0]) == [
            "pdf2img",
            "layout",
            "ocr",
            "convert",
        ]
    finally:
        cleanup_pipeline(pipe)
        ray.shutdown()


def test_runtime_dag_executor_localizes_errors_with_batches_inflight() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=16)
    pipe = FaultPipe()

    def make_batch(index: int, pdfs: list[str], names: list[str]) -> MicroBatch:
        return MicroBatch.source(
            {"pdf": pdfs, "meta": [{"name": name} for name in names]},
            dataset=f"fault-overlap-{index}",
        )

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

        results = RuntimeDagExecutor(pipe, max_batches_inflight=4).run(batches)

        assert [len(result.batch) for result in results] == [4, 3, 3, 2]
        assert [len(result.quarantined) for result in results] == [0, 1, 1, 2]
        assert [(r.op, r.values["pdf"]) for r in results[1].quarantined] == [
            ("pdf2img", "b1_bad_pdf.pdf")
        ]
        assert [(r.op, r.values["meta"]["name"]) for r in results[2].quarantined] == [
            ("ocr", "b2_bad_ocr")
        ]
        b3_by_op = {record.op: record for record in results[3].quarantined}
        assert trace_path(results[3].paths, b3_by_op["pdf2img"].path_id) == []
        assert trace_path(results[3].paths, b3_by_op["ocr"].path_id) == [
            "pdf2img",
            "layout",
        ]
    finally:
        cleanup_pipeline(pipe)
        ray.shutdown()


def test_runtime_dag_executor_wraps_column_data_into_microbatches() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=16)
    pipe = RuntimeMineruPipe()
    try:
        pdfs = [f"column_{i}.pdf" for i in range(10)]
        meta = [{"name": f"column_{i}"} for i in range(10)]

        results = RuntimeDagExecutor(
            pipe,
            batch_size=4,
            max_batches_inflight=3,
            dataset="column-dataset",
        ).run(pdf=pdfs, meta=meta)

        assert isinstance(results, list)
        assert [len(result.batch) for result in results] == [4, 4, 2]
        assert [
            row_id
            for result in results
            for row_id in result.batch.row_ids
        ] == [f"column-dataset:{index}" for index in range(10)]
        assert [
            markdown
            for result in results
            for markdown in result.batch.columns["markdown"]
        ] == [f"column_{index}.md pages=2 blocks=2" for index in range(10)]
    finally:
        cleanup_pipeline(pipe)
        ray.shutdown()


def test_runtime_dag_executor_rejects_invalid_column_data() -> None:
    pipe = RuntimeMineruPipe()
    try:
        with pytest.raises(ValueError, match="batch_size is required"):
            RuntimeDagExecutor(pipe).run(pdf=["a.pdf"], meta=[{"name": "a"}])

        with pytest.raises(ValueError, match="column lengths must match"):
            RuntimeDagExecutor(pipe, batch_size=2).run(
                {"pdf": ["a.pdf", "b.pdf"], "meta": [{"name": "a"}]}
            )
    finally:
        cleanup_pipeline(pipe)


def test_runtime_module_handles_varied_microbatch_sizes() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=8)
    module = RuntimeRayModule(
        Pdf2ImgOp,
        inputs=("pdf", "meta"),
        outputs=("images", "meta"),
        op="pdf2img",
        replicas=4,
        max_inflight=4,
    ).pre_init()
    try:
        for n, bad_idx in [(1, None), (2, 0), (3, 2), (4, 1), (5, 4), (16, 7)]:
            pdfs = [f"case_{n}_{i}.pdf" for i in range(n)]
            if bad_idx is not None:
                pdfs[bad_idx] = f"case_{n}_bad.pdf"
            source = MicroBatch.source(
                {
                    "pdf": pdfs,
                    "meta": [{"name": f"case_{n}_{i}"} for i in range(n)],
                },
                dataset=f"varied-{n}",
            )
            result = module(source)
            assert len(result.batch) == n - (bad_idx is not None)
            assert len(result.quarantined) == (bad_idx is not None)
    finally:
        cleanup_modules(module)
        ray.shutdown()
