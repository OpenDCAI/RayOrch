from __future__ import annotations

import pytest

from rayorch import RuntimeDagExecutor, RuntimeRayModule
from rayorch.runtime import MicroBatch, RuntimeResult

from test.runtime.helpers import (
    FaultPipe,
    Pdf2ImgOp,
    RuntimeMineruPipe,
    trace_path,
)

pytestmark = pytest.mark.usefixtures("ray_cluster")


def test_module_shards_rows_across_replicas_and_merges_results() -> None:
    module = RuntimeRayModule(
        Pdf2ImgOp,
        inputs=("pdf", "meta"),
        outputs=("images", "meta"),
        op="pdf2img",
        replicas=4,
        max_inflight=4,
    ).start()
    try:
        pdfs = [f"doc_{i}.pdf" for i in range(8)]
        pdfs[3] = "doc_bad.pdf"
        source = MicroBatch.source(
            {
                "pdf": pdfs,
                "meta": [{"name": f"doc_{i}"} for i in range(8)],
            },
            dataset="shard",
        )

        result = module(source)

        assert isinstance(result, RuntimeResult)
        assert len(result.batch) == 7
        assert [record.values["pdf"] for record in result.quarantined] == [
            "doc_bad.pdf"
        ]
        assert trace_path(result.paths, result.batch.path_ids[0]) == ["pdf2img"]
    finally:
        module.close()


def test_executor_runs_flash_mineru_pipeline_and_isolates_bad_pdf() -> None:
    pipe = RuntimeMineruPipe(
        pdf_replicas=2,
        layout_replicas=2,
        ocr_replicas=2,
        output_replicas=1,
        max_inflight=2,
    )
    source = MicroBatch.source(
        {
            "pdf": ["a.pdf", "bad.pdf", "c.pdf"],
            "meta": [{"name": "a"}, {"name": "bad"}, {"name": "c"}],
        },
        dataset="mineru",
    )

    with RuntimeDagExecutor(pipe) as executor:
        result = executor.run(source)

    assert result.batch.columns["markdown"] == [
        "a.md pages=2 blocks=2",
        "c.md pages=2 blocks=2",
    ]
    assert [record.values["pdf"] for record in result.quarantined] == ["bad.pdf"]
    assert trace_path(result.paths, result.batch.path_ids[0]) == [
        "pdf2img",
        "layout",
        "ocr",
        "convert",
    ]


def test_executor_localizes_errors_per_inflight_microbatch() -> None:
    pipe = FaultPipe()
    batches = [
        MicroBatch.source(
            {
                "pdf": [f"b{batch}_doc0.pdf", pdf],
                "meta": [{"name": f"b{batch}_doc0"}, {"name": name}],
            },
            dataset=f"batch-{batch}",
        )
        for batch, pdf, name in [
            (0, "b0_doc1.pdf", "b0_doc1"),
            (1, "b1_bad_pdf.pdf", "b1_bad_pdf"),
            (2, "b2_doc1.pdf", "b2_bad_ocr"),
        ]
    ]

    with RuntimeDagExecutor(pipe, max_batches_inflight=3) as executor:
        results = executor.run(batches)

    assert [len(result.quarantined) for result in results] == [0, 1, 1]
    assert results[1].quarantined[0].op == "pdf2img"
    assert results[2].quarantined[0].op == "ocr"
    assert trace_path(
        results[2].paths, results[2].quarantined[0].path_id
    ) == ["pdf2img", "layout"]


def test_column_input_is_split_into_global_row_id_microbatches() -> None:
    pipe = RuntimeMineruPipe(
        pdf_replicas=1,
        layout_replicas=1,
        ocr_replicas=1,
        output_replicas=1,
    )
    pdfs = [f"column_{i}.pdf" for i in range(5)]
    meta = [{"name": f"column_{i}"} for i in range(5)]

    with RuntimeDagExecutor(
        pipe,
        batch_size=2,
        max_batches_inflight=2,
        dataset="columns",
    ) as executor:
        results = executor.run(pdf=pdfs, meta=meta)

    assert [len(result.batch) for result in results] == [2, 2, 1]
    assert [
        row_id
        for result in results
        for row_id in result.batch.row_ids
    ] == [f"columns:{index}" for index in range(5)]


def test_column_input_validates_batch_size_and_column_lengths() -> None:
    pipe = RuntimeMineruPipe(
        pdf_replicas=1,
        layout_replicas=1,
        ocr_replicas=1,
        output_replicas=1,
    )
    with RuntimeDagExecutor(pipe) as executor:
        with pytest.raises(ValueError, match="batch_size is required"):
            executor.run(pdf=["a.pdf"], meta=[{"name": "a"}])

    with RuntimeDagExecutor(pipe, batch_size=2) as executor:
        with pytest.raises(ValueError, match="column lengths must match"):
            executor.run(
                {"pdf": ["a.pdf", "b.pdf"], "meta": [{"name": "a"}]}
            )
