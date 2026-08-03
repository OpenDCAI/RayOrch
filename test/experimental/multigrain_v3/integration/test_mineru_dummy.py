"""CPU integration coverage for the V3 MinerU-shaped dummy workload."""

from __future__ import annotations

import pytest

from rayorch.experimental.multigrain_v3.benchmark.mineru import (
    DummyAssembleDoc,
    DummyDocument,
    DummyOcrPage,
    DummyPdfToPages,
    MinerUDummyPipeline,
    compile_pipeline,
    execute_pipeline,
    make_dummy_pdfs,
)


@pytest.mark.cpu
def test_dummy_reference_flow_handles_multi_pdf_variable_fanout():
    """The dependency-free UDFs provide an always-runnable semantic oracle."""

    pdfs = make_dummy_pdfs(4, page_counts=(3, 1, 4, 2))
    page_groups = DummyPdfToPages().run(pdfs)

    # Complete physical OCR chunks in reverse order, then reconstruct the
    # logical reduce groups by document/page identity.
    flat_pages = [page for group in page_groups for page in group]
    chunks = [flat_pages[index : index + 2] for index in range(0, 10, 2)]
    completed = [
        content
        for chunk in reversed(chunks)
        for content in DummyOcrPage().run(chunk)
    ]
    by_identity = {
        (content.document_id, content.page_id): content
        for content in completed
    }
    ordered_groups = [
        [
            by_identity[(page.document_id, page.page_id)]
            for page in page_group
        ]
        for page_group in page_groups
    ]
    documents = DummyAssembleDoc().run(
        pdfs,
        ordered_groups,
        page_groups,
    )
    assert [document.page_ids for document in documents] == [
        (0, 1, 2),
        (0,),
        (0, 1, 2, 3),
        (0, 1),
    ]


@pytest.mark.cpu
@pytest.mark.usefixtures("ray_cluster")
def test_dummy_pipeline_executes_cross_pdf_ordered_reduce_on_cpu():
    """ExecutorSession should preserve ordinals despite cross-root batching."""

    pdfs = make_dummy_pdfs(5, page_counts=(3, 0, 1, 4, 2))
    pipeline = MinerUDummyPipeline(
        replicas=2,
        batch_size=2,
        render_replicas=1,
        render_batch_size=3,
        assemble_replicas=1,
        assemble_batch_size=4,
        delay_scale_s=0.001,
    )
    graph = compile_pipeline(pipeline)
    documents = execute_pipeline(pipeline, pdfs)

    assert documents == [
        DummyDocument(
            document_id="dummy-000",
            page_ids=(0, 1, 2),
            markdown="\n".join(
                f"text(rendered:dummy-000:{page_id})"
                for page_id in range(3)
            ),
        ),
        DummyDocument(
            document_id="dummy-001",
            page_ids=(),
            markdown="",
        ),
        DummyDocument(
            document_id="dummy-002",
            page_ids=(0,),
            markdown="text(rendered:dummy-002:0)",
        ),
        DummyDocument(
            document_id="dummy-003",
            page_ids=(0, 1, 2, 3),
            markdown="\n".join(
                f"text(rendered:dummy-003:{page_id})"
                for page_id in range(4)
            ),
        ),
        DummyDocument(
            document_id="dummy-004",
            page_ids=(0, 1),
            markdown="\n".join(
                f"text(rendered:dummy-004:{page_id})"
                for page_id in range(2)
            ),
        ),
    ]
