"""Docling core-stage adapter 的 Ray-free 快速测试。"""

from __future__ import annotations

import pytest

from rayorch.experimental.multigrain_v3.benchmark.document_docling.core_stages import (
    _group_indices_by_document,
)
from rayorch.experimental.multigrain_v3.benchmark.document_docling.core_table_workflow import (
    DoclingTableCore,
    ExpandDoclingTableJobs,
    ExpandDoclingTableV2Jobs,
)
from rayorch.experimental.multigrain_v3.benchmark.document_docling.core_v3 import (
    DoclingCoreV3Pipeline,
)


from rayorch.experimental.multigrain_v3.benchmark.document_docling.core_values import (
    DetachedDoclingPageBackend,
    DoclingPageSource,
)


def test_group_indices_preserves_first_document_order() -> None:
    """跨 document batch 内的 context 分组必须稳定。"""

    sources = [
        DoclingPageSource("a.pdf", "a", 1, None, None, {}),
        DoclingPageSource("b.pdf", "b", 1, None, None, {}),
        DoclingPageSource("a.pdf", "a", 2, None, None, {}),
    ]

    groups = _group_indices_by_document(sources)

    assert [path for path, _, _ in groups] == ["a.pdf", "b.pdf"]
    assert [indices for _, _, indices in groups] == [[0, 2], [1]]


def test_core_pipeline_compiles_with_parent_bound_and_elastic() -> None:
    """核心模型 pipeline 应只通过普通 V3 原语表达调度模式。"""

    for mode in ("elastic", "parent_bound"):
        compiled = DoclingCoreV3Pipeline(batch_scope=mode).compile()
        assert [stage.kind.value for stage in compiled.dag.stages] == [
            "source",
            "expand",
            "map",
            "map",
            "map",
            "expand",
            "map",
            "reduce",
            "reduce",
        ]
        assert compiled.dag.stages[7].reduce.scope_path == (5,)


def test_table_core_physical_batch_is_independent() -> None:
    """只缩小 TableCore RPC，不改变上游 Page stage 的 batch cap。"""

    compiled = DoclingCoreV3Pipeline(
        table_batch_size=16,
        table_core_batch_size=4,
    ).compile()

    assert compiled.dag.stages[4].execution.batch_size == 16
    assert compiled.dag.stages[5].execution.batch_size == 16
    assert compiled.dag.stages[6].execution.batch_size == 4
    assert compiled.dag.stages[7].execution.batch_size == 16


@pytest.mark.parametrize(
    "stage_type",
    (ExpandDoclingTableJobs, ExpandDoclingTableV2Jobs),
)
def test_no_table_page_skips_render_and_emits_empty_child_group(
    monkeypatch,
    stage_type,
) -> None:
    """无表页必须保留 Page lineage，但不能触发 2x page render。"""

    from types import SimpleNamespace

    def unexpected_render(*args, **kwargs):
        raise AssertionError("no-table page must not be rendered")

    monkeypatch.setattr(
        "rayorch.experimental.multigrain_v3.benchmark.document_docling."
        "core_table_workflow.page_from_source",
        unexpected_render,
    )
    stage = stage_type()
    page = SimpleNamespace(
        source=SimpleNamespace(),
        layout=SimpleNamespace(clusters=[]),
        ocr=SimpleNamespace(),
    )

    assert stage.run([page]) == [[]]
    assert stage.batch_audit() == {
        "pages": 1,
        "table_pages": 0,
        "skipped_pages": 1,
        "jobs": 0,
    }


def test_decoder_batch_mode_is_opt_in_and_fail_fast(monkeypatch) -> None:
    """decoder batch 不允许静默走 serial reference，避免伪加速。"""

    pipeline = DoclingCoreV3Pipeline(table_batch_mode="decoder_accelerated")
    assert pipeline.compile().dag.stages[6].udf.target is DoclingTableCore
    with pytest.raises(ValueError, match="table_actor_concurrency"):
        DoclingCoreV3Pipeline(
            table_batch_mode="decoder_accelerated",
            table_actor_concurrency=2,
        )

    core = object.__new__(DoclingTableCore)
    core.table_batch_mode = "decoder_accelerated"
    core.table_batch_max_jobs = 16
    core.total_table_batch_audit = {
        "jobs": 0,
        "batches": 0,
        "batch_errors": 0,
        "serial_fallback_jobs": 0,
    }

    def fail_decoder(*args, **kwargs):
        raise RuntimeError("decoder failed")

    monkeypatch.setattr(
        "rayorch.experimental.multigrain_v3.benchmark.document_docling."
        "table_decoder_adapter_v2.run_decoder_batched",
        fail_decoder,
    )
    with pytest.raises(RuntimeError, match="decoder failed"):
        core.run([object()])
    assert core.batch_audit()["batch_errors"] == 1
    assert core.batch_audit()["serial_fallback_jobs"] == 0


def test_detached_backend_is_pickleable_and_crops() -> None:
    """detached backend 必须跨 Ray 序列化并保留 image crop 能力。"""

    import pickle
    from io import BytesIO

    from PIL import Image

    docling_core = pytest.importorskip("docling_core")
    del docling_core
    from docling_core.types.doc import BoundingBox, CoordOrigin, Size
    from docling_core.types.doc.page import (
        BoundingRectangle,
        PdfPageBoundaryType,
        PdfPageGeometry,
        SegmentedPdfPage,
    )
    bbox = BoundingBox(
        l=0,
        t=0,
        r=10,
        b=8,
        coord_origin=CoordOrigin.TOPLEFT,
    )

    image = Image.new("RGB", (10, 8), "white")
    stream = BytesIO()
    image.save(stream, format="PNG")
    source = DoclingPageSource(
        document_path="doc.pdf",
        document_hash="a" * 64,
        page_no=1,
        size=Size(width=10, height=8),
        segmented_page=SegmentedPdfPage(
            dimension=PdfPageGeometry(
                angle=0,
                rect=BoundingRectangle(
                    r_x0=0,
                    r_y0=0,
                    r_x1=10,
                    r_y1=0,
                    r_x2=10,
                    r_y2=8,
                    r_x3=0,
                    r_y3=8,
                    coord_origin=CoordOrigin.TOPLEFT,
                ),
                boundary_type=PdfPageBoundaryType.CROP_BOX,
                art_bbox=bbox,
                bleed_bbox=bbox,
                crop_bbox=bbox,
                media_bbox=bbox,
                trim_bbox=bbox,
            ),
            char_cells=[],
            word_cells=[],
            textline_cells=[],
        ),
        images_by_scale={1.0: stream.getvalue()},
    )
    backend = DetachedDoclingPageBackend(source)

    restored = pickle.loads(pickle.dumps(backend))

    assert restored.page_no == 1
    assert restored.get_page_image(scale=1.0).size == (10, 8)
