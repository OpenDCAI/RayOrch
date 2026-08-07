"""Docling v3.6 graph structure and kernel-reuse regression tests."""

from __future__ import annotations

import json

import pytest

from rayorch.experimental.multigrain_v3.benchmark.document_docling.core_stages import (
    DoclingLayoutPages,
    DoclingOcrPages,
    ExpandDoclingPages,
    ReduceDoclingDocument,
)
from rayorch.experimental.multigrain_v3.benchmark.document_docling.core_table_workflow import (
    DoclingPostprocessPages,
    DoclingTableCore,
    ExpandDoclingTableJobs,
    ExpandDoclingTableV2Jobs,
    ReduceDoclingPage,
)
from rayorch.experimental.multigrain_v3.benchmark.document_docling.tableformer_v1_batch import (
    DoclingTableFormerV1BatchCore,
)
from rayorch.experimental.multigrain_v3.benchmark.document_docling.tableformer_v2_batch import (
    DoclingTableFormerV2BatchCore,
)
from rayorch.experimental.multigrain_v3_6.benchmark.document_docling import (
    DoclingTableFormerV1BatchV36Pipeline,
    DoclingTableFormerV2BatchV36Pipeline,
    DoclingTableJobV36Pipeline,
)
from rayorch.experimental.multigrain_v3_6.benchmark.document_docling import paired
from rayorch.experimental.multigrain_v3.benchmark.document_docling.core_compare import (
    CoreMatrixConfig,
)
from rayorch.experimental.multigrain_v3_6.logical import ExpandOrigin, ReduceOrigin


@pytest.mark.parametrize(
    ("pipeline_type", "prepare_target", "table_target"),
    [
        (DoclingTableJobV36Pipeline, ExpandDoclingTableJobs, DoclingTableCore),
        (
            DoclingTableFormerV1BatchV36Pipeline,
            ExpandDoclingTableJobs,
            DoclingTableFormerV1BatchCore,
        ),
        (
            DoclingTableFormerV2BatchV36Pipeline,
            ExpandDoclingTableV2Jobs,
            DoclingTableFormerV2BatchCore,
        ),
    ],
)
@pytest.mark.parametrize("optimize", [False, True])
def test_docling_v36_is_exactly_document_page_table_job_graph(
    pipeline_type,
    prepare_target,
    table_target,
    optimize,
):
    compiled = pipeline_type().compile(optimize=optimize)

    assert [spec.udf.target for spec in compiled.logical.calls.values()] == [
        ExpandDoclingPages,
        DoclingLayoutPages,
        DoclingOcrPages,
        DoclingPostprocessPages,
        prepare_target,
        table_target,
        ReduceDoclingPage,
        ReduceDoclingDocument,
    ]
    assert len(compiled.plan.actor_pools_by_call) == 8
    assert sum(
        isinstance(spec.origin, ExpandOrigin)
        for spec in compiled.logical.ports.values()
    ) == 2
    assert sum(
        isinstance(spec.origin, ReduceOrigin)
        for spec in compiled.logical.ports.values()
    ) == 2

    domains = list(compiled.logical.domains.values())
    assert len(domains) == 3
    assert domains[0].parent is None
    assert domains[1].parent == domains[0].ref
    assert domains[2].parent == domains[1].ref


def test_docling_v36_only_batch_scope_changes_cross_parent_pools():
    elastic = DoclingTableFormerV1BatchV36Pipeline(
        batch_scope="elastic"
    ).compile()
    parent = DoclingTableFormerV1BatchV36Pipeline(
        batch_scope="parent_bound"
    ).compile()

    assert elastic.logical.calls == parent.logical.calls
    changed = []
    for left, right in zip(
        elastic.plan.actor_pools_by_call.values(),
        parent.plan.actor_pools_by_call.values(),
    ):
        if left != right:
            changed.append((left, right))

    assert len(changed) == 5
    assert all(left.batch_scope == "elastic" for left, _ in changed)
    assert all(right.batch_scope == "parent_bound" for _, right in changed)


@pytest.mark.parametrize(
    ("option", "message"),
    [
        ({"batch_scope": "unknown"}, "batch_scope"),
        ({"ocr_batch_mode": "unknown"}, "ocr_batch_mode"),
        ({"table_batch_mode": "unknown"}, "table_batch_mode"),
        ({"table_batch_max_jobs": 0}, "table_batch_max_jobs"),
        ({"infra_retries": -1}, "infra_retries"),
    ],
)
def test_docling_v36_rejects_unsupported_or_invalid_options(option, message):
    with pytest.raises(ValueError, match=message):
        DoclingTableJobV36Pipeline(**option)


def test_docling_paired_projection_drops_only_explicit_v3_scheduler_options():
    config = CoreMatrixConfig(
        device="cuda",
        ocr_device="cpu",
        stage_batch_size=16,
        table_core_batch_size=4,
        parse_replicas=16,
        layout_replicas=2,
        ocr_replicas=6,
        table_replicas=2,
        reduce_replicas=4,
        layout_num_gpus=1,
        table_num_gpus=1,
        ocr_batch_mode="recognition_accelerated",
        table_batch_mode="v1_batch",
        max_pending_per_actor=1,
        microbatch_size=24,
        max_inflight_arenas=4,
    )

    old, new = paired._arm_options(368, config, "elastic")

    assert set(old) - set(new) == paired._V3_ONLY_OPTIONS
    assert new["batch_scope"] == "elastic"
    assert new["table_batch_mode"] == "v1_batch"
    assert new["table_core_batch_size"] == 4
    assert new["reduce_replicas"] == 4
    DoclingTableFormerV1BatchV36Pipeline(**new).compile()


def test_docling_paired_correctness_reports_exact_structure_and_jaccard():
    old = [
        {
            "pages": 1,
            "texts": 2,
            "tables": 1,
            "pictures": 0,
            "markdown": "hello stable table",
        }
    ]
    new = [{**old[0], "markdown": "hello stable table extra"}]

    comparison = paired._compare_documents(old, new)

    assert comparison["document_count_matches"]
    assert comparison["identity_exact"] == (True,)
    assert comparison["structure_exact"] == (True,)
    assert comparison["markdown_exact"] == (False,)
    assert comparison["markdown_jaccard"] == (0.75,)


def test_docling_audit_gate_only_rejects_nonzero_error_or_fallback_counts():
    summary = {
        "Ocr": {
            "audit": {
                "jobs": 10.0,
                "batch_errors": 0.0,
                "rect_fallbacks": 2.0,
            }
        },
        "Table": {"audit": {"serial_fallback_jobs": 0.0}},
    }

    assert paired._audit_violations(summary) == ("Ocr.rect_fallbacks=2.0",)
    assert paired._v3_audit_violations(
        {
            "actor_audit_stage_ocr_jobs": 10,
            "actor_audit_stage_ocr_batch_errors": 0,
            "actor_audit_stage_table_serial_fallback_jobs": 3,
        }
    ) == ("actor_audit_stage_table_serial_fallback_jobs=3",)


def test_docling_paired_cli_defaults_to_promoted_full_gate_configuration():
    args = paired.build_parser().parse_args(["--manifest", "docs.json"])

    assert args.limit == 4
    assert args.table_batch_mode == "v1_batch"
    assert args.microbatch_size == 24
    assert args.max_active_microbatches == 4
    assert (
        args.parse_replicas,
        args.layout_replicas,
        args.ocr_replicas,
        args.table_replicas,
        args.reduce_replicas,
    ) == (16, 2, 6, 2, 4)


def test_docling_manifest_and_each_arm_have_independent_identity_gates(
    tmp_path,
):
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {"path": str(first), "pages": 2, "size_bytes": 5},
                {"path": str(second), "pages": 3, "size_bytes": 6},
            ]
        )
    )

    manifest = paired._read_manifest(str(manifest_path), limit=0)

    assert manifest.paths == (str(first), str(second))
    assert manifest.pdf_ids == ("first", "second")
    assert manifest.expected_pages == 5
    assert manifest.input_bytes == 11
    documents = (
        {"pdf": "first", "pages": 2, "tables": 1},
        {"pdf": "second", "pages": 3, "tables": 2},
    )
    assert paired._validate_arm_outputs(
        "v36",
        documents,
        manifest,
        expected_tables=3,
    ) == {"documents": 2, "pages": 5, "tables": 3}

    with pytest.raises(ValueError, match="identity/order"):
        paired._validate_arm_outputs(
            "v36",
            tuple(reversed(documents)),
            manifest,
            expected_tables=3,
        )
    with pytest.raises(ValueError, match="golden requires 4"):
        paired._validate_arm_outputs(
            "v36",
            documents,
            manifest,
            expected_tables=4,
        )


def test_docling_manifest_rejects_partial_or_stale_metadata(tmp_path):
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {"path": str(first), "pages": 2, "size_bytes": 5},
                {"path": str(second), "size_bytes": 6},
            ]
        )
    )

    with pytest.raises(ValueError, match="partially populated"):
        paired._read_manifest(str(manifest_path), limit=0)

    manifest_path.write_text(
        json.dumps([{"path": str(first), "pages": 2, "size_bytes": 99}])
    )
    with pytest.raises(ValueError, match="size mismatch"):
        paired._read_manifest(str(manifest_path), limit=0)

    manifest_path.write_text(json.dumps([str(first), {"path": str(second)}]))
    with pytest.raises(ValueError, match="must not mix"):
        paired._read_manifest(str(manifest_path), limit=0)
