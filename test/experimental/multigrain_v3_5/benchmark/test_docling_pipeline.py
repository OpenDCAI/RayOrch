"""Docling v3.5 graph structure and kernel-reuse regression tests."""

from __future__ import annotations

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
from rayorch.experimental.multigrain_v3_5.benchmark.document_docling import (
    DoclingTableFormerV1BatchV35Pipeline,
    DoclingTableFormerV2BatchV35Pipeline,
    DoclingTableJobV35Pipeline,
)
from rayorch.experimental.multigrain_v3_5.benchmark.document_docling import paired
from rayorch.experimental.multigrain_v3.benchmark.document_docling.core_compare import (
    CoreMatrixConfig,
)
from rayorch.experimental.multigrain_v3_5.logical import ExpandOrigin, GroupOrigin


@pytest.mark.parametrize(
    ("pipeline_type", "prepare_target", "table_target"),
    [
        (DoclingTableJobV35Pipeline, ExpandDoclingTableJobs, DoclingTableCore),
        (
            DoclingTableFormerV1BatchV35Pipeline,
            ExpandDoclingTableJobs,
            DoclingTableFormerV1BatchCore,
        ),
        (
            DoclingTableFormerV2BatchV35Pipeline,
            ExpandDoclingTableV2Jobs,
            DoclingTableFormerV2BatchCore,
        ),
    ],
)
@pytest.mark.parametrize("optimize", [False, True])
def test_docling_v35_is_exactly_document_page_table_job_graph(
    pipeline_type,
    prepare_target,
    table_target,
    optimize,
):
    compiled = pipeline_type().compile(optimize=optimize)

    assert [spec.kernel.target for spec in compiled.logical.calls.values()] == [
        ExpandDoclingPages,
        DoclingLayoutPages,
        DoclingOcrPages,
        DoclingPostprocessPages,
        prepare_target,
        table_target,
        ReduceDoclingPage,
        ReduceDoclingDocument,
    ]
    assert len(compiled.runtime.pools) == 8
    assert sum(
        isinstance(spec.origin, ExpandOrigin)
        for spec in compiled.logical.ports.values()
    ) == 2
    assert sum(
        isinstance(spec.origin, GroupOrigin)
        for spec in compiled.logical.ports.values()
    ) == 2

    domains = list(compiled.logical.domains.values())
    assert len(domains) == 3
    assert domains[0].parent is None
    assert domains[1].parent == domains[0].ref
    assert domains[2].parent == domains[1].ref


def test_docling_v35_only_batch_scope_changes_cross_parent_pools():
    elastic = DoclingTableFormerV1BatchV35Pipeline(
        batch_scope="elastic"
    ).compile()
    parent = DoclingTableFormerV1BatchV35Pipeline(
        batch_scope="parent_bound"
    ).compile()

    assert elastic.logical.calls == parent.logical.calls
    changed = []
    for left, right in zip(
        elastic.runtime.pools.values(),
        parent.runtime.pools.values(),
    ):
        left_options = dict(left.options)
        right_options = dict(right.options)
        if left_options != right_options:
            changed.append((left_options, right_options))

    assert len(changed) == 5
    assert all(left["batch_scope"] == "elastic" for left, _ in changed)
    assert all(right["batch_scope"] == "parent_bound" for _, right in changed)


@pytest.mark.parametrize(
    ("option", "message"),
    [
        ({"batch_scope": "unknown"}, "batch_scope"),
        ({"ocr_batch_mode": "unknown"}, "ocr_batch_mode"),
        ({"table_batch_mode": "unknown"}, "table_batch_mode"),
        ({"table_batch_max_jobs": 0}, "table_batch_max_jobs"),
        ({"max_retries": -1}, "max_retries"),
    ],
)
def test_docling_v35_rejects_unsupported_or_invalid_options(option, message):
    with pytest.raises(ValueError, match=message):
        DoclingTableJobV35Pipeline(**option)


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
    DoclingTableFormerV1BatchV35Pipeline(**new).compile()


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
    assert args.arena_size == 24
    assert args.max_in_flight == 4
    assert (
        args.parse_replicas,
        args.layout_replicas,
        args.ocr_replicas,
        args.table_replicas,
        args.reduce_replicas,
    ) == (16, 2, 6, 2, 4)
