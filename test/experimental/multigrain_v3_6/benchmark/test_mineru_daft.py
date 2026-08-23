"""Optional configuration tests for the Daft MinerU paper baseline."""

from __future__ import annotations

import argparse

import pytest

pytest.importorskip("daft")

from rayorch.experimental.multigrain_v3_6.benchmark.mineru_daft import (
    _ContentPayload,
    _PagePayload,
    _build_dataframe,
    _effective_source_partitions,
    _explain,
    _package_ocr_batch,
    build_parser,
)


def _args(*extra: str) -> argparse.Namespace:
    return build_parser().parse_args(
        [
            "--output-dir",
            "/tmp/out",
            "--artifact-dir",
            "/tmp/artifacts",
            "--result-jsonl",
            "/tmp/results.jsonl",
            *extra,
        ]
    )


def test_daft_source_partitions_default_to_render_replicas():
    assert _effective_source_partitions(_args(), source_count=48) == 4


def test_daft_source_partitions_are_independent_and_bounded():
    assert _effective_source_partitions(
        _args("--partitions", "16"), source_count=48
    ) == 16
    assert _effective_source_partitions(
        _args("--partitions", "16"), source_count=4
    ) == 4


def test_daft_source_partitions_must_be_positive():
    with pytest.raises(ValueError, match="partitions must be positive"):
        _effective_source_partitions(
            _args("--partitions", "0"), source_count=48
        )


def test_daft_plan_has_group_regroup_without_global_output_sort():
    args = _args("--smoke-no-model", "--partitions", "2")

    plan = _explain(_build_dataframe(args, ["a.pdf", "b.pdf"]))
    physical = plan.split("== Physical Plan ==", maxsplit=1)[1]

    assert "GroupedAggregate" in physical
    assert "* Sort" not in physical


def test_daft_control_modes_are_explicit_and_default_to_natural_baseline():
    natural = _args()
    reference = _args(
        "--regroup-mode",
        "reference_only",
        "--assemble-mode",
        "metadata_only",
    )

    assert natural.regroup_mode == "full_value"
    assert natural.assemble_mode == "full"
    assert reference.regroup_mode == "reference_only"
    assert reference.assemble_mode == "metadata_only"


def test_full_value_packaging_preserves_one_payload_per_input_row():
    pages = [
        _PagePayload({"page_id": 0}),
        _PagePayload({"page_id": 1}),
    ]

    packaged = _package_ocr_batch(
        pages=pages,
        contents=["left", "right"],
        parent_ids=[0, 0],
        stores=(),
        batch_token="batch",
        actor_init_started_s=1.0,
        actor_ready_s=2.0,
        batch_started_s=3.0,
        batch_finished_s=4.0,
    )

    assert all(isinstance(value, _ContentPayload) for value in packaged)
    assert [value.value for value in packaged] == ["left", "right"]
    assert [value.batch_size for value in packaged] == [2, 2]


def _group_input_passthrough(plan: str) -> str:
    physical = plan.split("== Physical Plan ==", maxsplit=1)[1]
    return next(
        line for line in physical.splitlines()
        if "Passthrough Columns" in line
    )


def test_reference_control_projects_page_out_before_hash_regroup():
    pdfs = ["a.pdf", "b.pdf"]
    natural = _explain(
        _build_dataframe(_args("--smoke-no-model"), pdfs)
    )
    reference = _explain(
        _build_dataframe(
            _args(
                "--smoke-no-model",
                "--regroup-mode",
                "reference_only",
            ),
            pdfs,
            stores=(object(),),
        )
    )

    assert "page" in _group_input_passthrough(natural)
    assert "page" not in _group_input_passthrough(reference)
