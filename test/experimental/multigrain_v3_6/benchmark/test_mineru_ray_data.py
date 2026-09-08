"""Ray-free contract tests for the V3.6 Ray Data MinerU experiment."""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from rayorch.experimental.multigrain_v3_6.benchmark.mineru_ray_data import (
    _DocumentObservation,
    _TimedAssemblePdf,
    _batch_observations,
    _compact_payload,
    _decode_document,
    _ordered_rows,
    _ray_data_shuffle_observation,
    _ray_data_ocr_remote_tasks,
    _source_blocks,
    build_dataset,
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


def test_control_modes_are_explicit_and_natural_is_default():
    natural = _args()
    reference = _args("--regroup-mode", "reference_only")

    assert natural.regroup_mode == "full_value"
    assert natural.assemble_mode == "full"
    assert natural.poison_count == 0
    assert natural.poison_policy == "skip_page"
    assert natural.poison_pdf_index is None
    assert natural.input_manifest is None
    assert _args("--poison-policy", "drop_parent").poison_policy == "drop_parent"
    assert reference.regroup_mode == "reference_only"


def test_source_blocks_default_to_render_replicas():
    assert _source_blocks(_args(), source_count=48) == 4


def test_source_blocks_are_independent_bounded_and_positive():
    assert _source_blocks(_args("--source-blocks", "16"), 48) == 16
    assert _source_blocks(_args("--source-blocks", "16"), 4) == 4
    with pytest.raises(ValueError, match="source_blocks must be positive"):
        _source_blocks(_args("--source-blocks", "0"), 48)


def test_reference_mode_requires_stores_before_building_a_plan():
    with pytest.raises(ValueError, match="requires payload stores"):
        build_dataset(
            _args("--regroup-mode", "reference_only"),
            ["a.pdf"],
        )


def test_ordered_rows_restores_ordinals_and_rejects_bad_groups():
    batch = {
        "parent_id": np.asarray([3, 3]),
        "page_ordinal": np.asarray([1, 0]),
        "pdf_path": np.asarray(["a.pdf", "a.pdf"]),
    }
    assert [row["page_ordinal"] for row in _ordered_rows(batch)] == [0, 1]

    duplicate = {**batch, "page_ordinal": np.asarray([0, 0])}
    with pytest.raises(ValueError, match="non-contiguous or duplicate"):
        _ordered_rows(duplicate)

    mixed = {**batch, "pdf_path": np.asarray(["a.pdf", "b.pdf"])}
    with pytest.raises(ValueError, match="mixes multiple PDFs"):
        _ordered_rows(mixed)


def _observation(*, token: str, batch_size: int) -> _DocumentObservation:
    return _DocumentObservation(
        value={"pages": 1},
        page_ids=(0,),
        batch_tokens=(token,),
        batch_sizes=(batch_size,),
        render_started_s=(1.0,),
        render_finished_s=(2.0,),
        actor_init_started_s=(3.0,),
        actor_ready_s=(4.0,),
        batch_started_s=(5.0,),
        batch_finished_s=(6.0,),
        payload_publish_started_s=(7.0,),
        payload_publish_finished_s=(8.0,),
        payload_block_bytes=(9,),
        assemble_started_s=10.0,
        assemble_finished_s=11.0,
    )


def test_batch_observations_deduplicate_repeated_physical_batch():
    first = _observation(token="batch", batch_size=2)
    second = _observation(token="batch", batch_size=2)

    assert list(_batch_observations([first, second])) == ["batch"]

    conflict = _observation(token="batch", batch_size=3)
    with pytest.raises(ValueError, match="conflicting OCR observation"):
        _batch_observations([first, conflict])


def test_full_value_assembly_restores_business_page_order():
    count = 2
    batch = {
        "parent_id": np.asarray([7, 7]),
        "page_ordinal": np.asarray([1, 0]),
        "pdf_path": np.asarray(["a.pdf", "a.pdf"]),
        "page_id": np.asarray([1, 0]),
        "image_rgb": np.zeros((count, 2, 3, 3), dtype="uint8"),
        "scale": np.ones(count),
        "page_width": np.asarray([3, 3]),
        "page_height": np.asarray([2, 2]),
        "pdf_len": np.asarray([2, 2]),
        "content": np.asarray(["one", "zero"], dtype=object),
        "poisoned": np.asarray([False, False]),
        "batch_token": np.asarray(["batch", "batch"]),
        "batch_size_observed": np.asarray([2, 2]),
        "render_started_s": np.asarray([1.0, 1.0]),
        "render_finished_s": np.asarray([2.0, 2.0]),
        "actor_init_started_s": np.asarray([3.0, 3.0]),
        "actor_ready_s": np.asarray([4.0, 4.0]),
        "batch_started_s": np.asarray([5.0, 5.0]),
        "batch_finished_s": np.asarray([6.0, 6.0]),
        "payload_publish_started_s": np.zeros(count),
        "payload_publish_finished_s": np.zeros(count),
        "payload_block_bytes": np.zeros(count),
    }
    assembler = _TimedAssemblePdf(
        output_dir="/tmp/out",
        metadata_only=True,
        regroup_mode="full_value",
        stores=(),
        poison_policy="skip_page",
    )

    output = assembler(batch)
    document = _decode_document(output["result_blob"][0])

    assert output["parent_id"].tolist() == [7]
    assert document.page_ids == (0, 1)
    assert document.value["page_ids"] == [0, 1]


def test_full_value_assembly_skips_only_marked_bad_page():
    count = 2
    batch = {
        "parent_id": np.asarray([7, 7]),
        "page_ordinal": np.asarray([0, 1]),
        "pdf_path": np.asarray(["a.pdf", "a.pdf"]),
        "page_id": np.asarray([0, 1]),
        "image_rgb": np.zeros((count, 2, 3, 3), dtype="uint8"),
        "scale": np.ones(count),
        "page_width": np.asarray([3, 3]),
        "page_height": np.asarray([2, 2]),
        "pdf_len": np.asarray([2, 2]),
        "content": np.asarray(["zero", None], dtype=object),
        "poisoned": np.asarray([False, True]),
        "batch_token": np.asarray(["batch", "batch"]),
        "batch_size_observed": np.asarray([2, 2]),
        "render_started_s": np.asarray([1.0, 1.0]),
        "render_finished_s": np.asarray([2.0, 2.0]),
        "actor_init_started_s": np.asarray([3.0, 3.0]),
        "actor_ready_s": np.asarray([4.0, 4.0]),
        "batch_started_s": np.asarray([5.0, 5.0]),
        "batch_finished_s": np.asarray([6.0, 6.0]),
        "payload_publish_started_s": np.zeros(count),
        "payload_publish_finished_s": np.zeros(count),
        "payload_block_bytes": np.zeros(count),
    }
    assembler = _TimedAssemblePdf(
        output_dir="/tmp/out",
        metadata_only=True,
        regroup_mode="full_value",
        stores=(),
        poison_policy="skip_page",
    )

    document = _decode_document(assembler(batch)["result_blob"][0])

    assert document.page_ids == (0,)
    assert document.value == {
        "pdf": "a",
        "pages": 1,
        "page_ids": [0],
        "input_pages": 2,
        "poisoned_pages": 1,
        "poison_page_ids": [1],
    }


def test_full_value_assembly_can_drop_the_entire_poison_parent():
    count = 2
    batch = {
        "parent_id": np.asarray([7, 7]),
        "page_ordinal": np.asarray([0, 1]),
        "pdf_path": np.asarray(["a.pdf", "a.pdf"]),
        "page_id": np.asarray([0, 1]),
        "image_rgb": np.zeros((count, 2, 3, 3), dtype="uint8"),
        "scale": np.ones(count),
        "page_width": np.asarray([3, 3]),
        "page_height": np.asarray([2, 2]),
        "pdf_len": np.asarray([2, 2]),
        "content": np.asarray(["zero", None], dtype=object),
        "poisoned": np.asarray([False, True]),
        "batch_token": np.asarray(["batch", "batch"]),
        "batch_size_observed": np.asarray([2, 2]),
        "render_started_s": np.asarray([1.0, 1.0]),
        "render_finished_s": np.asarray([2.0, 2.0]),
        "actor_init_started_s": np.asarray([3.0, 3.0]),
        "actor_ready_s": np.asarray([4.0, 4.0]),
        "batch_started_s": np.asarray([5.0, 5.0]),
        "batch_finished_s": np.asarray([6.0, 6.0]),
        "payload_publish_started_s": np.zeros(count),
        "payload_publish_finished_s": np.zeros(count),
        "payload_block_bytes": np.zeros(count),
    }
    assembler = _TimedAssemblePdf(
        output_dir="/tmp/out",
        metadata_only=True,
        regroup_mode="full_value",
        stores=(),
        poison_policy="drop_parent",
    )

    document = _decode_document(assembler(batch)["result_blob"][0])

    assert document.page_ids == ()
    assert document.value["dropped_parent"] is True
    assert document.value["pages"] == 0
    assert document.value["input_pages"] == 2
    assert document.value["poison_page_ids"] == [1]


def test_shuffle_observation_extracts_map_and_finalize_bytes():
    stats = """Operator 3 Shuffle(key_columns=('parent_id',)): executed in 4.2s
	Suboperator 0 thing_shuffle: 1 tasks executed
	* Output size bytes per block: 10 min, 10 max, 10 mean, 10 total
	Suboperator 1 thing_finalize: 2 tasks executed
	* Output size bytes per block: 6 min, 7 max, 6 mean, 13 total
Operator 4 Map: executed in 1s
"""

    assert _ray_data_shuffle_observation(stats) == {
        "operator_wall_s": 4.2,
        "map_output_bytes": 10,
        "finalize_output_bytes": 13,
    }


def test_ocr_remote_tasks_are_distinct_from_udf_batch_calls():
    stats = (
        "Operator 2 MapBatches(_TimedOcrPages): "
        "99 tasks executed, 589 blocks produced in 10s\n"
    )

    assert _ray_data_ocr_remote_tasks(stats) == 99


def test_compact_payload_keeps_rollup_metrics_not_per_document_hashes():
    payload = {
        "run_wall_s": 1.0,
        "output_signatures": [{"pdf": "a"}],
    }

    assert _compact_payload(payload) == {"run_wall_s": 1.0}
    assert "output_signatures" in payload
