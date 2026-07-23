from __future__ import annotations

from test.experimental.multigrain_v2.reference_semantics.semantic_cases import (
    CASES,
    CASES_BY_NAME,
)


EXPECTED_CASES = {
    "linear_map",
    "map_multi_output",
    "filter_all_false",
    "select_mask_annotations",
    "expand_zero_one_many",
    "reduce_complete_empty_fiber",
    "reduce_missing_required_member",
    "key_relate_repeated_keys",
    "custom_relate_named_parents",
    "diamond_reorder_filter_suppression",
    "zero_output_terminal_writer",
    "multiple_independent_input_groups",
    "external_descriptor_feed",
}


def test_phase0_fixture_matrix_is_complete_and_unique() -> None:
    assert set(CASES_BY_NAME) == EXPECTED_CASES
    assert len(CASES_BY_NAME) == len(CASES)


def test_every_fixture_freezes_semantic_closure_and_direct_parents() -> None:
    for case in CASES:
        coordinates = [unit.coordinate for unit in case.work_units]
        assert len(coordinates) == len(set(coordinates)), case.name
        for entity in case.entities:
            assert entity.key
            assert len(entity.parents) == len(set(entity.parents)), case.name
        for failure in case.failures:
            assert failure.outcome in {"permanently_missing", "suppressed"}
            assert failure.causes


def test_reduce_and_relate_use_non_row_closures() -> None:
    reduce_case = CASES_BY_NAME["reduce_complete_empty_fiber"]
    assert reduce_case.work_units[0].kind == "fiber"
    assert len(reduce_case.work_units[0].members) == 1

    suppressed = CASES_BY_NAME["reduce_missing_required_member"]
    assert suppressed.work_units == ()
    assert suppressed.failures[0].coordinate == "document:0"

    key_case = CASES_BY_NAME["key_relate_repeated_keys"]
    assert key_case.work_units[0].kind == "join-key"
    assert len(key_case.work_units[0].members) == 3

    custom_case = CASES_BY_NAME["custom_relate_named_parents"]
    assert custom_case.work_units[0].kind == "whole-invocation"


def test_multi_output_map_shares_entity_identity_across_ports() -> None:
    outputs = CASES_BY_NAME["map_multi_output"].entities
    assert {entity.port for entity in outputs} == {"left", "right"}
    assert len({entity.key for entity in outputs}) == 1


def test_select_freezes_targets_and_annotation_outputs() -> None:
    outputs = CASES_BY_NAME["select_mask_annotations"].entities
    assert {entity.port for entity in outputs} == {
        "selected_score_input",
        "selected_context",
        "annotation",
    }
    assert len({entity.key for entity in outputs}) == 1
