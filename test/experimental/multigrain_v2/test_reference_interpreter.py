from __future__ import annotations

from test.experimental.multigrain_v2.reference_semantics.interpreter import (
    interpret_custom_relate,
    interpret_diamond,
    interpret_expand,
    interpret_key_relate,
    interpret_map,
    interpret_reduce,
    interpret_select,
    interpret_writer,
)
from test.experimental.multigrain_v2.reference_semantics.semantic_cases import (
    CASES_BY_NAME,
)


def test_reference_map_filter_select_and_expand_match_hand_goldens() -> None:
    linear = interpret_map(
        "linear_map",
        {"value": (("source:0", "source:0"), ("source:1", "source:1"))},
        ("mapped",),
    )
    assert linear == CASES_BY_NAME["linear_map"]

    multi = interpret_map(
        "map_multi_output",
        {"value": (("source:0", "source:0"),)},
        ("left", "right"),
    )
    assert multi == CASES_BY_NAME["map_multi_output"]

    filtered = interpret_select(
        "filter_all_false",
        {
            "value": (
                ("source:0", "source:0"),
                ("source:1", "source:1"),
            )
        },
        (False, False),
        ("filtered",),
        primitive="filter",
    )
    expected_filter = CASES_BY_NAME["filter_all_false"]
    assert filtered == expected_filter

    selected = interpret_select(
        "select_mask_annotations",
        {
            "score_input": (("entity:0", "source:0"),),
            "context": (("entity:0", "context:0"),),
        },
        (True,),
        ("selected_score_input", "selected_context"),
        ("annotation",),
    )
    assert selected == CASES_BY_NAME["select_mask_annotations"]

    expanded = interpret_expand(
        "expand_zero_one_many",
        ("parent:0", "parent:1", "parent:2"),
        (0, 1, 2),
    )
    assert expanded == CASES_BY_NAME["expand_zero_one_many"]


def test_reference_reduce_and_relates_match_hand_golden_closures() -> None:
    complete = interpret_reduce(
        "reduce_complete_empty_fiber",
        ("document:0",),
        {"pages": {}},
    )
    assert complete == CASES_BY_NAME["reduce_complete_empty_fiber"]

    suppressed = interpret_reduce(
        "reduce_missing_required_member",
        ("document:0",),
        {"pages": {"document:0": ("page:0", "page:1")}},
        frozenset({"document:0"}),
        {"document:0": ("page:1",)},
    )
    expected_suppressed = CASES_BY_NAME["reduce_missing_required_member"]
    assert suppressed == expected_suppressed

    related = interpret_key_relate(
        "key_relate_repeated_keys",
        {
            "left": (("left:0", "a"), ("left:1", "a")),
            "right": (("right:0", "a"),),
        },
    )
    expected_related = CASES_BY_NAME["key_relate_repeated_keys"]
    assert related == expected_related

    custom = interpret_custom_relate(
        "custom_relate_named_parents",
        {"user": ("user:1",), "event": ("event:3",)},
        {"user": (0,), "event": (0,)},
    )
    expected_custom = CASES_BY_NAME["custom_relate_named_parents"]
    assert custom == expected_custom


def test_reference_remaining_phase0_motifs_match_hand_goldens() -> None:
    diamond = interpret_diamond(
        "diamond_reorder_filter_suppression",
        {
            "left": (
                ("row:1", "left:1"),
                ("row:2", "left:2"),
            ),
            "right": (
                ("row:1", "right:1"),
                ("row:2", "right:2"),
            ),
        },
        (1,),
        {"row:1": ("right:1",)},
    )
    assert diamond == CASES_BY_NAME["diamond_reorder_filter_suppression"]

    writer = interpret_writer(
        "zero_output_terminal_writer",
        {"value": (("source:0", "source:0"),)},
    )
    assert writer == CASES_BY_NAME["zero_output_terminal_writer"]

    multiple_inputs = interpret_key_relate(
        "multiple_independent_input_groups",
        {
            "users": (("user:0", 7),),
            "events": (("event:0", 7),),
        },
    )
    assert multiple_inputs == CASES_BY_NAME[
        "multiple_independent_input_groups"
    ]

    descriptor = "s3://bucket/a#row-group=2"
    external = interpret_map(
        "external_descriptor_feed",
        {"descriptor": (("descriptor:0", descriptor),)},
        ("",),
    )
    assert external == CASES_BY_NAME["external_descriptor_feed"]
