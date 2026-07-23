"""Hand-authored Phase 0 semantic fixtures."""
from __future__ import annotations

from .models import ExpectedEntity, ExpectedFailure, ExpectedWorkUnit, SemanticCase


CASES = (
    SemanticCase(
        "linear_map",
        "map",
        (
            ExpectedEntity("source:0", (("value", "source:0"),), "mapped"),
            ExpectedEntity("source:1", (("value", "source:1"),), "mapped"),
        ),
        (
            ExpectedWorkUnit("row", "source:0", ("source:0",)),
            ExpectedWorkUnit("row", "source:1", ("source:1",)),
        ),
    ),
    SemanticCase(
        "map_multi_output",
        "map",
        (
            ExpectedEntity("source:0", (("value", "source:0"),), "left"),
            ExpectedEntity("source:0", (("value", "source:0"),), "right"),
        ),
        (ExpectedWorkUnit("row", "source:0", ("source:0",)),),
    ),
    SemanticCase(
        "filter_all_false",
        "filter",
        (),
        (
            ExpectedWorkUnit("row", "source:0", ("source:0",)),
            ExpectedWorkUnit("row", "source:1", ("source:1",)),
        ),
    ),
    SemanticCase(
        "select_mask_annotations",
        "select",
        (
            ExpectedEntity(
                "entity:0",
                (("score_input", "source:0"), ("context", "context:0")),
                "selected_score_input",
            ),
            ExpectedEntity(
                "entity:0",
                (("score_input", "source:0"), ("context", "context:0")),
                "selected_context",
            ),
            ExpectedEntity(
                "entity:0",
                (("score_input", "source:0"), ("context", "context:0")),
                "annotation",
            ),
        ),
        (
            ExpectedWorkUnit(
                "row", "entity:0", ("source:0", "context:0")
            ),
        ),
    ),
    SemanticCase(
        "expand_zero_one_many",
        "expand",
        (
            ExpectedEntity("parent:1/child:0", (("parent", "parent:1"),)),
            ExpectedEntity("parent:2/child:0", (("parent", "parent:2"),)),
            ExpectedEntity("parent:2/child:1", (("parent", "parent:2"),)),
        ),
        (
            ExpectedWorkUnit("parent", "parent:0", ("parent:0",)),
            ExpectedWorkUnit("parent", "parent:1", ("parent:1",)),
            ExpectedWorkUnit("parent", "parent:2", ("parent:2",)),
        ),
    ),
    SemanticCase(
        "reduce_complete_empty_fiber",
        "reduce",
        (ExpectedEntity("document:0", (("anchor", "document:0"),)),),
        (
            ExpectedWorkUnit(
                "fiber", "document:0", ("document:0",)
            ),
        ),
    ),
    SemanticCase(
        "reduce_missing_required_member",
        "reduce",
        (),
        (),
        (
            ExpectedFailure(
                "document:0", "suppressed", ("page:1",)
            ),
        ),
    ),
    SemanticCase(
        "key_relate_repeated_keys",
        "relate-key",
        (
            ExpectedEntity(
                "key:a/left:0/right:0",
                (("left", "left:0"), ("right", "right:0")),
            ),
            ExpectedEntity(
                "key:a/left:1/right:0",
                (("left", "left:1"), ("right", "right:0")),
            ),
        ),
        (
            ExpectedWorkUnit(
                "join-key", "key:a", ("left:0", "left:1", "right:0")
            ),
        ),
    ),
    SemanticCase(
        "custom_relate_named_parents",
        "relate-custom",
        (
            ExpectedEntity(
                "relation:0", (("user", "user:1"), ("event", "event:3"))
            ),
        ),
        (
            ExpectedWorkUnit(
                "whole-invocation",
                "custom_relate_named_parents",
                ("user:*", "event:*"),
            ),
        ),
    ),
    SemanticCase(
        "diamond_reorder_filter_suppression",
        "diamond",
        (
            ExpectedEntity(
                "row:2", (("left", "left:2"), ("right", "right:2"))
            ),
        ),
        (ExpectedWorkUnit("row", "row:2", ("left:2", "right:2")),),
        (ExpectedFailure("row:1", "suppressed", ("right:1",)),),
    ),
    SemanticCase(
        "zero_output_terminal_writer",
        "map-writer",
        (),
        (ExpectedWorkUnit("row", "source:0", ("source:0",)),),
    ),
    SemanticCase(
        "multiple_independent_input_groups",
        "relate-key",
        (
            ExpectedEntity(
                "key:7/user:0/event:0",
                (("users", "user:0"), ("events", "event:0")),
            ),
        ),
        (
            ExpectedWorkUnit(
                "join-key", "key:7", ("user:0", "event:0")
            ),
        ),
    ),
    SemanticCase(
        "external_descriptor_feed",
        "map",
        (
            ExpectedEntity(
                "descriptor:0", (("descriptor", "s3://bucket/a#row-group=2"),)
            ),
        ),
        (
            ExpectedWorkUnit(
                "row",
                "descriptor:0",
                ("s3://bucket/a#row-group=2",),
            ),
        ),
    ),
)


CASES_BY_NAME = {case.name: case for case in CASES}
