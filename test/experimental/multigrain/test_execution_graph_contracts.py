from __future__ import annotations

from dataclasses import replace

import pytest

from rayorch.experimental.multigrain.data import PortBatch
from rayorch.experimental.multigrain.ir import (
    AggregateOf,
    ChildrenOf,
    ExecutionGraph,
    ExpandOp,
    GraphInputRef,
    GraphInputSpec,
    GraphValidationError,
    IncompleteGroupPolicy,
    MapOp,
    NodeOutputRef,
    NodeSpec,
    OperatorFactorySpec,
    OutputSpec,
    RelatedFrom,
    ReduceOp,
    RelateOp,
    SameAs,
    validate_shard_plan,
    verify_graph,
)


def _mixed_expand_graph() -> ExecutionGraph:
    source = GraphInputRef("documents")
    pages = NodeOutputRef("parse", "pages")
    page_meta = NodeOutputRef("parse", "page_meta")
    document_meta = NodeOutputRef("parse", "document_meta")
    node = NodeSpec(
        name="parse",
        inputs=(source,),
        outputs=(
            OutputSpec(pages, "page", ChildrenOf(source)),
            OutputSpec(page_meta, "page", SameAs(pages)),
            OutputSpec(document_meta, "documents", SameAs(source)),
        ),
        operation=ExpandOp(
            OperatorFactorySpec(
                "test.experimental.multigrain.test_pdf_mvp.PdfToImages"
            )
        ),
    )
    return ExecutionGraph(
        name="mixed",
        inputs=(GraphInputSpec(source, "documents"),),
        nodes=(node,),
        outputs=(pages, page_meta, document_meta),
    )


def test_deferred_mixed_expand_is_not_yet_an_executable_graph() -> None:
    graph = _mixed_expand_graph()
    with pytest.raises(GraphValidationError, match="cannot produce SameAs"):
        verify_graph(graph)


def test_identity_forest_rejects_forward_output_reference() -> None:
    graph = _mixed_expand_graph()
    node = graph.nodes[0]
    bad_first = replace(
        node.outputs[0],
        relation=ChildrenOf(node.outputs[1].ref),
    )
    bad_graph = replace(
        graph,
        nodes=(replace(node, outputs=(bad_first, *node.outputs[1:])),),
    )

    with pytest.raises(GraphValidationError, match="self or forward"):
        verify_graph(bad_graph)


def _reduce_graph(*, child_root: str, mixed_policy: bool) -> ExecutionGraph:
    docs = GraphInputRef("docs")
    other = GraphInputRef("other")
    child_parent = docs if child_root == "docs" else other
    children = NodeOutputRef("split")
    split = NodeSpec(
        name="split",
        inputs=(child_parent,),
        outputs=(OutputSpec(children, "page", ChildrenOf(child_parent)),),
        operation=ExpandOp(
            OperatorFactorySpec(
                "test.experimental.multigrain.test_pdf_mvp.PdfToImages"
            )
        ),
    )
    reduced = NodeOutputRef("reduce")
    reduced_meta = NodeOutputRef("reduce", "meta")
    reduce = NodeSpec(
        name="reduce",
        inputs=(docs, children),
        outputs=(
            OutputSpec(reduced, "docs", AggregateOf(docs)),
            OutputSpec(
                reduced_meta,
                "docs",
                AggregateOf(
                    docs,
                    IncompleteGroupPolicy.FAIL_CLOSED
                    if mixed_policy
                    else IncompleteGroupPolicy.FAIL_OPEN,
                ),
            ),
        ),
        operation=ReduceOp(
            OperatorFactorySpec(
                "test.experimental.multigrain.test_pdf_mvp.Assemble"
            )
        ),
    )
    return ExecutionGraph(
        name="reduce_contract",
        inputs=(
            GraphInputSpec(docs, "docs"),
            GraphInputSpec(other, "other"),
        ),
        nodes=(split, reduce),
        outputs=(reduced, reduced_meta),
    )


def test_reduce_members_must_descend_from_declared_anchor() -> None:
    with pytest.raises(GraphValidationError, match="not a descendant"):
        verify_graph(_reduce_graph(child_root="other", mixed_policy=False))


def test_multi_output_reduce_requires_one_incomplete_policy() -> None:
    with pytest.raises(GraphValidationError, match="share one AggregateOf"):
        verify_graph(_reduce_graph(child_root="docs", mixed_policy=True))


def test_port_batch_rejects_duplicate_record_identity() -> None:
    with pytest.raises(ValueError, match="record_ids must be unique"):
        PortBatch(
            name="row",
            values=["a", "b"],
            record_ids=["same", "same"],
            display_keys=["a", "b"],
            ancestors=[{}, {}],
            ancestor_display=[{}, {}],
            ordinals=[{}, {}],
            lineage=[(), ()],
        )


def test_operator_factory_requires_importable_path_syntax() -> None:
    graph = _reduce_graph(child_root="docs", mixed_policy=False)
    split = graph.nodes[0]
    bad = replace(
        graph,
        nodes=(
            replace(
                split,
                operation=ExpandOp(OperatorFactorySpec("not_importable")),
            ),
            graph.nodes[1],
        ),
    )
    with pytest.raises(GraphValidationError, match="not an importable class"):
        verify_graph(bad)


def test_relate_rejects_unknown_matcher_during_verification() -> None:
    left = GraphInputRef("left")
    right = GraphInputRef("right")
    related = NodeOutputRef("relate")
    graph = ExecutionGraph(
        name="bad_matcher",
        inputs=(
            GraphInputSpec(left, "row"),
            GraphInputSpec(right, "row"),
        ),
        nodes=(
            NodeSpec(
                name="relate",
                inputs=(left, right),
                outputs=(
                    OutputSpec(
                        related,
                        "pair",
                        RelatedFrom(("left", "right")),
                    ),
                ),
                operation=RelateOp(
                    OperatorFactorySpec(
                        "test.experimental.multigrain.test_relate_key_join.Match"
                    ),
                    object(),  # type: ignore[arg-type]
                ),
            ),
        ),
        outputs=(related,),
    )

    with pytest.raises(GraphValidationError, match="unsupported matcher object"):
        verify_graph(graph)


def test_operator_factory_requires_resolvable_class() -> None:
    graph = _reduce_graph(child_root="docs", mixed_policy=False)
    split = graph.nodes[0]
    bad = replace(
        graph,
        nodes=(
            replace(
                split,
                operation=ExpandOp(
                    OperatorFactorySpec("test.experimental.multigrain.missing.NoOp")
                ),
            ),
            graph.nodes[1],
        ),
    )
    with pytest.raises(GraphValidationError, match="not an importable class"):
        verify_graph(bad)


def test_operator_factory_arguments_must_be_pickleable() -> None:
    graph = _reduce_graph(child_root="docs", mixed_policy=False)
    split = graph.nodes[0]
    bad = replace(
        graph,
        nodes=(
            replace(
                split,
                operation=ExpandOp(
                    OperatorFactorySpec(
                        "test.experimental.multigrain.test_pdf_mvp.PdfToImages",
                        args=(lambda value: value,),
                    )
                ),
            ),
            graph.nodes[1],
        ),
    )
    with pytest.raises(GraphValidationError, match="must be pickleable"):
        verify_graph(bad)


def test_map_static_ancestry_includes_all_aligned_inputs() -> None:
    primary = GraphInputRef("primary")
    context = GraphInputRef("context")
    mapped = NodeOutputRef("map")
    reduced = NodeOutputRef("reduce")
    graph = ExecutionGraph(
        name="multi_input_map_ancestry",
        inputs=(
            GraphInputSpec(primary, "row"),
            GraphInputSpec(context, "row"),
        ),
        nodes=(
            NodeSpec(
                name="map",
                inputs=(primary, context),
                outputs=(OutputSpec(mapped, "row", SameAs(primary)),),
                operation=MapOp(
                    OperatorFactorySpec(
                        "test.experimental.multigrain.test_pdf_mvp.PdfToImages"
                    )
                ),
            ),
            NodeSpec(
                name="reduce",
                inputs=(context, mapped),
                outputs=(OutputSpec(reduced, "row", AggregateOf(context)),),
                operation=ReduceOp(
                    OperatorFactorySpec(
                        "test.experimental.multigrain.test_pdf_mvp.Assemble"
                    )
                ),
            ),
        ),
        outputs=(reduced,),
    )

    verify_graph(graph)


@pytest.mark.parametrize(
    "partitions, message",
    [
        ([[0], [0, 1]], "duplicate"),
        ([[0]], "cover every row"),
        ([[0, 2]], "out-of-range"),
    ],
)
def test_shard_plan_must_be_an_exact_partition(partitions, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_shard_plan(partitions, 2)


def test_shard_plan_accepts_arbitrary_reordering() -> None:
    assert validate_shard_plan([[3, 0], [2, 1]], 4) == (
        (3, 0),
        (2, 1),
    )
