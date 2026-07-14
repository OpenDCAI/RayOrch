from __future__ import annotations

from dataclasses import replace

import pytest

from rayorch.experimental.multigrain.ir import (
    ChildrenOf,
    ExecutionGraph,
    ExpandOp,
    GraphInputRef,
    GraphInputSpec,
    GraphValidationError,
    NodeOutputRef,
    NodeSpec,
    OperatorFactorySpec,
    OutputSpec,
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
            OutputSpec(pages, "page", ChildrenOf(source, "page")),
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


def test_expand_outputs_form_an_invocation_local_identity_forest() -> None:
    graph = _mixed_expand_graph()
    assert verify_graph(graph) is graph


def test_identity_forest_rejects_forward_output_reference() -> None:
    graph = _mixed_expand_graph()
    node = graph.nodes[0]
    bad_first = replace(
        node.outputs[0],
        relation=SameAs(node.outputs[1].ref),
        grain=node.outputs[1].grain,
    )
    bad_graph = replace(
        graph,
        nodes=(replace(node, outputs=(bad_first, *node.outputs[1:])),),
    )

    with pytest.raises(GraphValidationError, match="self or forward"):
        verify_graph(bad_graph)


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
