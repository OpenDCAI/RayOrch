from __future__ import annotations

from dataclasses import replace

import pytest

from rayorch.experimental import multigrain as mg
from rayorch.experimental.multigrain.ir.capabilities import (
    is_row_partitionable,
)
from rayorch.experimental.multigrain.execution.handlers import (
    DEFAULT_HANDLER_REGISTRY,
    FilterByMaskHandler,
    MapHandler,
)
from rayorch.experimental.multigrain.primitives.output import merge_aligned_inputs
from rayorch.experimental.multigrain.ir import (
    GraphInputRef,
    GraphValidationError,
    SameAs,
    verify_graph,
)

from test.experimental.multigrain.test_executor import ExecSelectPipe
from test.experimental.multigrain.test_pdf_mvp import (
    Assemble,
    Layout,
    PdfToImages,
)


class _CorePipe(mg.Pipeline):
    def __init__(self) -> None:
        super().__init__()
        self.expand = mg.Expand(
            PdfToImages,
            {"a": 1},
            num_outputs=2,
        )
        self.map = mg.Map(Layout)
        self.reduce = mg.Reduce(Assemble)

    def forward(self, docs):
        pages, metadata = self.expand(docs)
        mapped = self.map(pages)
        return self.reduce(mg.group_by(docs, mapped, metadata))


def test_scheduling_facts_are_derived_from_verified_operations() -> None:
    graph = _CorePipe().compile()

    assert is_row_partitionable(graph.node("Layout")) is True
    assert is_row_partitionable(graph.node("PdfToImages")) is True
    assert is_row_partitionable(graph.node("Assemble")) is False


def test_handler_registry_resolves_primitive_and_internal_recipes() -> None:
    select_graph = ExecSelectPipe().compile()

    assert isinstance(
        DEFAULT_HANDLER_REGISTRY.resolve(select_graph.node("ScoreAndKeep__map")),
        MapHandler,
    )
    assert isinstance(
        DEFAULT_HANDLER_REGISTRY.resolve(
            select_graph.node("ScoreAndKeep__filter")
        ),
        FilterByMaskHandler,
    )


def test_structural_verifier_rejects_relation_parent_drift() -> None:
    graph = _CorePipe().compile()
    node = graph.node("Layout")
    bad_output = replace(
        node.outputs[0],
        relation=SameAs(GraphInputRef("missing")),
    )
    bad_node = replace(node, outputs=(bad_output,))
    bad_graph = replace(
        graph,
        nodes=tuple(
            bad_node if current.name == node.name else current
            for current in graph.nodes
        ),
    )

    with pytest.raises(GraphValidationError, match="unavailable"):
        verify_graph(bad_graph)


def test_aligned_lineage_merge_rejects_conflicting_ancestor_identity() -> None:
    left = mg.source(["a"], name="row")
    right = left.with_values(["b"], name="other", op_name="other")
    right.ancestors[0]["row"] = "different-id"

    try:
        merge_aligned_inputs((left, right))
    except ValueError as exc:
        assert "conflicting ancestor identity" in str(exc)
    else:
        raise AssertionError("conflicting lineage metadata was silently merged")
