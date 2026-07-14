from __future__ import annotations

from dataclasses import replace

from rayorch.experimental import multigrain as mg
from rayorch.experimental.multigrain.ir.capabilities import (
    RelationEvidenceFamily,
    capabilities_for,
)
from rayorch.experimental.multigrain.execution.handlers import (
    DEFAULT_HANDLER_REGISTRY,
    MapHandler,
    SelectFilterHandler,
)
from rayorch.experimental.multigrain.primitives.output import merge_aligned_inputs
from rayorch.experimental.multigrain.ir import VerifyPass

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


def test_capabilities_are_derived_from_relation_contracts() -> None:
    graph = _CorePipe().compile()

    map_caps = capabilities_for(graph.node("Layout"))
    assert map_caps.identity_alignment is True
    assert map_caps.row_partitionable is True
    assert map_caps.relation_evidence is RelationEvidenceFamily.ALIGNED

    expand_caps = capabilities_for(graph.node("PdfToImages"))
    assert expand_caps.row_partitionable is True
    assert expand_caps.relation_evidence is RelationEvidenceFamily.PARENT

    reduce_caps = capabilities_for(graph.node("Assemble"))
    assert reduce_caps.group_completion is True
    assert reduce_caps.row_partitionable is False
    assert reduce_caps.relation_evidence is RelationEvidenceFamily.ANCHOR


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
        SelectFilterHandler,
    )


def test_structural_verifier_rejects_relation_parent_drift() -> None:
    graph = _CorePipe().compile()
    node = graph.node("Layout")
    relation = replace(node.contract.relations[0], parents=())
    bad_node = replace(
        node,
        contract=replace(node.contract, relations=(relation,)),
    )
    bad_graph = replace(
        graph,
        nodes=tuple(
            bad_node if current.name == node.name else current
            for current in graph.nodes
        ),
    )

    result = VerifyPass().run(bad_graph)

    assert result.ok is False
    assert any(
        "relation parents" in diagnostic.message
        for diagnostic in result.diagnostics
    )


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
