from __future__ import annotations

from dataclasses import replace

import pytest

from rayorch.experimental import multigrain as mg
from rayorch.experimental.multigrain.ir import (
    AggregateOf,
    ChildrenOf,
    ExpandOp,
    GraphValidationError,
    MapOp,
    NodeOutputRef,
    SameAs,
    verify_graph,
)

from test.experimental.multigrain.test_pdf_mvp import (
    Assemble,
    Layout,
    OCR,
    PdfToImages,
)


class TraceMineruPipe(mg.Pipeline):
    def __init__(self) -> None:
        super().__init__()
        self.pdf_to_images = mg.Expand(
            PdfToImages,
            {"a.pdf": 2},
            parent=0,
            child_label="page",
            num_outputs=2,
        )
        self.layout = mg.Map(Layout)
        self.ocr = mg.Map(OCR)
        self.assemble = mg.Reduce(Assemble)

    def forward(self, pdfs):
        images, page_meta = self.pdf_to_images(pdfs)
        layouts = self.layout(images)
        texts = self.ocr(images, layouts)
        return self.assemble(mg.group_by(pdfs, texts, page_meta))


def test_compile_traces_minimal_relation_execution_graph() -> None:
    graph = TraceMineruPipe().compile()

    assert tuple(node.name for node in graph.nodes) == (
        "PdfToImages",
        "Layout",
        "OCR",
        "Assemble",
    )

    expand = graph.node("PdfToImages")
    assert isinstance(expand.operation, ExpandOp)
    assert [output.name for output in expand.outputs] == ["out", "out_1"]
    assert [output.grain for output in expand.outputs] == [
        "page",
        "page",
    ]
    assert all(isinstance(output.relation, ChildrenOf) for output in expand.outputs)
    assert all(
        output.relation.parent == graph.inputs[0].ref
        for output in expand.outputs
        if isinstance(output.relation, ChildrenOf)
    )

    layout = graph.node("Layout")
    assert isinstance(layout.operation, MapOp)
    assert layout.inputs == (expand.outputs[0].ref,)
    assert isinstance(layout.outputs[0].relation, SameAs)

    ocr = graph.node("OCR")
    assert isinstance(ocr.operation, MapOp)
    assert ocr.inputs == (expand.outputs[0].ref, layout.outputs[0].ref)
    assert ocr.outputs[0].grain == "page"

    reduce = graph.node("Assemble")
    relation = reduce.outputs[0].relation
    assert isinstance(relation, AggregateOf)
    assert relation.anchor == graph.inputs[0].ref
    assert reduce.inputs[1:] == (ocr.outputs[0].ref, expand.outputs[1].ref)
    assert reduce.outputs[0].grain == "pdfs"
    assert graph.outputs == (reduce.outputs[0].ref,)


def test_graph_indexes_are_derived_from_port_refs() -> None:
    graph = TraceMineruPipe().compile()

    assert graph.dependencies == {
        "PdfToImages": (),
        "Layout": ("PdfToImages",),
        "OCR": ("PdfToImages", "Layout"),
        "Assemble": ("OCR", "PdfToImages"),
    }
    assert graph.consumers["PdfToImages"] == ("Layout", "OCR", "Assemble")


def test_graph_is_displayable_serializable_and_has_no_trace_state() -> None:
    graph = TraceMineruPipe().compile()

    payload = graph.to_dict()
    assert payload["name"] == "TraceMineruPipe"
    assert "tracer" not in str(payload)

    description = graph.describe()
    assert "ExecutionGraph(TraceMineruPipe)" in description
    assert "[EXPAND] PdfToImages" in description
    assert "[REDUCE] Assemble" in description
    assert "flowchart TD" in graph.to_mermaid()


def test_mandatory_verifier_rejects_relation_grain_drift() -> None:
    graph = TraceMineruPipe().compile()
    ocr = graph.node("OCR")
    bad_output = replace(ocr.outputs[0], grain="pdfs")
    bad_node = replace(ocr, outputs=(bad_output,))
    bad_graph = replace(
        graph,
        nodes=tuple(bad_node if node.name == "OCR" else node for node in graph.nodes),
    )

    with pytest.raises(GraphValidationError, match="keep source grain"):
        verify_graph(bad_graph)


class FanoutPipe(mg.Pipeline):
    def __init__(self) -> None:
        super().__init__()
        self.left = mg.Map(Layout, name="left")
        self.right = mg.Map(Layout, name="right")
        self.merge = mg.Map(OCR, name="merge")

    def forward(self, pages):
        left = self.left(pages)
        right = self.right(pages)
        return self.merge(left, right)


def test_compile_traces_fanout_and_same_grain_fanin() -> None:
    graph = FanoutPipe().compile()
    assert tuple(node.name for node in graph.nodes) == ("left", "right", "merge")
    assert graph.node("merge").inputs == (
        NodeOutputRef("left"),
        NodeOutputRef("right"),
    )


class ExpandWithSideInputPipe(mg.Pipeline):
    def __init__(self) -> None:
        super().__init__()
        self.expand = mg.Expand(
            PdfToImages,
            {"a.pdf": 2},
            parent=1,
            child_label="page",
            name="expand_docs",
        )

    def forward(self, settings, documents):
        return self.expand(settings, documents)


def test_expand_relation_names_non_first_parent_directly() -> None:
    graph = ExpandWithSideInputPipe().compile()
    relation = graph.node("expand_docs").outputs[0].relation
    assert isinstance(relation, ChildrenOf)
    assert relation.parent == graph.inputs[1].ref


class InvalidCrossGrainPipe(mg.Pipeline):
    def __init__(self) -> None:
        super().__init__()
        self.expand = mg.Expand(PdfToImages, {"a.pdf": 2}, child_label="page")
        self.bad = mg.Map(Layout, name="bad")

    def forward(self, documents):
        pages = self.expand(documents)
        return self.bad(documents, pages)


def test_compile_rejects_cross_grain_map_without_group_by() -> None:
    with pytest.raises(ValueError, match="same-grain"):
        InvalidCrossGrainPipe().compile()


class UngroupedReducePipe(mg.Pipeline):
    def __init__(self) -> None:
        super().__init__()
        self.reduce = mg.Reduce(Assemble)

    def forward(self, documents):
        return self.reduce(documents)


def test_reduce_requires_explicit_group_by() -> None:
    with pytest.raises(TypeError, match="group_by"):
        UngroupedReducePipe().compile()


class ReusedNamePipe(mg.Pipeline):
    def __init__(self) -> None:
        super().__init__()
        self.step = mg.Map(Layout, name="step")

    def forward(self, pages):
        return self.step(self.step(pages))


def test_compile_assigns_unique_node_names_for_reused_wrapper() -> None:
    graph = ReusedNamePipe().compile()
    assert tuple(node.name for node in graph.nodes) == ("step", "step_1")
    assert graph.node("step_1").inputs == (NodeOutputRef("step"),)
