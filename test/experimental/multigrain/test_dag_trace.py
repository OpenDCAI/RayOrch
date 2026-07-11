from __future__ import annotations

from dataclasses import replace

import pytest

from rayorch.experimental import multigrain as mg
from rayorch.experimental.multigrain.graph import (
    MaterializePolicy,
    MaterializeReason,
    NodeKind,
    RelationKind,
)
from rayorch.experimental.multigrain.passes import PassManager, VerifyPass

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


def test_compile_traces_multigrain_pdf_dag_contracts() -> None:
    graph = TraceMineruPipe().compile()

    assert graph.topo_order == (
        "PdfToImages",
        "Layout",
        "OCR",
        "Assemble",
    )

    pdf_to_images = graph.node("PdfToImages")
    assert pdf_to_images.kind == "EXPAND"
    assert pdf_to_images.parent_input == 0
    assert [port.name for port in pdf_to_images.outputs] == [
        "PdfToImages",
        "PdfToImages_1",
    ]
    assert [port.grain for port in pdf_to_images.outputs] == [
        "PdfToImages",
        "PdfToImages",
    ]

    layout = graph.node("Layout")
    assert layout.kind == "MAP"
    assert [port.node for port in layout.inputs] == ["PdfToImages"]
    assert layout.outputs[0].grain == "PdfToImages"

    ocr = graph.node("OCR")
    assert ocr.kind == "MAP"
    assert [port.node for port in ocr.inputs] == ["PdfToImages", "Layout"]
    assert ocr.outputs[0].grain == "PdfToImages"

    assemble = graph.node("Assemble")
    assert assemble.kind == "REDUCE"
    assert assemble.grouped is True
    assert assemble.parent_input == 0
    assert [port.node for port in assemble.inputs] == [
        "__input__pdfs",
        "OCR",
        "PdfToImages",
    ]
    assert assemble.outputs[0].grain == "pdfs"
    assert [port.node for port in graph.outputs] == ["Assemble"]


def test_ir_records_relation_contracts_and_derived_graph_indexes() -> None:
    graph = TraceMineruPipe().compile()

    expand = graph.node("PdfToImages")
    assert expand.contract.kind == NodeKind.EXPAND
    assert [relation.relation for relation in expand.contract.relations] == [
        RelationKind.EXPAND,
        RelationKind.EXPAND,
    ]
    assert [relation.parent_input for relation in expand.contract.relations] == [0, 0]
    assert [spec.ref.port for spec in expand.output_specs] == ["out", "out_1"]
    assert [spec.grain for spec in expand.output_specs] == [
        "PdfToImages",
        "PdfToImages",
    ]
    assert expand.physical.prefer_rebatch is True

    reduce = graph.node("Assemble")
    assert reduce.contract.kind == NodeKind.REDUCE
    assert reduce.contract.relations[0].relation == RelationKind.REDUCE
    assert reduce.contract.relations[0].anchor.node == "__input__pdfs"
    assert reduce.output_specs[0].grain == "pdfs"

    assert graph.deps == {
        "PdfToImages": (),
        "Layout": ("PdfToImages",),
        "OCR": ("PdfToImages", "Layout"),
        "Assemble": ("OCR", "PdfToImages"),
    }
    assert graph.consumers["PdfToImages"] == ("Layout", "OCR", "Assemble")
    assert graph.graph_outputs[0].node == "Assemble"


def test_ir_is_displayable_and_serializable_without_live_tracer_state() -> None:
    graph = TraceMineruPipe().compile()
    graph = graph.with_materialization(
        graph.outputs[0],
        policy=MaterializePolicy.ON_FAILURE,
        reason=MaterializeReason.TRACE,
        storage={"kind": "local"},
    )

    payload = graph.to_dict()
    assert payload["name"] == "TraceMineruPipe"
    assert payload["nodes"][0]["op"]["cls_ref"].endswith("PdfToImages")
    assert payload["nodes"][0]["contract"]["relations"][0]["relation"] == "EXPAND"
    assert payload["materialization"][0]["policy"] == "on_failure"
    assert payload["materialization"][0]["reason"] == "trace"
    assert "tracer" not in str(payload)

    description = graph.describe()
    assert "MultigrainIR(TraceMineruPipe)" in description
    assert "[EXPAND] PdfToImages" in description
    assert "[REDUCE] Assemble" in description

    mermaid = graph.to_mermaid()
    assert "flowchart TD" in mermaid
    assert "PdfToImages -->|out| Layout" in mermaid
    assert "Assemble --> __sink__" in mermaid


def test_verify_pass_accepts_valid_ir_and_flags_contract_corner_cases() -> None:
    graph = TraceMineruPipe().compile()
    manager = PassManager((VerifyPass(),))

    assert manager.run(graph).ok is True

    ocr = graph.node("OCR")
    bad_ocr = replace(
        ocr,
        contract=replace(ocr.contract, input_grains=("pdfs", "PdfToImages")),
    )
    bad_graph = replace(
        graph,
        nodes=tuple(bad_ocr if node.name == "OCR" else node for node in graph.nodes),
    )

    result = manager.run(bad_graph)
    assert result.ok is False
    assert result.diagnostics[0].node == "OCR"
    assert "same-grain" in result.diagnostics[0].message


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

    assert graph.topo_order == ("left", "right", "merge")
    assert [port.node for port in graph.node("left").inputs] == ["__input__pages"]
    assert [port.node for port in graph.node("right").inputs] == ["__input__pages"]
    assert [port.node for port in graph.node("merge").inputs] == ["left", "right"]
    assert graph.node("merge").kind == "MAP"
    assert graph.node("merge").outputs[0].grain == "pages"


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


def test_expand_can_use_non_first_parent_input() -> None:
    graph = ExpandWithSideInputPipe().compile()
    expand = graph.node("expand_docs")

    assert [port.node for port in expand.inputs] == [
        "__input__settings",
        "__input__documents",
    ]
    assert expand.parent_input == 1
    assert expand.contract.relations[0].parent_input == 1
    assert VerifyPass().run(graph).ok is True


class InvalidCrossGrainPipe(mg.Pipeline):
    def __init__(self) -> None:
        super().__init__()
        self.expand = mg.Expand(
            PdfToImages,
            {"a.pdf": 2},
            parent=0,
            child_label="page",
        )
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


def test_reduce_requires_explicit_group_by_for_user_friendly_api() -> None:
    with pytest.raises(TypeError, match="group_by"):
        UngroupedReducePipe().compile()


class ReusedNamePipe(mg.Pipeline):
    def __init__(self) -> None:
        super().__init__()
        self.step = mg.Map(Layout, name="step")

    def forward(self, pages):
        first = self.step(pages)
        return self.step(first)


def test_compile_assigns_unique_node_names_for_reused_wrapper() -> None:
    graph = ReusedNamePipe().compile()

    assert graph.topo_order == ("step", "step_1")
    assert [port.node for port in graph.node("step_1").inputs] == ["step"]

