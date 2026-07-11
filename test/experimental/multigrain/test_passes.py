from __future__ import annotations

from rayorch.experimental import multigrain as mg
from rayorch.experimental.multigrain.passes import (
    InsertRebatchAfterExpandPass,
    MarkMapFilterFusionCandidatesPass,
    PassManager,
    PlanReduceGroupsPass,
    RebatchCandidatePass,
    RelationSummaryPass,
    VerifyPass,
)

from test.experimental.multigrain.test_dag_trace import TraceMineruPipe
from test.experimental.multigrain.test_upper_primitives import FilterSelectPipe


def test_relation_summary_and_planning_passes_read_multigrain_ir() -> None:
    graph = TraceMineruPipe().compile()
    manager = PassManager(
        (
            RelationSummaryPass(),
            RebatchCandidatePass(),
            PlanReduceGroupsPass(),
        )
    )

    result = manager.run(graph)

    assert result.ok is True
    assert result.metadata["node_kinds"] == {
        "EXPAND": 1,
        "MAP": 2,
        "REDUCE": 1,
    }
    assert result.metadata["relation_kinds"] == {
        "EXPAND": 2,
        "PRESERVE": 2,
        "REDUCE": 1,
    }
    assert result.metadata["rebatch_candidates"] == [
        {
            "node": "PdfToImages",
            "port": "out",
            "grain": "PdfToImages",
            "consumers": ["Layout", "OCR"],
        },
        {
            "node": "PdfToImages",
            "port": "out_1",
            "grain": "PdfToImages",
            "consumers": ["Assemble"],
        },
    ]
    assert result.metadata["reduce_group_plans"] == [
        {
            "node": "Assemble",
            "anchor": "__input__pdfs",
            "descendants": ["OCR", "PdfToImages"],
            "order_by": "ordinal_or_stable_key",
            "missing": "fail_open",
        }
    ]


def test_insert_rebatch_after_expand_rewrites_downstream_ir_refs() -> None:
    graph = TraceMineruPipe().compile()

    result = InsertRebatchAfterExpandPass().run(graph)
    rewritten = result.graph

    assert result.metadata["inserted_rebatch_nodes"] == [
        "PdfToImages__rebatch_out",
        "PdfToImages__rebatch_out_1",
    ]
    assert rewritten.topo_order == (
        "PdfToImages",
        "PdfToImages__rebatch_out",
        "PdfToImages__rebatch_out_1",
        "Layout",
        "OCR",
        "Assemble",
    )
    assert rewritten.node("Layout").input_refs[0].node == "PdfToImages__rebatch_out"
    assert rewritten.node("OCR").input_refs[0].node == "PdfToImages__rebatch_out"
    assert rewritten.node("Assemble").input_refs[2].node == (
        "PdfToImages__rebatch_out_1"
    )
    assert VerifyPass().run(rewritten).ok is True


def test_map_filter_fusion_candidate_pass_reads_select_lowering_ir() -> None:
    graph = FilterSelectPipe().compile()

    result = MarkMapFilterFusionCandidatesPass().run(graph)

    assert result.metadata["fusion_candidates"] == [
        {
            "filter": "ScoreAndKeep__filter",
            "maps": ("ScoreAndKeep__map",),
            "strategy": "physical_select_fusion",
        }
    ]
