"""End-to-end dummy-operator coverage for the experimental multigrain paradigm.

This file is the durable version of the ad hoc "does the representation actually
execute?" checks. It uses fresh dummy operators (independent from the PDF/mineru
fixtures) and exercises the full path for every graph motif:

    trace -> canonical/passive IR -> VerifyPass -> pickle round-trip
          -> MultigrainExecutor (local) -> transformed IR execution

Keep it as the reference example for "what shapes the paradigm supports and how
they run". The motifs mirror ``docs/todos/09-multi-grain-port-cardinality-api.md``
("Minimal Graph Motifs") and the passive-IR decision in
``docs/todos/10-multigrain-ir-mvp-plan.md``.
"""
from __future__ import annotations

import pickle

import pytest

from rayorch.experimental import multigrain as mg
from rayorch.experimental.multigrain.graph import SymbolicPort
from rayorch.experimental.multigrain.passes import (
    InsertRebatchAfterExpandPass,
    MarkMapFilterFusionCandidatesPass,
    PlanReduceGroupsPass,
    RebatchCandidatePass,
    RelationSummaryPass,
    VerifyPass,
)


# ---------------------------------------------------------------------------
# Dummy operators (fresh; not shared with the PDF/mineru fixtures)
# ---------------------------------------------------------------------------
class DocToChunks:
    """Expand 1:N -- one document string -> one chunk group."""

    def run(self, docs: list[str]) -> list[list[str]]:
        return [doc.split("-") for doc in docs]


class Embed:
    """Map 1:1 -- one chunk -> one embedding token."""

    def run(self, chunks: list[str]) -> list[str]:
        return [f"emb:{chunk}" for chunk in chunks]


class KeepGood:
    """Filter 1:0/1 -- drop embeddings marked as garbage."""

    def run(self, embeds: list[str]) -> list[bool]:
        return ["drop" not in embed for embed in embeds]


class AssembleChunks:
    """Reduce N:1 -- group chunk embeddings back to their document."""

    def run(self, docs: list[str], grouped: list[list[str]]) -> list[str]:
        return [f"{doc}=>[{','.join(group)}]" for doc, group in zip(docs, grouped)]


class Score:
    """Select -- return (keep_mask, score annotation)."""

    def run(self, chunks: list[str]) -> tuple[list[bool], list[int]]:
        scores = [len(chunk) for chunk in chunks]
        return [score >= 2 for score in scores], scores


class Match:
    """Relate M:N raw op -- pair up i-th image with i-th caption."""

    def run(self, images: list[str], captions: list[str]) -> list[str]:
        return [f"{image}|{caption}" for image, caption in zip(images, captions)]


def match_relation(raw_values: list[str]) -> list[tuple[str, dict[str, int]]]:
    """Adapter: expose invocation-local parent indexes as relation evidence."""
    return [
        (value, {"image": index, "caption": index})
        for index, value in enumerate(raw_values)
    ]


# ---------------------------------------------------------------------------
# Pipelines, one per motif family
# ---------------------------------------------------------------------------
class GovPipe(mg.Pipeline):
    """Expand -> Map -> Filter -> Reduce (the canonical governance chain)."""

    def __init__(self) -> None:
        super().__init__()
        self.chunks = mg.Expand(DocToChunks, parent=0, child_label="chunk")
        self.embed = mg.Map(Embed)
        self.keep = mg.Filter(KeepGood)
        self.assemble = mg.Reduce(AssembleChunks)

    def forward(self, docs):
        chunks = self.chunks(docs)
        embeds = self.embed(chunks)
        kept = self.keep(embeds)
        return self.assemble(mg.group_by(docs, kept))


class FanoutPipe(mg.Pipeline):
    """Fanout + same-grain fan-in over plain Map."""

    def __init__(self) -> None:
        super().__init__()
        self.left = mg.Map(Embed, name="left")
        self.right = mg.Map(Embed, name="right")
        self.merge = mg.Map(Embed, name="merge")

    def forward(self, chunks):
        left = self.left(chunks)
        right = self.right(chunks)
        return self.merge(left, right)


class SelectPipe(mg.Pipeline):
    """Expand -> Select (lowers to Map + SelectFilter + Project)."""

    def __init__(self) -> None:
        super().__init__()
        self.chunks = mg.Expand(DocToChunks, parent=0, child_label="chunk")
        self.select = mg.Select(Score, num_annotations=1)

    def forward(self, docs):
        chunks = self.chunks(docs)
        return self.select(chunks)


class RelatePipe(mg.Pipeline):
    """Relate M:N with an explicit relation adapter."""

    def __init__(self) -> None:
        super().__init__()
        self.match = mg.Relate(Match, roles=("image", "caption"), output_grain="pair")

    def forward(self, images, captions):
        return self.match(images, captions)


# ---------------------------------------------------------------------------
# Motif representation matrix: trace + verify + passive + pickle
# ---------------------------------------------------------------------------
def _has_symbolic_port(ir) -> bool:
    for node in ir.nodes:
        for spec in (*node.input_specs, *node.output_specs):
            if isinstance(spec, SymbolicPort):
                return True
    for port in (*ir.inputs, *ir.outputs):
        if isinstance(port, SymbolicPort):
            return True
    return False


MOTIF_PIPELINES = {
    "fanout_fanin_map": FanoutPipe,
    "expand_map_filter_reduce": GovPipe,
    "expand_select_lowering": SelectPipe,
    "relate_m_n": RelatePipe,
}


@pytest.mark.parametrize("name", sorted(MOTIF_PIPELINES))
def test_motif_traces_verifies_and_is_passive_picklable(name: str) -> None:
    ir = MOTIF_PIPELINES[name]().compile()

    assert VerifyPass().run(ir).ok is True
    assert _has_symbolic_port(ir) is False

    reloaded = pickle.loads(pickle.dumps(ir))
    assert reloaded.describe() == ir.describe()
    assert reloaded.to_dict() == ir.to_dict()
    assert "tracer" not in str(ir.to_dict())


# ---------------------------------------------------------------------------
# End-to-end execution over the passive IR
# ---------------------------------------------------------------------------
def test_gov_pipeline_executes_expand_map_filter_reduce() -> None:
    ir = GovPipe().compile()
    docs = mg.source(["a-dropme-c", "x-y"], name="docs")

    out = mg.MultigrainExecutor().execute(ir, {"docs": docs})

    assert out.values == [
        "a-dropme-c=>[emb:a,emb:c]",   # "dropme" chunk filtered out
        "x-y=>[emb:x,emb:y]",
    ]
    assert out.record_ids == docs.record_ids


def test_gov_pipeline_executes_from_pickled_reloaded_ir() -> None:
    """A pickled-then-reloaded IR (no live Pipeline) still drives execution."""
    ir = GovPipe().compile()
    reloaded = pickle.loads(pickle.dumps(ir))
    docs = mg.source(["a-dropme-c", "x-y"], name="docs")

    original = mg.MultigrainExecutor().execute(ir, {"docs": docs})
    detached = mg.MultigrainExecutor().execute(reloaded, {"docs": docs})

    assert detached.values == original.values


def test_gov_pipeline_executes_after_rebatch_transform() -> None:
    ir = GovPipe().compile()
    transformed = InsertRebatchAfterExpandPass().run(ir).graph
    docs = mg.source(["a-dropme-c", "x-y"], name="docs")

    assert [node.kind.value for node in transformed.nodes] == [
        "EXPAND",
        "REBATCH",
        "MAP",
        "FILTER",
        "REDUCE",
    ]
    out = mg.MultigrainExecutor().execute(transformed, {"docs": docs})
    assert out.values == ["a-dropme-c=>[emb:a,emb:c]", "x-y=>[emb:x,emb:y]"]


def test_select_pipeline_executes_lowered_map_filter_project() -> None:
    ir = SelectPipe().compile()
    docs = mg.source(["a-dropme-c"], name="docs")  # chunks: a, dropme, c

    kept_chunks, kept_scores = mg.MultigrainExecutor().execute(ir, {"docs": docs})

    assert kept_chunks.values == ["dropme"]  # only len >= 2 survives
    assert kept_scores.values == [6]


def test_relate_pipeline_executes_with_relation_adapter() -> None:
    ir = RelatePipe().compile()
    images = mg.source(["img0", "img1"], name="image")
    captions = mg.source(["cap0", "cap1"], name="caption")

    pairs = mg.MultigrainExecutor(relation_fns={"Match": match_relation}).execute(
        ir, {"images": images, "captions": captions}
    )

    assert pairs.values == ["img0|cap0", "img1|cap1"]
    assert pairs.display_keys == [
        "image=img0/caption=cap0",
        "image=img1/caption=cap1",
    ]
    assert pairs.relations[0][0].role == "image"
    assert pairs.relations[0][1].role == "caption"


# ---------------------------------------------------------------------------
# Analysis passes still read the passive IR
# ---------------------------------------------------------------------------
def test_analysis_passes_read_gov_pipeline_ir() -> None:
    ir = GovPipe().compile()

    summary = RelationSummaryPass().run(ir).metadata
    assert summary["relation_kinds"]["EXPAND"] == 1
    assert summary["relation_kinds"]["REDUCE"] == 1

    rebatch = RebatchCandidatePass().run(ir).metadata["rebatch_candidates"]
    assert len(rebatch) == 1  # the single Expand output

    plans = PlanReduceGroupsPass().run(ir).metadata["reduce_group_plans"]
    assert plans[0]["anchor"] == "__input__docs"


def test_select_lowering_marks_map_filter_fusion_candidate() -> None:
    ir = SelectPipe().compile()

    candidates = MarkMapFilterFusionCandidatesPass().run(ir).metadata[
        "fusion_candidates"
    ]

    assert candidates
    assert candidates[0]["filter"] == "Score__filter"
