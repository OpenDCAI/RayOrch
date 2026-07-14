"""Tier ② (declarative ``on=`` key-join) and tier ③ (by-ref adapter) coverage.

These prove the "three-tier relation model" from
``docs/todos/12-relation-model-three-tiers.md``:

* ②  ``Relate(op, on={role: field})`` -- a cross-branch equi-join declared with a
      single line, no relation code in ``forward``. It attaches lineage that the
      two branches did not share, so a downstream ``Reduce`` can regroup by an
      ancestor that only one branch carried.
* ③  ``Relate(op, relation_adapter="pkg.mod:fn")`` -- a by-reference adapter for
      arbitrary (non equi-join) relations, resolved from a dotted path so the IR
      stays serializable.
"""
from __future__ import annotations

import pickle

from rayorch.experimental import multigrain as mg
from rayorch.experimental.multigrain.execution import MultigrainExecutor
from rayorch.experimental.multigrain.ir import (
    KeyJoinSpec,
    RelatedFrom,
    RelateOp,
)


# ---------------------------------------------------------------------------
# Dummy operators
# ---------------------------------------------------------------------------
class PdfToPages:
    """Expand 1:N -- pdf name -> page dicts that carry their own page_id."""

    _COUNTS = {"a.pdf": 2, "b.pdf": 1}

    def run(self, pdfs: list[str]) -> list[list[dict]]:
        return [
            [
                {"page_id": f"{pdf}#p{i}", "text": f"{pdf}-page{i}"}
                for i in range(self._COUNTS[pdf])
            ]
            for pdf in pdfs
        ]


class LinkFig:
    """Relate op -- combine a matched (page, fig) pair into one linked record."""

    def run(self, by_role: dict) -> dict:
        return {
            "page_id": by_role["page"]["page_id"],
            "fig": by_role["fig"]["fig_id"],
        }


class CountFigsPerDoc:
    """Reduce N:1 -- per pdf, count the figs that linked into it."""

    def run(self, pdfs: list[str], fig_groups: list[list[dict]]) -> list[dict]:
        return [
            {"pdf": pdf, "num_figs": len(group)}
            for pdf, group in zip(pdfs, fig_groups)
        ]


class Match:
    """Relate M:N raw op -- pair up i-th image with i-th caption."""

    def run(self, images: list[str], captions: list[str]) -> list[str]:
        return [f"{image}|{caption}" for image, caption in zip(images, captions)]


# An independent figure stream: NOT expanded from pages, but carries page_id.
def _fig_source() -> mg.PortBatch:
    return mg.source(
        [
            {"fig_id": "f0", "page_id": "a.pdf#p0"},
            {"fig_id": "f1", "page_id": "a.pdf#p0"},
            {"fig_id": "f2", "page_id": "a.pdf#p1"},
            {"fig_id": "f3", "page_id": "b.pdf#p0"},
            {"fig_id": "f4", "page_id": "c.pdf#p9"},  # no page -> inner join drops it
        ],
        name="figs",
        display_key=lambda fig: fig["fig_id"],
    )


# ---------------------------------------------------------------------------
# Tier ②: declarative on= key-join (eager)
# ---------------------------------------------------------------------------
def test_key_join_inner_joins_and_attaches_cross_branch_lineage() -> None:
    pdfs = mg.source(["a.pdf", "b.pdf"], name="pdfs")
    pages = mg.Expand(PdfToPages, parent=0)(pdfs)
    figs = _fig_source()

    linked = mg.Relate(
        LinkFig,
        on={"page": "page_id", "fig": "page_id"},
        output_grain="page_fig",
    )(pages, figs)

    # inner join drops the unmatched fig (c.pdf#p9)
    assert [value["fig"] for value in linked.values] == ["f0", "f1", "f2", "f3"]

    # cross-branch lineage: the joined row now knows its pdf via the page side,
    # even though the fig stream never carried pdf ancestry.
    assert linked.ancestors[0]["pdfs"] == pdfs.record_ids[0]

    # relation evidence carries both roles, human-readable, no internal ids leaked.
    # (page display_key is the lineage path built by Expand; fig uses its display_key.)
    assert [(ref.role, ref.display_key) for ref in linked.relations[0]] == [
        ("page", "a.pdf/PdfToPages=0"),
        ("fig", "f0"),
    ]


def test_key_join_is_truly_many_to_many_when_both_roles_repeat_a_key() -> None:
    pages = mg.source(
        [
            {"page_id": "shared", "page": 0},
            {"page_id": "shared", "page": 1},
        ],
        name="pages",
    )
    figs = mg.source(
        [
            {"page_id": "shared", "fig_id": "f0"},
            {"page_id": "shared", "fig_id": "f1"},
        ],
        name="figs",
    )

    linked = mg.Relate(
        LinkFig,
        on={"page": "page_id", "fig": "page_id"},
    )(pages, figs)

    assert len(linked) == 4
    assert {
        (value["page_id"], value["fig"])
        for value in linked.values
    } == {("shared", "f0"), ("shared", "f1")}
    assert len(set(linked.record_ids)) == 4


def test_key_join_then_reduce_regroups_by_ancestor_only_one_branch_had() -> None:
    pdfs = mg.source(["a.pdf", "b.pdf"], name="pdfs")
    pages = mg.Expand(PdfToPages, parent=0)(pdfs)
    figs = _fig_source()

    linked = mg.Relate(
        LinkFig, on={"page": "page_id", "fig": "page_id"}, output_grain="page_fig"
    )(pages, figs)
    result = mg.Reduce(CountFigsPerDoc, name="count")(mg.group_by(pdfs, linked))

    counts = {value["pdf"]: value["num_figs"] for value in result.values}
    assert counts == {"a.pdf": 3, "b.pdf": 1}


# ---------------------------------------------------------------------------
# Tier ②: compile -> passive IR -> pickle -> execute end to end
# ---------------------------------------------------------------------------
class LinkPipe(mg.Pipeline):
    """forward() stays torch-clean: the join semantics live in __init__."""

    def __init__(self) -> None:
        super().__init__()
        self.split = mg.Expand(PdfToPages, parent=0)
        self.link = mg.Relate(
            LinkFig, on={"page": "page_id", "fig": "page_id"}, output_grain="page_fig"
        )
        self.count = mg.Reduce(CountFigsPerDoc, name="count")

    def forward(self, pdfs, figs):
        pages = self.split(pdfs)
        linked = self.link(pages, figs)
        return self.count(mg.group_by(pdfs, linked))


def test_key_join_pipeline_compiles_verifies_and_carries_on_in_ir() -> None:
    ir = LinkPipe().compile()

    relate = ir.node("LinkFig")
    assert isinstance(relate.operation, RelateOp)
    relation = relate.outputs[0].relation
    assert isinstance(relation, RelatedFrom)
    assert tuple(binding.role for binding in relation.roles) == ("page", "fig")
    assert isinstance(relate.operation.matcher, KeyJoinSpec)
    assert dict(relate.operation.matcher.fields) == {
        "page": "page_id",
        "fig": "page_id",
    }


def test_key_join_pipeline_executes_from_pickled_ir() -> None:
    ir = LinkPipe().compile()
    reloaded = pickle.loads(pickle.dumps(ir))  # no live objects survive here

    pdfs = mg.source(["a.pdf", "b.pdf"], name="pdfs")
    figs = _fig_source()

    out = MultigrainExecutor().execute(reloaded, {"pdfs": pdfs, "figs": figs})

    counts = {value["pdf"]: value["num_figs"] for value in out.values}
    assert counts == {"a.pdf": 3, "b.pdf": 1}


# ---------------------------------------------------------------------------
# Tier ③: by-ref adapter resolves from a dotted path
# ---------------------------------------------------------------------------
def test_relate_adapter_by_ref_resolves_dotted_path() -> None:
    images = mg.source(["img0", "img1"], name="image")
    captions = mg.source(["cap0", "cap1"], name="caption")

    pairs = mg.Relate(
        Match,
        roles=("image", "caption"),
        output_grain="pair",
        relation_adapter="test.experimental.multigrain.relate_adapters:link_by_index",
    )(images, captions)

    assert pairs.values == ["img0|cap0", "img1|cap1"]
    assert pairs.display_keys == ["image=img0/caption=cap0", "image=img1/caption=cap1"]
    assert [ref.role for ref in pairs.relations[0]] == ["image", "caption"]


def test_relate_adapter_is_stored_as_typed_dotted_path() -> None:
    relate = mg.Relate(
        Match,
        roles=("image", "caption"),
        relation_adapter="test.experimental.multigrain.relate_adapters:link_by_index",
    )
    assert relate.relation_adapter == (
        "test.experimental.multigrain.relate_adapters:link_by_index"
    )
