"""A lost descendant must cascade to its anchor under FAIL_CLOSED.

Guards the production concern: when a page is permanently quarantined, the
document it belonged to must not be assembled into a silently-truncated output.
Under ``MissingChildPolicy.FAIL_CLOSED`` the Reduce suppresses that document (its
UDF is never called) and emits an anchor-grain error; other documents are
unaffected. Under FAIL_OPEN the legacy best-effort behaviour is preserved.
"""
from __future__ import annotations

import rayorch.experimental.multigrain as mg
from rayorch.experimental.multigrain.executor import MultigrainExecutor
from rayorch.experimental.multigrain.graph import MissingChildPolicy

from test.experimental.multigrain.lineage_ops import (
    AssembleDoc,
    BOOM_DOC,
    BOOM_PAGE,
    EmbedPage,
    SplitPages,
)


class DocPipe(mg.Pipeline):
    def __init__(self, missing: MissingChildPolicy) -> None:
        super().__init__()
        self.split = mg.Expand(SplitPages, parent=0, child_label="page")
        self.embed = mg.Map(EmbedPage)
        self.assemble = mg.Reduce(AssembleDoc, missing_child=missing)

    def forward(self, docs):
        pages = self.split(docs)
        emb = self.embed(pages)
        return self.assemble(mg.group_by(docs, emb))


def _docs():
    return mg.source(
        [
            {"name": "d01", "page_works": [1, 1]},
            {"name": BOOM_DOC, "page_works": [1, 1, 1, 1]},  # page 2 fails in embed
            {"name": "d02", "page_works": [1, 1, 1]},
        ],
        name="docs",
        display_key=lambda d: d["name"],
    )


def test_fail_closed_suppresses_only_the_affected_document():
    ir = DocPipe(MissingChildPolicy.FAIL_CLOSED).compile()
    out = MultigrainExecutor().execute(ir, {"docs": _docs()})

    by_doc = dict(zip(out.display_keys, out.values))
    # Healthy docs assemble normally.
    assert by_doc["d01"] == "d01=[emb(d01#p0)|emb(d01#p1)]"
    assert by_doc["d02"] == "d02=[emb(d02#p0)|emb(d02#p1)|emb(d02#p2)]"
    # The poisoned doc is suppressed: a placeholder, NOT a truncated string.
    assert isinstance(by_doc[BOOM_DOC], dict)
    assert by_doc[BOOM_DOC]["status"] == "incomplete"
    assert f"{BOOM_DOC}/page={BOOM_PAGE}" in by_doc[BOOM_DOC]["lost"][0]

    actions = {e.action for e in out.errors}
    assert "quarantined" in actions  # the page-level failure
    assert "suppressed_incomplete" in actions  # cascaded to the document
    doc_err = next(e for e in out.errors if e.action == "suppressed_incomplete")
    assert doc_err.grain == "docs"
    assert doc_err.logical_item == BOOM_DOC


def test_fail_open_still_assembles_from_survivors():
    ir = DocPipe(MissingChildPolicy.FAIL_OPEN).compile()
    out = MultigrainExecutor().execute(ir, {"docs": _docs()})

    by_doc = dict(zip(out.display_keys, out.values))
    # Best-effort: the poisoned doc assembles from the pages that survived.
    assert by_doc[BOOM_DOC] == f"{BOOM_DOC}=[emb({BOOM_DOC}#p0)|emb({BOOM_DOC}#p1)|emb({BOOM_DOC}#p3)]"
    actions = {e.action for e in out.errors}
    assert "quarantined" in actions
    assert "suppressed_incomplete" not in actions
