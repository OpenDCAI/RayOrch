"""Multi-stage Flash-MinerU on the multigrain IR -- LOCAL, no scheduling.

Goal of this suite (before any Ray/LPT/replay): prove the *new* granularity-
separated MinerU ops are correct with the single-task local executor.

What it proves:
* **page and block are first-class records** -- one flat block batch spans all
  docs/pages, so every block is independently addressable (and later shardable);
* the framework reassembles blocks back to their document in reading order across
  TWO expand levels (doc -> page -> block -> ... -> doc);
* **lineage is a framework capability, not operator code** -- the pure ops never
  touch ids/ordinals, yet ancestors/ordinals are fully populated;
* ops are **value-pure**: permuting the input permutes the output identically
  (id/order independence), even though each op is a stateful, model-holding class;
* the figure<->caption M:N relation works via declarative ``on=``;
* the compiled passive IR survives pickling and still executes (ready for Ray).
"""
from __future__ import annotations

import pickle
import random

from rayorch.experimental import multigrain as mg

from test.experimental.multigrain.mineru_ops import (
    AssembleDoc,
    KeepType,
    LinkFigCap,
    OcrBlock,
    PageToBlocks,
    PdfToPages,
    make_docs,
)


def _docs():
    return mg.source(make_docs(), name="docs", display_key=lambda d: d["name"])


class MinerU(mg.Pipeline):
    """doc --Expand--> pages --Expand--> blocks --Map--> texts --Reduce--> doc."""

    def __init__(self) -> None:
        super().__init__()
        self.to_pages = mg.Expand(PdfToPages, parent=0, child_label="page")
        self.to_blocks = mg.Expand(PageToBlocks, parent=0, child_label="block")
        self.ocr = mg.Map(OcrBlock)
        self.assemble = mg.Reduce(AssembleDoc)

    def forward(self, docs):
        pages = self.to_pages(docs)
        blocks = self.to_blocks(pages)
        texts = self.ocr(blocks)
        return self.assemble(mg.group_by(docs, texts))


class MinerURelate(mg.Pipeline):
    """Figure<->caption join branch: blocks --Filter x2--> Relate(on=page_key)."""

    def __init__(self) -> None:
        super().__init__()
        self.to_pages = mg.Expand(PdfToPages, parent=0, child_label="page")
        self.to_blocks = mg.Expand(PageToBlocks, parent=0, child_label="block")
        self.figs = mg.Filter(KeepType, "figure", name="KeepFigures")
        self.caps = mg.Filter(KeepType, "caption", name="KeepCaptions")
        self.link = mg.Relate(
            LinkFigCap,
            on={"figure": "page_key", "caption": "page_key"},
            output_grain="fig_cap",
        )

    def forward(self, docs):
        pages = self.to_pages(docs)
        blocks = self.to_blocks(pages)
        return self.link(self.figs(blocks), self.caps(blocks))


# ---------------------------------------------------------------------------
# grains are first-class + framework owns lineage
# ---------------------------------------------------------------------------
def test_pages_and_blocks_are_flat_first_class_records() -> None:
    docs = _docs()
    pages = mg.Expand(PdfToPages, parent=0, child_label="page")(docs)
    blocks = mg.Expand(PageToBlocks, parent=0, child_label="block")(pages)

    assert len(pages) == 5  # 2 + 1 + 2
    assert len(blocks) == 11  # 4 + 2 + 5

    # one flat block batch spanning every doc -> each block is independently
    # addressable / shardable, not buried in a per-PDF nested list.
    assert {b["doc"] for b in blocks.values} == {"paper0", "paper1", "paper2"}
    assert [b["type"] for b in blocks.values].count("figure") == 3


def test_framework_tracks_lineage_though_ops_are_pure() -> None:
    docs = _docs()
    pages = mg.Expand(PdfToPages, parent=0, child_label="page")(docs)
    blocks = mg.Expand(PageToBlocks, parent=0, child_label="block")(pages)

    # PageToBlocks/PdfToPages never write ids or ordinals, yet the block knows its
    # full ancestry (doc AND page) and its position within each level.
    anc = blocks.ancestors[0]
    ords = blocks.ordinals[0]
    assert docs.identity_domain in anc  # doc ancestor
    assert pages.identity_domain in anc  # page ancestor too
    assert docs.identity_domain in ords  # page-index ordinal under the doc
    # human-readable trace path built entirely by the framework
    assert blocks.display_keys[0].startswith("paper0/page=0/block=0")


# ---------------------------------------------------------------------------
# two-level reassembly in reading order
# ---------------------------------------------------------------------------
def test_full_pipeline_assembles_each_doc_in_reading_order() -> None:
    out = mg.MultigrainExecutor().execute(MinerU().compile(), {"docs": _docs()})

    md = {value.split("::")[0]: value.split("::")[1] for value in out.values}
    assert out.record_ids == _docs().record_ids  # one row per doc, in doc order
    assert md["paper0"] == (
        "tex<paper0#p0#b0> | fig<paper0#p0#b1> | "
        "cap<paper0#p0#b2> | tex<paper0#p1#b0>"
    )
    assert md["paper1"] == "tex<paper1#p0#b0> | tex<paper1#p0#b1>"
    assert md["paper2"] == (
        "fig<paper2#p0#b0> | cap<paper2#p0#b1> | tex<paper2#p0#b2> | "
        "fig<paper2#p1#b0> | cap<paper2#p1#b1>"
    )


# ---------------------------------------------------------------------------
# value purity: permuting input permutes output identically
# ---------------------------------------------------------------------------
def test_ocr_is_value_pure_under_input_permutation() -> None:
    docs = _docs()
    pages = mg.Expand(PdfToPages, parent=0, child_label="page")(docs)
    blocks = mg.Expand(PageToBlocks, parent=0, child_label="block")(pages)

    texts = mg.Map(OcrBlock)(blocks)

    perm = list(range(len(blocks)))
    random.Random(7).shuffle(perm)
    texts_perm = mg.Map(OcrBlock)(blocks.take(perm))

    # output (and identity) for a block depend ONLY on the block, not its position
    assert texts_perm.values == [texts.values[p] for p in perm]
    assert texts_perm.record_ids == [texts.record_ids[p] for p in perm]


# ---------------------------------------------------------------------------
# figure <-> caption M:N relation
# ---------------------------------------------------------------------------
def test_relate_pairs_figure_and_caption_on_same_page() -> None:
    docs = _docs()
    pages = mg.Expand(PdfToPages, parent=0, child_label="page")(docs)
    blocks = mg.Expand(PageToBlocks, parent=0, child_label="block")(pages)

    figs = mg.Filter(KeepType, "figure", name="KeepFigures")(blocks)
    caps = mg.Filter(KeepType, "caption", name="KeepCaptions")(blocks)
    pairs = mg.Relate(
        LinkFigCap,
        on={"figure": "page_key", "caption": "page_key"},
        output_grain="fig_cap",
    )(figs, caps)

    assert len(pairs) == 3
    assert {value["page_key"] for value in pairs.values} == {
        "paper0#p0",
        "paper2#p0",
        "paper2#p1",
    }


# ---------------------------------------------------------------------------
# passive IR survives pickling and still executes (scheduling readiness)
# ---------------------------------------------------------------------------
def test_pipeline_executes_from_pickled_passive_ir() -> None:
    ir = MinerU().compile()
    reloaded = pickle.loads(pickle.dumps(ir))  # no live objects cross this boundary

    out = mg.MultigrainExecutor().execute(reloaded, {"docs": _docs()})
    assert len(out) == 3
    assert all("::" in value for value in out.values)
