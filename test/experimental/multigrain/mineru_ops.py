"""Granularity-separated MinerU dummy operators (stateful classes; framework owns lineage).

Why this module exists
----------------------
Native Flash-MinerU ops run at *PDF grain*: one row == one PDF, and pages/blocks
live as nested lists inside that row, so a single PDF's pages can never be sharded
/ inferred independently and merged back. These ops fix that: **page and block are
first-class records**. ``Expand`` flattens them into their own grain, so every
block is an independent record that can be sharded across workers and reassembled
by the framework.

Operator contract (matches the rest of the multigrain MVP)
----------------------------------------------------------
* Ops are **stateful classes**: the (dummy) "model" is loaded once in ``__init__``
  and reused across every ``run`` call -- exactly how a real layout/VLM-OCR model
  is loaded once per replica/actor and infers many records. Swap the dummy body
  for a real checkpoint load and the wiring is unchanged.
* Ops are **value-pure / id-independent**: ``run(values) -> outputs`` depends only
  on the input values and the fixed model, never on record ids, ordinals, order,
  or history. Identity / lineage / ordering are the FRAMEWORK's job
  (Expand/Map/Reduce/Relate + PortBatch). This is the property the
  reordering-invariance theorem needs -- NOT statelessness.

CPU + deterministic so correctness is provable with NO scheduling first, then the
exact same classes run under Ray / LPT / partial replay unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Stage ops (doc -> pages -> blocks -> ocr -> assemble)
# ---------------------------------------------------------------------------
class PdfToPages:
    """Expand 1:N -- a PDF fans out into its pages (each page a first-class record).

    Input doc: ``{"name": str, "pages": [page_blocks, ...]}`` where ``page_blocks``
    is a list of block specs. Output page: ``{"doc", "page", "blocks": page_blocks}``.
    """

    def __init__(self, renderer: str = "pdfium") -> None:
        # A real renderer/handle would be initialised here, once per replica.
        self.renderer = renderer

    def run(self, docs: List[dict]) -> List[List[dict]]:
        return [
            [
                {"doc": doc["name"], "page": page_index, "blocks": page_blocks}
                for page_index, page_blocks in enumerate(doc["pages"])
            ]
            for doc in docs
        ]


class PageToBlocks:
    """Expand 1:N -- layout detection turns a page into its blocks.

    Output block is a standalone record carrying everything OCR needs in
    isolation, plus a ``page_key`` used later for the figure<->caption join.
    """

    def __init__(self, model: str = "layout-det", conf: float = 0.5) -> None:
        # A real layout-detection checkpoint loads here, once per replica.
        self.model = model
        self.conf = conf

    def run(self, pages: List[dict]) -> List[List[dict]]:
        blocks_per_page: List[List[dict]] = []
        for page in pages:
            page_key = f"{page['doc']}#p{page['page']}"
            blocks_per_page.append(
                [
                    {
                        "doc": page["doc"],
                        "page": page["page"],
                        "block": block_index,
                        "type": spec["type"],
                        "work": int(spec["work"]),
                        "page_key": page_key,
                    }
                    for block_index, spec in enumerate(page["blocks"])
                ]
            )
        return blocks_per_page


class OcrBlock:
    """Map 1:1 -- OCR/VLM a single block (the expensive, independently-shardable stage).

    Stateful: the (dummy) model is loaded once in ``__init__``. Value-pure: the
    output token is a deterministic function of the block's own content only.
    """

    def __init__(self, model: str = "vlm-ocr", max_tokens: int = 256) -> None:
        # A real VLM/OCR checkpoint loads here, once per replica/actor.
        self.model = model
        self.max_tokens = max_tokens

    def run(self, blocks: List[dict]) -> List[str]:
        return [
            f"{block['type'][:3]}<{block['doc']}#p{block['page']}#b{block['block']}>"
            for block in blocks
        ]


class AssembleDoc:
    """Reduce N:1 -- block texts (framework-restored to reading order) -> one markdown."""

    def __init__(self, sep: str = " | ") -> None:
        self.sep = sep

    def run(self, docs: List[dict], grouped_texts: List[List[str]]) -> List[str]:
        return [
            f"{doc['name']}::{self.sep.join(texts)}"
            for doc, texts in zip(docs, grouped_texts)
        ]


# ---------------------------------------------------------------------------
# Figure <-> caption relation branch (M:N via declarative on= key-join)
# ---------------------------------------------------------------------------
class KeepType:
    """Filter -- keep only blocks of a given type (parameterised, stateful class)."""

    def __init__(self, kind: str) -> None:
        self.kind = kind

    def run(self, blocks: List[dict], *unused: Any) -> List[bool]:
        return [block["type"] == self.kind for block in blocks]


class LinkFigCap:
    """Relate op -- fuse a matched (figure, caption) pair on the same page."""

    def __init__(self) -> None:
        pass

    def run(self, by_role: Dict[str, dict]) -> dict:
        figure, caption = by_role["figure"], by_role["caption"]
        return {
            "page_key": figure["page_key"],
            "figure_block": figure["block"],
            "caption_block": caption["block"],
        }


# ---------------------------------------------------------------------------
# Weight function + deterministic data generator
# ---------------------------------------------------------------------------
def block_work(block: dict) -> float:
    """Per-block GPU cost proxy for work-aware (LPT) sharding later."""
    return float(block["work"])


def make_docs() -> List[dict]:
    """Deterministic corpus with variable pages/doc, blocks/page, and block types.

    Counts (used by the correctness assertions):
      paper0: pages=2, blocks=4  (fig=1, cap=1)
      paper1: pages=1, blocks=2  (fig=0, cap=0)
      paper2: pages=2, blocks=5  (fig=2, cap=2)
      totals: pages=5, blocks=11, same-page figure x caption pairs = 3
    """
    return [
        {
            "name": "paper0",
            "pages": [
                [
                    {"type": "text", "work": 3},
                    {"type": "figure", "work": 5},
                    {"type": "caption", "work": 1},
                ],
                [{"type": "text", "work": 2}],
            ],
        },
        {
            "name": "paper1",
            "pages": [
                [{"type": "text", "work": 4}, {"type": "text", "work": 1}],
            ],
        },
        {
            "name": "paper2",
            "pages": [
                [
                    {"type": "figure", "work": 6},
                    {"type": "caption", "work": 2},
                    {"type": "text", "work": 3},
                ],
                [{"type": "figure", "work": 2}, {"type": "caption", "work": 1}],
            ],
        },
    ]
