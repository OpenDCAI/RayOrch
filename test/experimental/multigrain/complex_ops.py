"""Importable ops for the *complex* multi-stage bubble experiment.

Pipeline shape (two-level fan-out + data-dependent filter + heterogeneous GPU):

    doc --Expand--> pages --Expand--> blocks --Filter--> dense blocks
        --Map(GPU OCR, cost proportional to block work)--> texts --Reduce--> doc

This deliberately compounds several bubble sources at once:
* two long-tailed fan-outs (pages-per-doc and blocks-per-page),
* a long-tailed per-block work,
* a data-dependent filter that survives an uneven set per shard.

Kept importable so Ray workers can reconstruct the ops from the IR.
"""
from __future__ import annotations

from .gpu_ops import gpu_matmul_work


class DocToPages:
    """Expand 1:N -- doc -> pages. Each doc is {name, pages: [[block_work,...], ...]}."""

    def run(self, docs: list[dict]) -> list[list[dict]]:
        return [
            [
                {"doc": doc["name"], "page": i, "blocks": blocks}
                for i, blocks in enumerate(doc["pages"])
            ]
            for doc in docs
        ]


class PageToBlocks:
    """Expand 1:N -- page -> blocks. Compounds the fan-out long tail."""

    def run(self, pages: list[dict]) -> list[list[dict]]:
        return [
            [
                {"doc": page["doc"], "page": page["page"], "block": j, "work": int(w)}
                for j, w in enumerate(page["blocks"])
            ]
            for page in pages
        ]


class KeepDense:
    """Filter -- drop sparse blocks (data-dependent; survivor set varies per shard)."""

    def __init__(self, threshold: int = 4) -> None:
        self.threshold = threshold

    def run(self, blocks: list[dict]) -> list[bool]:
        return [block["work"] >= self.threshold for block in blocks]


class OcrBlocks:
    """Map (GPU) -- OCR a block; cost proportional to block content ('work')."""

    def run(self, blocks: list[dict]) -> list[str]:
        for block in blocks:
            gpu_matmul_work(block["work"])
        return [f"txt[{b['doc']}#p{b['page']}#b{b['block']}]" for b in blocks]


class AssembleDoc:
    """Reduce N:1 -- surviving blocks grouped back to their document."""

    def run(self, docs: list[dict], grouped: list[list[str]]) -> list[str]:
        return [f"{doc['name']}|blocks={len(group)}" for doc, group in zip(docs, grouped)]


def block_work(block: dict) -> float:
    return float(block["work"])
