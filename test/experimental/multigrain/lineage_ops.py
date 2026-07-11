"""Importable CPU dummy operators for the lineage-under-parallelism experiment.

Kept in a real module (not a pytest file) so Ray workers can reconstruct the
operators from ``OperatorRecipe.cls_ref`` in the IR. Pages carry a ``work`` field
purely so a work-aware LPT planner produces a *non-trivial* cross-shard
permutation -- that is what stresses the lineage/ordinal machinery.
"""
from __future__ import annotations

from rayorch.runtime import BadRecordError

BOOM_DOC = "d03"  # the document whose page BOOM_PAGE fails during Map
BOOM_PAGE = 2


class SplitPages:
    """Expand 1:N -- doc dict -> variable-length group of page dicts."""

    def run(self, docs: list[dict]) -> list[list[dict]]:
        groups: list[list[dict]] = []
        for doc in docs:
            groups.append(
                [
                    {"doc": doc["name"], "page": i, "work": int(w)}
                    for i, w in enumerate(doc["page_works"])
                ]
            )
        return groups


class EmbedPage:
    """Map 1:1 -- embed a page; quarantine one designated page via BadRecordError.

    ``index`` is the position within the rows this invocation actually received
    (i.e. within a Ray shard), which is exactly what the row-isolation path
    expects; the framework maps it back to the correct global record identity.
    """

    def run(self, pages: list[dict]) -> list[str]:
        for index, page in enumerate(pages):
            if page["doc"] == BOOM_DOC and page["page"] == BOOM_PAGE:
                raise BadRecordError("embed failed on page", index=index)
        return [f"emb({page['doc']}#p{page['page']})" for page in pages]


class EmbedPageOk:
    """Map 1:1 -- same as EmbedPage but never fails (identity/ordinal checks)."""

    def run(self, pages: list[dict]) -> list[str]:
        return [f"emb({page['doc']}#p{page['page']})" for page in pages]


class AssembleDoc:
    """Reduce N:1 -- regroup page embeddings back into their document, in order."""

    def run(self, docs: list[dict], grouped: list[list[str]]) -> list[str]:
        return [
            f"{doc['name']}=[{'|'.join(group)}]"
            for doc, group in zip(docs, grouped)
        ]


def page_work(page: dict) -> float:
    return float(page["work"])
