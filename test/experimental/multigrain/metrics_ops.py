"""Importable dummy operators + auto data generators for the M2 metric tests.

CPU-only, no torch/GPU: ``SleepMap`` burns wall time proportional to a per-row
``work`` field so an imbalanced 1:N fan-out produces a *real* idle bubble that the
instrumentation can measure and LPT can shrink. Auto-generates both a tabular and
an image-like source so the tests never touch real data.

Kept in a real module (not a pytest file) so Ray workers can rebuild the
operators from ``OperatorRecipe.cls_ref`` in the passive IR.
"""
from __future__ import annotations

import random
import time

WORK_UNIT_S = 0.01  # seconds of sleep per unit of work


class MakeRows:
    """Expand 1:N -- a source doc/image fans out into variable-work child rows."""

    def run(self, docs: list[dict]) -> list[list[dict]]:
        groups: list[list[dict]] = []
        for doc in docs:
            groups.append(
                [
                    {"src": doc["name"], "row": i, "work": int(w)}
                    for i, w in enumerate(doc["works"])
                ]
            )
        return groups


class SleepMap:
    """Map 1:1 -- spend ``work * WORK_UNIT_S`` seconds, emit a token.

    The sleep makes shard busy-time proportional to assigned work, so a lopsided
    contiguous split leaves fast shards idle (a measurable bubble) while LPT
    balances total work per shard.
    """

    def run(self, rows: list[dict]) -> list[str]:
        out: list[str] = []
        for row in rows:
            time.sleep(row["work"] * WORK_UNIT_S)
            out.append(f"emb({row['src']}#r{row['row']})")
        return out


class Assemble:
    """Reduce N:1 -- regroup child tokens back into their source, in order."""

    def run(self, docs: list[dict], grouped: list[list[str]]) -> list[str]:
        return [
            f"{doc['name']}=[{'|'.join(group)}]"
            for doc, group in zip(docs, grouped)
        ]


def row_work(row: dict) -> float:
    return float(row["work"])


def make_tabular_docs(seed: int = 0) -> list[dict]:
    """Auto-generated tabular sources with a long-tailed per-row work fan-out."""
    rng = random.Random(seed)
    docs: list[dict] = []
    for i in range(6):
        n = rng.randint(1, 6)
        works = [max(1, int(rng.paretovariate(1.3))) for _ in range(n)]
        docs.append({"name": f"tab{i:02d}", "works": works})
    return docs


def make_image_docs(seed: int = 1) -> list[dict]:
    """Auto-generated image-like sources: work == number of detected regions."""
    rng = random.Random(seed)
    docs: list[dict] = []
    for i in range(6):
        n = rng.randint(1, 5)
        # each 'region' costs work proportional to its (fake) pixel area
        works = [max(1, int(rng.gauss(6, 4))) for _ in range(n)]
        docs.append({"name": f"img{i:02d}", "works": works})
    return docs
