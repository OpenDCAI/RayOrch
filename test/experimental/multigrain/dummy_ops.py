"""Importable dummy operators for multigrain Ray tests.

These live in a real importable module (not a pytest-collected test file) so that
Ray workers can reconstruct them from the ``OperatorRecipe.cls_ref`` stored in
the IR. Operators sleep per row so that parallelism shows up in wall-clock time.
"""
from __future__ import annotations

import time

SLEEP = 0.2


class SlowEmbed:
    """Map op: sleep per row, then embed."""

    def run(self, chunks: list[str]) -> list[str]:
        for _ in chunks:
            time.sleep(SLEEP)
        return [f"emb:{chunk}" for chunk in chunks]


class SlowDrop:
    """Filter op: sleep per row, drop chunks containing 'x'."""

    def run(self, chunks: list[str]) -> list[bool]:
        for _ in chunks:
            time.sleep(SLEEP)
        return ["x" not in chunk for chunk in chunks]
